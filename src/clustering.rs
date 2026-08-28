use linfa::prelude::*;
use linfa_clustering::Dbscan;
use ndarray::Array2;
use std::collections::HashMap;

use crate::http_client::ResponseFeatures;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Cluster {
    pub id: usize,
    pub features: Vec<ResponseFeatures>,
    pub centroid: ClusterCentroid,
    pub representative_response: Option<usize>,
    pub sample_payloads: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClusterCentroid {
    pub mode_status_code: u16,
    pub avg_response_length: f64,
    pub avg_ttfb_ms: f64,
    pub avg_response_words: f64,
    pub avg_response_lines: f64,
    pub avg_content_length_minus_payload: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DbscanResult {
    pub clusters: Vec<Cluster>,
    pub outliers: Vec<usize>,
    pub noise_count: usize,
}

const IANA_STATUS_CODES: &[u16] = &[
    100, 101, 102, 103, 104, 200, 201, 202, 203, 204, 205, 206, 207, 208, 226, 300, 301, 302, 303,
    304, 305, 306, 307, 308, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413,
    414, 415, 416, 417, 418, 421, 422, 423, 424, 425, 426, 428, 429, 431, 451, 500, 501, 502, 503,
    504, 505, 506, 507, 508, 510, 511,
];

const NUM_STATUS_CODES: usize = IANA_STATUS_CODES.len();
const UNKNOWN_COL: usize = NUM_STATUS_CODES;
const NUM_STATUS_FEATURES: usize = NUM_STATUS_CODES + 1;

const CONTINUOUS_FEATURES: usize = 5;
/// Number of continuous features when timing analysis is disabled (TTFB excluded).
const CONTINUOUS_FEATURES_NO_TIMING: usize = 4;
/// Offset of the first continuous column relative to `NUM_STATUS_FEATURES`.
///
/// Continuous columns are laid out densely (length, [ttfb], words, lines,
/// clmp); the TTFB column is omitted when timing analysis is disabled, so
/// only the first offset is a fixed constant.
const COL_LENGTH: usize = 0;

fn status_one_hot_index(code: u16) -> usize {
    match IANA_STATUS_CODES.binary_search(&code) {
        Ok(i) => i,
        Err(_) => UNKNOWN_COL,
    }
}

/// Feature scaler using StandardScaler (z-score: (value - mean) / std).
/// Chosen over log1p because DBSCAN's fixed tolerance needs a consistent
/// statistical scale (radius = ~1 sigma), and z-score keeps extreme responses
/// far from the bulk so anomalies are flagged as noise. Log1p compresses
/// large values, pulling outliers toward the main cluster and
/// causing false negatives.
struct Scaler {
    mean: Vec<f64>,
    std: Vec<f64>,
}

impl Scaler {
    fn fit(data: &Array2<f64>, n_status: usize) -> Self {
        let ncols = data.ncols();
        let nrows = data.nrows() as f64;
        let mut mean = vec![0.0f64; ncols];
        let mut std = vec![0.0f64; ncols];

        for j in n_status..ncols {
            let col_sum: f64 = data.column(j).sum();
            mean[j] = col_sum / nrows;
            let var_sum: f64 = data.column(j).iter().map(|&v| (v - mean[j]).powi(2)).sum();
            std[j] = (var_sum / nrows).sqrt();
        }

        Scaler { mean, std }
    }

    fn transform(&self, data: &mut Array2<f64>, n_status: usize) {
        let ncols = data.ncols();
        for j in n_status..ncols {
            if self.std[j] == 0.0 {
                for i in 0..data.nrows() {
                    data[[i, j]] = 0.0;
                }
            } else {
                for i in 0..data.nrows() {
                    data[[i, j]] = (data[[i, j]] - self.mean[j]) / self.std[j];
                }
            }
        }
    }
}

pub fn perform_clustering(
    features: &[ResponseFeatures],
    tolerance: f64,
    min_samples: usize,
    max_clusters: usize,
    include_timing: bool,
) -> DbscanResult {
    if features.is_empty() {
        return DbscanResult {
            clusters: Vec::new(),
            outliers: Vec::new(),
            noise_count: 0,
        };
    }

    let (mut data, original_indices) = features_to_array(features, include_timing);

    let scaler = Scaler::fit(&data, NUM_STATUS_FEATURES);
    scaler.transform(&mut data, NUM_STATUS_FEATURES);

    let dataset = DatasetBase::from(data);

    let cluster_memberships = Dbscan::params(min_samples)
        .tolerance(tolerance)
        .transform(dataset)
        .expect("DBSCAN clustering failed");

    let mut cluster_map: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut outliers = Vec::new();

    for (i, cluster_id) in cluster_memberships.targets().iter().enumerate() {
        match cluster_id {
            None => {
                outliers.push(original_indices[i]);
            }
            Some(id) => {
                cluster_map
                    .entry(*id)
                    .or_default()
                    .push(original_indices[i]);
            }
        }
    }

    let mut cluster_results: Vec<(usize, Vec<usize>)> = cluster_map.into_iter().collect();
    cluster_results.sort_by_key(|b| std::cmp::Reverse(b.1.len()));

    if cluster_results.len() > max_clusters {
        let excess = cluster_results.split_off(max_clusters);
        for (_, indices) in excess {
            outliers.extend(indices);
        }
        outliers.sort();
    }

    let noise_count = outliers.len();

    let mut final_clusters = Vec::new();
    for (new_id, (_orig_label, indices)) in cluster_results.into_iter().enumerate() {
        let cluster_features: Vec<ResponseFeatures> =
            indices.iter().map(|&i| features[i].clone()).collect();
        let centroid = calculate_centroid(&cluster_features);
        let representative =
            find_representative(&cluster_features, &centroid, &scaler, include_timing);
        let sample_payloads = collect_sample_payloads(&indices, features, 5);

        final_clusters.push(Cluster {
            id: new_id,
            features: cluster_features,
            centroid,
            representative_response: representative,
            sample_payloads,
        });
    }

    DbscanResult {
        clusters: final_clusters,
        outliers,
        noise_count,
    }
}

/// Fuzzing-stage payloads for display: sample payloads from the cluster,
/// excluding baseline entries (the first `n_baseline` features, which carry
/// the original parameter value rather than a fuzz payload).
fn collect_sample_payloads(
    indices: &[usize],
    features: &[ResponseFeatures],
    max_samples: usize,
) -> Vec<String> {
    indices
        .iter()
        .filter_map(|&i| features[i].payload.clone())
        .take(max_samples)
        .collect()
}

fn features_to_array(
    features: &[ResponseFeatures],
    include_timing: bool,
) -> (Array2<f64>, Vec<usize>) {
    let n_continuous = if include_timing {
        CONTINUOUS_FEATURES
    } else {
        CONTINUOUS_FEATURES_NO_TIMING
    };
    let n_features = NUM_STATUS_FEATURES + n_continuous;
    let n_samples = features.len();

    let mut data = Array2::zeros((n_samples, n_features));
    let mut original_indices = Vec::with_capacity(n_samples);

    for (i, feature) in features.iter().enumerate() {
        let hot = status_one_hot_index(feature.status_code);
        data[[i, hot]] = 1.0;

        let base = NUM_STATUS_FEATURES;
        data[[i, base + COL_LENGTH]] = feature.response_length as f64;
        // Continuous columns are laid out densely; the TTFB column is omitted
        // entirely when timing analysis is disabled.
        let mut col = base + COL_LENGTH + 1;
        if include_timing {
            data[[i, col]] = feature.time_to_first_byte_ms as f64;
            col += 1;
        }
        data[[i, col]] = feature.response_words as f64;
        col += 1;
        data[[i, col]] = feature.response_lines as f64;
        col += 1;
       // data[[i, col]] = feature.content_length_minus_payload as f64;
        data[[i, col]] = 0 as f64;
        original_indices.push(i);
    }

    (data, original_indices)
}

fn calculate_centroid(features: &[ResponseFeatures]) -> ClusterCentroid {
    let mut counts: HashMap<u16, usize> = HashMap::new();
    for f in features {
        *counts.entry(f.status_code).or_insert(0) += 1;
    }
    let mode_status_code = counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(code, _)| code)
        .unwrap_or(0);

    let n = features.len() as f64;

    ClusterCentroid {
        mode_status_code,
        avg_response_length: features
            .iter()
            .map(|f| f.response_length as f64)
            .sum::<f64>()
            / n,
        avg_ttfb_ms: features
            .iter()
            .map(|f| f.time_to_first_byte_ms as f64)
            .sum::<f64>()
            / n,
        avg_response_words: features
            .iter()
            .map(|f| f.response_words as f64)
            .sum::<f64>()
            / n,
        avg_response_lines: features
            .iter()
            .map(|f| f.response_lines as f64)
            .sum::<f64>()
            / n,
        avg_content_length_minus_payload: features
            .iter()
            .map(|f| f.content_length_minus_payload as f64)
            .sum::<f64>()
            / n,
    }
}

fn find_representative(
    features: &[ResponseFeatures],
    centroid: &ClusterCentroid,
    scaler: &Scaler,
    include_timing: bool,
) -> Option<usize> {
    if features.is_empty() {
        return None;
    }

    let hot = status_one_hot_index(centroid.mode_status_code);

    let base = NUM_STATUS_FEATURES;
    // Must match the dense column layout used by `features_to_array`.
    let centroid_vals: Vec<f64> = if include_timing {
        vec![
            centroid.avg_response_length,
            centroid.avg_ttfb_ms,
            centroid.avg_response_words,
            centroid.avg_response_lines,
            centroid.avg_content_length_minus_payload,
        ]
    } else {
        vec![
            centroid.avg_response_length,
            centroid.avg_response_words,
            centroid.avg_response_lines,
            centroid.avg_content_length_minus_payload,
        ]
    };

    let n = centroid_vals.len();
    let mut scaled_centroid = vec![0.0f64; n];
    for j in 0..n {
        if scaler.std[base + j] == 0.0 {
            scaled_centroid[j] = 0.0;
        } else {
            scaled_centroid[j] = (centroid_vals[j] - scaler.mean[base + j]) / scaler.std[base + j];
        }
    }

    let mut min_dist = f64::MAX;
    let mut representative = 0;

    for (i, feature) in features.iter().enumerate() {
        let code_hot = status_one_hot_index(feature.status_code);
        let status_diff = if code_hot == hot { 0.0f64 } else { 1.0f64 };

        let raw: Vec<f64> = if include_timing {
            vec![
                feature.response_length as f64,
                feature.time_to_first_byte_ms as f64,
                feature.response_words as f64,
                feature.response_lines as f64,
                feature.content_length_minus_payload as f64,
            ]
        } else {
            vec![
                feature.response_length as f64,
                feature.response_words as f64,
                feature.response_lines as f64,
                feature.content_length_minus_payload as f64,
            ]
        };

        let mut dist_sq = status_diff.powi(2);
        for j in 0..n {
            let scaled = if scaler.std[base + j] == 0.0 {
                0.0
            } else {
                (raw[j] - scaler.mean[base + j]) / scaler.std[base + j]
            };
            dist_sq += (scaled - scaled_centroid[j]).powi(2);
        }

        let dist = dist_sq.sqrt();
        if dist < min_dist {
            min_dist = dist;
            representative = i;
        }
    }

    Some(representative)
}

pub fn analyze_clusters(result: &DbscanResult) -> Vec<String> {
    let mut observations = Vec::new();

    if result.clusters.len() == 1 && result.noise_count == 0 {
        observations.push(
            "All responses belong to a single cluster - responses are consistent.".to_string(),
        );
    } else {
        observations.push(format!(
            "Found {} distinct response clusters.",
            result.clusters.len()
        ));

        for cluster in &result.clusters {
            observations.push(format!(
                "Cluster {}: {} responses (status: {:>3}, length: {:>6}, TTFB: {:>4}ms)",
                cluster.id,
                cluster.features.len(),
                cluster.centroid.mode_status_code,
                cluster.centroid.avg_response_length as usize,
                cluster.centroid.avg_ttfb_ms as u64,
            ));
            for payload in &cluster.sample_payloads {
                observations.push(format!("    Sample payload: '{}'", payload));
            }
        }
    }

    if result.noise_count > 0 {
        observations.push(format!(
            "Found {} outlier(s) that don't fit any cluster.",
            result.noise_count
        ));
    }

    observations
}
