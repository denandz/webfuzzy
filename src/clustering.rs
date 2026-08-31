use linfa::prelude::*;
use linfa_clustering::Dbscan;
use ndarray::Array2;
use std::collections::{BTreeMap, HashMap};

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
    /// Status families where reflection was detected: (status, Pearson r).
    #[serde(default)]
    pub reflection: Vec<(u16, f64)>,
}

/// Number of continuous features - length, ttfb, words, lines
const CONTINUOUS_FEATURES: usize = 4;
/// Number of continuous features when timing analysis is disabled (TTFB excluded).
const CONTINUOUS_FEATURES_NO_TIMING: usize = 3;
/// Pearson r between payload length and response length required to treat
/// the payload as reflected in the response.
const REFLECTION_CORRELATION_THRESHOLD: f64 = 0.8;
/// Feature scaler using StandardScaler (z-score: (value - mean) / std).
/// Chosen over log1p because DBSCAN's fixed tolerance needs a consistent
/// scale, and z-score keeps extreme responses far from the bulk so
/// anomalies are flagged as noise; log1p pulls outliers toward the main
/// cluster.
struct Scaler {
    mean: Vec<f64>,
    std: Vec<f64>,
}

impl Scaler {
    fn fit(data: &Array2<f64>) -> Self {
        let ncols = data.ncols();
        let nrows = data.nrows() as f64;
        let mut mean = vec![0.0f64; ncols];
        let mut std = vec![0.0f64; ncols];

        for j in 0..ncols {
            let col_sum: f64 = data.column(j).sum();
            mean[j] = col_sum / nrows;
            let var_sum: f64 = data.column(j).iter().map(|&v| (v - mean[j]).powi(2)).sum();
            std[j] = (var_sum / nrows).sqrt();
        }

        Scaler { mean, std }
    }

    fn transform(&self, data: &mut Array2<f64>) {
        let ncols = data.ncols();
        for j in 0..ncols {
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

    /// Override the (mean, std) used for one column.
    fn set_scale(&mut self, col: usize, mean: f64, std: f64) {
        self.mean[col] = mean;
        self.std[col] = std;
    }
}

/// (mode status, mean, sigma) of the baseline TTFBs, with sigma floored to
/// `jitter_floor_ms`. None with fewer than 2 samples.
pub fn ttfb_baseline_ref(
    status_codes: &HashMap<u16, usize>,
    ttfbs: &[u64],
    jitter_floor_ms: f64,
) -> Option<(u16, f64, f64)> {
    let status = *status_codes.iter().max_by_key(|(_, c)| *c)?.0;
    if ttfbs.len() < 2 {
        return None;
    }
    let mean = ttfbs.iter().map(|t| *t as f64).sum::<f64>() / ttfbs.len() as f64;
    let variance = ttfbs
        .iter()
        .map(|t| {
            let d = *t as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / (ttfbs.len() - 1) as f64;
    let sigma = variance.sqrt();
    Some((status, mean, sigma.max(jitter_floor_ms)))
}

pub fn perform_clustering(
    features: &[ResponseFeatures],
    tolerance: f64,
    min_samples: usize,
    max_clusters: usize,
    include_timing: bool,
    ttfb_ref: Option<(u16, f64, f64)>,
    jitter_ms: f64,
) -> DbscanResult {
    if features.is_empty() {
        return DbscanResult {
            clusters: Vec::new(),
            outliers: Vec::new(),
            noise_count: 0,
            reflection: Vec::new(),
        };
    }

    // Cluster per HTTP status code in it's own scaled space: a 400s
    // response length must not influence the distance between 200s.
    // Groups hold global feature indices and BTree map means deterministic
    // status order.
    let features_by_status: BTreeMap<u16, Vec<usize>> = features
        .iter()
        .enumerate()
        .fold(BTreeMap::new(), |mut acc, (i, f)| {
            acc.entry(f.status_code).or_default().push(i);
            acc
        });

    let mut final_clusters = Vec::new();
    let mut cluster_members: Vec<Vec<usize>> = Vec::new(); // parallel to final_clusters
    let mut outliers: Vec<usize> = Vec::new();
    let mut reflection: Vec<(u16, f64)> = Vec::new();
    for (status, global_indices) in features_by_status.iter() {
        // Get all the features for a given status code
        let group_feats: Vec<&ResponseFeatures> =
            global_indices.iter().map(|&i| &features[i]).collect();
        // Reflected payloads make the response length track the payload
        // length, fragmenting the cluster; regress it out when detected.
        let regression = reflection_regression(&group_feats);
        if let Some((r, _slope)) = regression {
            reflection.push((*status, r));
        }
        let mut data = features_to_array(&group_feats, include_timing, regression);

        let mut scaler = Scaler::fit(&data);
        if include_timing {
            // TTFB column: when this family is the baseline's, scale against
            // the baseline's own jitter so normal endpoint wobble is absorbed
            // and only real timing anomalies stand out.
            if let Some((ref_status, ref_mean, ref_sigma)) = ttfb_ref {
                if ref_status == *status {
                    scaler.set_scale(1, ref_mean, ref_sigma);
                }
            }
            // Set the scaler's TTFB stddev to the jitter floor so millisecond
            // wobble isn't flagged as timing outliers. No-ops when the data
            // is naturally more variable than the floor.
            scaler.std[1] = scaler.std[1].max(jitter_ms);
        }
        scaler.transform(&mut data);

        let dataset = DatasetBase::from(data);

        let cluster_memberships = Dbscan::params(min_samples)
            .tolerance(tolerance)
            .transform(dataset)
            .expect("DBSCAN clustering failed");

        let mut cluster_map: HashMap<usize, Vec<usize>> = HashMap::new();

        for (i, cluster_id) in cluster_memberships.targets().iter().enumerate() {
            match cluster_id {
                None => outliers.push(global_indices[i]),
                Some(id) => cluster_map.entry(*id).or_default().push(global_indices[i]),
            }
        }

        let mut cluster_results: Vec<(usize, Vec<usize>)> = cluster_map.into_iter().collect();
        cluster_results.sort_by_key(|b| std::cmp::Reverse(b.1.len()));

        for (_orig_label, indices) in cluster_results {
            let cluster_features: Vec<ResponseFeatures> =
                indices.iter().map(|&i| features[i].clone()).collect();
            let centroid = calculate_centroid(&cluster_features);
            let representative = find_representative(
                &cluster_features,
                &centroid,
                &scaler,
                include_timing,
                regression,
            );
            let sample_payloads = collect_sample_payloads(&indices, features, 5);

            final_clusters.push(Cluster {
                id: 0, // re-assigned below after global ordering
                features: cluster_features,
                centroid,
                representative_response: representative,
                sample_payloads,
            });
            cluster_members.push(indices);
        }
    }

    // Order clusters globally, largest first, so Cluster 0 is the dominant one.
    let mut order: Vec<usize> = (0..final_clusters.len()).collect();
    order.sort_by(|&a, &b| final_clusters[b].features.len().cmp(&final_clusters[a].features.len()));

    // Global max_clusters cap: demote the smallest clusters to outliers.
    if order.len() > max_clusters {
        for &c in order.iter().skip(max_clusters) {
            outliers.extend(cluster_members[c].iter().copied());
        }
        order.truncate(max_clusters);
    }

    outliers.sort();
    let noise_count = outliers.len();

    let mut slots: Vec<Option<Cluster>> = final_clusters.into_iter().map(Some).collect();
    let final_clusters: Vec<Cluster> = order
        .into_iter()
        .enumerate()
        .map(|(new_id, c)| {
            let mut cluster = slots[c].take().expect("cluster index taken twice");
            cluster.id = new_id;
            cluster
        })
        .collect();

    DbscanResult {
        clusters: final_clusters,
        outliers,
        noise_count,
        reflection,
    }
}

/// Sample payloads from a cluster
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

/// Byte length of the raw payload; 0 when the request carried none.
fn payload_len(feature: &ResponseFeatures) -> f64 {
    feature.payload.as_ref().map(|p| p.len() as f64).unwrap_or(0.0)
}

/// (Pearson r, slope) of response length regressed on payload length, when
/// r exceeds the reflection threshold. The slope covers decoders too, where
/// the reflected length is a fraction of the payload length.
fn reflection_regression(features: &[&ResponseFeatures]) -> Option<(f64, f64)> {
    let pairs: Vec<(f64, f64)> = features
        .iter()
        .map(|f| (payload_len(f), f.response_length as f64))
        .collect();
    let n = pairs.len() as f64;
    if n < 2.0 {
        return None;
    }
    let mean_x = pairs.iter().map(|&(x, _)| x).sum::<f64>() / n;
    let mean_y = pairs.iter().map(|&(_, y)| y).sum::<f64>() / n;
    let var_x = pairs.iter().map(|&(x, _)| (x - mean_x).powi(2)).sum::<f64>() / n;
    let var_y = pairs.iter().map(|&(_, y)| (y - mean_y).powi(2)).sum::<f64>() / n;
    if var_x == 0.0 || var_y == 0.0 {
        return None;
    }
    let cov = pairs.iter().map(|&(x, y)| (x - mean_x) * (y - mean_y)).sum::<f64>() / n;
    let r = cov / (var_x.sqrt() * var_y.sqrt());
    if r < REFLECTION_CORRELATION_THRESHOLD {
        return None;
    }
    Some((r, cov / var_x))
}

/// Response length with the reflected payload component removed.
fn adjusted_length(feature: &ResponseFeatures, regression: Option<(f64, f64)>) -> f64 {
    match regression {
        Some((_r, slope)) => feature.response_length as f64 - slope * payload_len(feature),
        None => feature.response_length as f64,
    }
}

/// Dense feature matrix: [length, (ttfb), words, lines]; the TTFB column
/// is omitted when timing is disabled. Keep in sync with `find_representative`.
fn features_to_array(
    features: &[&ResponseFeatures],
    include_timing: bool,
    regression: Option<(f64, f64)>,
) -> Array2<f64> {
    let n_continuous = if include_timing {
        CONTINUOUS_FEATURES
    } else {
        CONTINUOUS_FEATURES_NO_TIMING
    };
    let n_samples = features.len();

    let mut data = Array2::zeros((n_samples, n_continuous));

    // A reflected payload drives the word and line counts too, and the
    // server's exact mapping is unknown; squash them to zero so they cannot
    // fragment the family. The fitted sigma of 0 makes their distance
    // contribution zero in `find_representative` as well.
    let squash = regression.is_some();
    for (i, feature) in features.iter().enumerate() {
        data[[i, 0]] = adjusted_length(feature, regression);
        let mut col = 1;
        if include_timing {
            data[[i, col]] = feature.time_to_first_byte_ms as f64;
            col += 1;
        }
        data[[i, col]] = if squash { 0.0 } else { feature.response_words as f64 };
        col += 1;
        data[[i, col]] = if squash { 0.0 } else { feature.response_lines as f64 };
    }

    data
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
    regression: Option<(f64, f64)>,
) -> Option<usize> {
    if features.is_empty() {
        return None;
    }

    // The centroid stores the RAW mean length (for reporting); the distance
    // math needs the adjusted mean when reflection was regressed out.
    let centroid_length = match regression {
        Some((_r, _slope)) => features
            .iter()
            .map(|f| adjusted_length(f, regression))
            .sum::<f64>()
            / features.len() as f64,
        None => centroid.avg_response_length,
    };

    // Column layout mirrors `features_to_array`.
    let centroid_vals: Vec<f64> = if include_timing {
        vec![
            centroid_length,
            centroid.avg_ttfb_ms,
            centroid.avg_response_words,
            centroid.avg_response_lines,
        ]
    } else {
        vec![
            centroid_length,
            centroid.avg_response_words,
            centroid.avg_response_lines,
        ]
    };

    let n = centroid_vals.len();
    let mut scaled_centroid = vec![0.0f64; n];
    for j in 0..n {
        if scaler.std[j] == 0.0 {
            scaled_centroid[j] = 0.0;
        } else {
            scaled_centroid[j] = (centroid_vals[j] - scaler.mean[j]) / scaler.std[j];
        }
    }

    let mut min_dist = f64::MAX;
    let mut representative = 0;

    for (i, feature) in features.iter().enumerate() {
        let raw: Vec<f64> = if include_timing {
            vec![
                adjusted_length(feature, regression),
                feature.time_to_first_byte_ms as f64,
                feature.response_words as f64,
                feature.response_lines as f64,
            ]
        } else {
            vec![
                adjusted_length(feature, regression),
                feature.response_words as f64,
                feature.response_lines as f64,
            ]
        };

        let mut dist_sq: f64 = 0.0;
        for j in 0..n {
            let scaled = if scaler.std[j] == 0.0 {
                0.0
            } else {
                (raw[j] - scaler.mean[j]) / scaler.std[j]
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

pub fn analyze_clusters(
    result: &DbscanResult,
    ttfb_ref: Option<(u16, f64, f64)>,
) -> Vec<String> {
    let mut observations = Vec::new();

    if let Some((_status, mean, sigma)) = ttfb_ref {
        observations.push(format!(
            "TTFB scaled against baseline jitter (mean {:.1} ms, std {:.1} ms).",
            mean, sigma
        ));
    }

    for (status, r) in &result.reflection {
        observations.push(format!(
            "Reflection detected on status {} (r = {:.2})",
            status, r
        ));
    }

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
                "Cluster {}: {} responses (status: {:>3})",
                cluster.id,
                cluster.features.len(),
                cluster.centroid.mode_status_code,
            ));
            let lengths: Vec<u64> =
                cluster.features.iter().map(|f| f.response_length as u64).collect();
            let ttfbs: Vec<u64> =
                cluster.features.iter().map(|f| f.time_to_first_byte_ms).collect();
            observations.push(format!(
                "    length: {} bytes",
                describe_range(&lengths)
            ));
            observations.push(format!("    ttfb:   {} ms", describe_range(&ttfbs)));
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

/// "mean 56, min 33, max 132" for a non-empty column of values.
fn describe_range(values: &[u64]) -> String {
    let sum = values.iter().sum::<u64>();
    format!(
        "mean {}, min {}, max {}",
        sum / values.len().max(1) as u64,
        values.iter().copied().min().unwrap_or(0),
        values.iter().copied().max().unwrap_or(0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feat(status: u16, len: usize, ttfb: u64, words: usize, lines: usize, payload: &str) -> ResponseFeatures {
        ResponseFeatures {
            status_code: status,
            content_type: "text/html".into(),
            response_length: len,
            time_to_first_byte_ms: ttfb,
            response_words: words,
            response_lines: lines,
            content_length_minus_payload: len as i64 - 10,
            payload: Some(payload.to_string()),
        }
    }

    /// Mixed-status fixture:
    /// - 6x 200s at (1000, 100ms, 100w, 10l)  -> one cluster
    /// - 1x 200  at (9000, 100ms, 100w, 10l)   -> outlier (global index 6)
    /// - 3x 500s at (300, 50ms, 30w, 3l)       -> own cluster
    fn mixed_features() -> Vec<ResponseFeatures> {
        let mut v = Vec::new();
        for i in 0..6 {
            v.push(feat(200, 1000, 100, 100, 10, &format!("bulk-{}", i)));
        }
        v.push(feat(200, 9000, 100, 100, 10, "far"));
        for i in 0..3 {
            v.push(feat(500, 300, 50, 30, 3, &format!("err-{}", i)));
        }
        v
    }

    #[test]
    fn clusters_are_homogeneous_and_indices_are_global() {
        let features = mixed_features();
        let result = perform_clustering(&features, 1.0, 3, 6, true, None, 0.0);

        // Two clusters: 6x200 and 3x500, largest first
        assert_eq!(result.clusters.len(), 2);
        assert_eq!(result.clusters[0].features.len(), 6);
        assert_eq!(result.clusters[1].features.len(), 3);
        for c in &result.clusters {
            let status = c.centroid.mode_status_code;
            assert!(c.features.iter().all(|f| f.status_code == status),
                "cluster {} mixes status codes", c.id);
        }
        assert_eq!(result.clusters[0].centroid.mode_status_code, 200);
        assert_eq!(result.clusters[1].centroid.mode_status_code, 500);

        // Outlier is the far 200, addressed by GLOBAL index
        assert_eq!(result.outliers, vec![6]);
        assert_eq!(result.noise_count, 1);
        assert_eq!(features[6].payload.as_deref(), Some("far"));

        // Sample payloads come from the cluster's real members
        assert!(result.clusters[0]
            .sample_payloads
            .iter()
            .all(|p| p.starts_with("bulk-")));
        assert!(result.clusters[1]
            .sample_payloads
            .iter()
            .all(|p| p.starts_with("err-")));
    }

    #[test]
    fn max_clusters_is_a_global_cap() {
        let features = mixed_features();
        let result = perform_clustering(&features, 1.0, 3, 1, true, None, 0.0);

        // Only the largest cluster survives; the 500s are demoted to outliers
        assert_eq!(result.clusters.len(), 1);
        assert_eq!(result.clusters[0].centroid.mode_status_code, 200);
        assert_eq!(result.outliers, vec![6, 7, 8, 9]);
        assert_eq!(result.noise_count, 4);
    }

    #[test]
    fn clustering_is_deterministic_across_runs() {
        let features = mixed_features();
        let a = perform_clustering(&features, 1.0, 3, 6, true, None, 0.0);
        let b = perform_clustering(&features, 1.0, 3, 6, true, None, 0.0);
        let sig = |r: &DbscanResult| {
            (
                r.clusters
                    .iter()
                    .map(|c| (c.id, c.centroid.mode_status_code, c.features.len()))
                    .collect::<Vec<_>>(),
                r.outliers.clone(),
            )
        };
        assert_eq!(sig(&a), sig(&b));
    }

    #[test]
    fn ttfb_baseline_ref_helper() {
        let mut statuses = HashMap::new();
        statuses.insert(200, 8);
        statuses.insert(500, 2);
        // mode status, mean, ddof=1 stddev above the floor: kept as-is
        let ref_ = ttfb_baseline_ref(&statuses, &[90, 110, 80, 120, 95, 105], 5.0).unwrap();
        assert_eq!(ref_.0, 200);
        assert!((ref_.1 - 100.0).abs() < 1e-9);
        assert!((ref_.2 - 14.4914).abs() < 1e-3);
        // near-stable endpoint: sigma floored to the given jitter
        let ref_ = ttfb_baseline_ref(&statuses, &[100; 10], 50.0).unwrap();
        assert!((ref_.2 - 50.0).abs() < 1e-9);
        // too few samples, or no statuses: no reference
        assert!(ttfb_baseline_ref(&statuses, &[100], 50.0).is_none());
        assert!(ttfb_baseline_ref(&HashMap::new(), &[100, 101], 50.0).is_none());
    }

    #[test]
    fn baseline_scale_flags_anomaly_within_infamily_sigma() {
        // Bulk at 96..109 ms around a 100 ms baseline (sigma 5 ms), plus a
        // +50 ms anomaly and a +5 s sleep. In-family scaling would absorb the
        // +50 ms anomaly into the bulk (its sigma is inflated by the sleep);
        // the baseline reference must flag it.
        let mut features = Vec::new();
        for (i, ttfb) in [96u64, 98, 100, 101, 103, 105, 107, 109].into_iter().enumerate() {
            features.push(feat(200, 1000, ttfb, 100, 10, &format!("bulk-{}", i)));
        }
        features.push(feat(200, 1000, 150, 100, 10, "slow"));
        features.push(feat(200, 1000, 6000, 100, 10, "sleep"));

        let result = perform_clustering(&features, 1.0, 3, 6, true, Some((200, 100.0, 5.0)), 0.0);

        assert_eq!(result.clusters.len(), 1);
        assert_eq!(result.clusters[0].features.len(), 8);
        assert_eq!(result.outliers, vec![8, 9]);
    }

    #[test]
    fn other_status_families_keep_their_own_scale() {
        // The reference is for 200s; 500s must be scaled in-family. If the
        // reference were (wrongly) applied to them, their 45..54 ms TTFBs
        // would sit at -110..-92 sigma and all be noise.
        let mut features = Vec::new();
        for (i, ttfb) in (45..=54).enumerate() {
            features.push(feat(500, 300, ttfb as u64, 30, 3, &format!("err-{}", i)));
        }
        let result = perform_clustering(&features, 1.0, 3, 6, true, Some((200, 100.0, 0.5)), 50.0);

        assert_eq!(result.clusters.len(), 1);
        assert_eq!(result.clusters[0].features.len(), 10);
        assert_eq!(result.noise_count, 0);
    }

    #[test]
    fn no_timing_ignores_the_reference() {
        // With --disable-timing the TTFB column is absent; the reference
        // must be a no-op and the layout [length, words, lines] holds.
        let features = mixed_features();
        let result = perform_clustering(&features, 1.0, 3, 6, false, Some((200, 100.0, 5.0)), 50.0);
        assert_eq!(result.clusters.len(), 2);
        assert_eq!(result.outliers, vec![6]);
    }

    #[test]
    fn jitter_floor_applies_to_other_families() {
        // A 500 family split 40/60 ms: in-family sigma (10 ms) breaks it
        // into two clusters, the 50 ms floor merges it into one.
        let mut features = Vec::new();
        for (i, ttfb) in [40u64, 40, 40, 60, 60, 60].into_iter().enumerate() {
            features.push(feat(500, 300, ttfb, 30, 3, &format!("err-{}", i)));
        }
        let no_floor = perform_clustering(&features, 1.0, 3, 6, true, None, 0.0);
        assert_eq!(no_floor.clusters.len(), 2);
        let floored = perform_clustering(&features, 1.0, 3, 6, true, None, 50.0);
        assert_eq!(floored.clusters.len(), 1);
        assert_eq!(floored.clusters[0].features.len(), 6);
        assert_eq!(floored.noise_count, 0);
    }
}
