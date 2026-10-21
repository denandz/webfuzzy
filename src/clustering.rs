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
    /// Status families where reflection was detected: (status, Spearman rho).
    #[serde(default)]
    pub reflection: Vec<(u16, f64)>,
    /// Theil-Sen slope (response bytes per payload byte) of each reflecting
    /// status code family, so reports can show the reflection residual 
    /// inside its clusters.
    #[serde(default)]
    pub reflection_slopes: BTreeMap<u16, f64>,
    /// Per-outlier distance to the nearest clustered (inlier) neighbour, keyed by the
    /// outlier's response index.
    pub outlier_distances: HashMap<usize, OutlierDistance>,
}

/// Per-feature distance from one outlier to its nearest clustered (inlier) neighbour,
/// measured in the outlier's family's z-scored feature space - the same
/// space DBSCAN clustered in. A "family" is one status code: the set of all
/// responses sharing it. Every `*_sq` component is a squared Euclidean
/// contribution along one feature; `content_type_sq` aggregates the whole
/// one-hot block.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OutlierDistance {
    /// Response index (position in the run-wide features list, the report's
    /// "Sequence N") of the nearest neighbour.
    pub nearest_entry: usize,
    /// Id of the cluster the nearest neighbour belongs to; `None` when the
    /// neighbour came from the same-status fallback (the family has no
    /// surviving cluster)
    #[serde(default)]
    pub nearest_cluster: Option<usize>,
    pub length_sq: f64,
    /// None when timing analysis is disabled and the matrix has no TTFB column.
    pub ttfb_sq: Option<f64>,
    pub words_sq: f64,
    pub lines_sq: f64,
    pub content_type_sq: f64,
}

/// Number of continuous features - length, ttfb, words, lines
const CONTINUOUS_FEATURES: usize = 4;
/// Number of continuous features when timing analysis is disabled (TTFB excluded).
const CONTINUOUS_FEATURES_NO_TIMING: usize = 3;
/// Response content types that get their own one-hot column, one type per
/// line in wordlists/content_types.txt.
static KNOWN_CONTENT_TYPES: &str = include_str!("../wordlists/content_types.txt");
/// One-hot column count: one per non-empty line of the embedded list, plus
/// the no-header and unknown columns.
fn content_type_columns() -> usize {
    KNOWN_CONTENT_TYPES
        .lines()
        .filter(|l| !l.trim().is_empty())
        .count()
        + 2
}

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
            reflection_slopes: BTreeMap::new(),
            outlier_distances: HashMap::new(),
        };
    }

    // Cluster per HTTP status code in it's own scaled space: a 400s
    // response length must not influence the distance between 200s.
    // Groups hold global feature indices and BTree map means deterministic
    // status order.
    let features_by_status: BTreeMap<u16, Vec<usize>> =
        features
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
    let mut reflection_slopes: BTreeMap<u16, f64> = BTreeMap::new();
    for (status, global_indices) in features_by_status.iter() {
        // Get all the features for a given status code
        let group_feats: Vec<&ResponseFeatures> =
            global_indices.iter().map(|&i| &features[i]).collect();

        // Reflected payloads make the response length track the payload
        // length, fragmenting the cluster; regress it out when detected.
        let regression = reflection_regression(&group_feats);
        if let Some((r, slope)) = regression {
            reflection.push((*status, r));
            reflection_slopes.insert(*status, slope);
        }

        let family = family_scaled_matrix(
            &group_feats,
            include_timing,
            regression,
            *status,
            ttfb_ref,
            jitter_ms,
        );

        let dataset = DatasetBase::from(family.matrix);
        let scaler = family.scaler;

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

        // Size desc, ties by DBSCAN label: HashMap iteration order is
        // random, and without the tie-break equal-size clusters (and their
        // ids, and max_clusters demotion) would shuffle between runs.
        cluster_results
            .sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| a.0.cmp(&b.0)));

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
    order.sort_by(|&a, &b| {
        final_clusters[b]
            .features
            .len()
            .cmp(&final_clusters[a].features.len())
    });

    // Global max_clusters cap: demote the smallest clusters to outliers.
    if order.len() > max_clusters {
        for &c in order.iter().skip(max_clusters) {
            outliers.extend(cluster_members[c].iter().copied());
        }
        order.truncate(max_clusters);
    }

    outliers.sort();
    let noise_count = outliers.len();

    // Per-outlier distances to the nearest surviving inlier neighbour, in
    // the family's z-space. Runs after the max_clusters demotion above:
    // `order` holds only the surviving clusters (`final_clusters` still
    // contains the demoted ones, indexed by their original position).
    // (final cluster id, cluster) pairs, in final id order.
    let surviving_clusters: Vec<(usize, &Cluster)> = order
        .iter()
        .enumerate()
        .map(|(id, &c)| (id, &final_clusters[c]))
        .collect();
    let outlier_distances = outlier_distances(
        features,
        &surviving_clusters,
        &outliers,
        include_timing,
        ttfb_ref,
        jitter_ms,
    );

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
        reflection_slopes,
        outlier_distances,
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
    feature
        .payload
        .as_ref()
        .map(|p| p.len() as f64)
        .unwrap_or(0.0)
}

/// Spearman rank correlation of paired values; None when undefined (fewer
/// than two points, or zero variance on either axis). Used for both the 
/// the per-family reflection gate and the run-level summary, so the
/// two report the same estimator. Spearman == ranked Pearson
pub fn spearman_rho(xs: &[f64], ys: &[f64]) -> Option<f64> {
    pearson_r(&average_ranks(xs), &average_ranks(ys))
}

/// (Spearman rho, Theil Sen slope) of response length against payload length
/// when reflection is detected. Both estimators selected so a few off-line
/// responses (giant error dumps, truncated reflection) don't drag the
/// correlation through the threshold for the rest of the family, which was 
/// an issue with raw Pearson/OLS. The slope covers decoders too, where the 
/// reflected length is a fraction of the payload length.
fn reflection_regression(features: &[&ResponseFeatures]) -> Option<(f64, f64)> {
    let payload_lens: Vec<f64> = features.iter().map(|f| payload_len(f)).collect();
    let response_lens: Vec<f64> = features.iter().map(|f| f.response_length as f64).collect();

    let rho = spearman_rho(&payload_lens, &response_lens)?;
    if rho < REFLECTION_CORRELATION_THRESHOLD {
        return None;
    }

    Some((rho, theil_sen_slope(&payload_lens, &response_lens)))
}

/// Pearson correlation of paired values; None when undefined (fewer than
/// two points, or zero variance on either axis).
fn pearson_r(xs: &[f64], ys: &[f64]) -> Option<f64> {
    let n = xs.len() as f64;
    if n < 2.0 {
        return None;
    }
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let var_x = xs
        .iter()
        .map(|x| (x - mean_x).powi(2))
        .sum::<f64>()
        / n;
    let var_y = ys
        .iter()
        .map(|y| (y - mean_y).powi(2))
        .sum::<f64>()
        / n;
    if var_x == 0.0 || var_y == 0.0 {
        return None;
    }
    let cov = xs
        .iter()
        .zip(ys)
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>()
        / n;
    Some(cov / (var_x.sqrt() * var_y.sqrt()))
}

/// 1-based ranks with ties averaged, (Pearson on ranks is Spearman)
fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());
    let mut ranks = vec![0.0; values.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && values[order[j + 1]] == values[order[i]] {
            j += 1;
        }
        let rank = (i + j) as f64 / 2.0 + 1.0;
        for &idx in &order[i..=j] {
            ranks[idx] = rank;
        }
        i = j + 1;
    }
    ranks
}

/// Theil-Sen slope: the median of all pairwise slopes. 
fn theil_sen_slope(xs: &[f64], ys: &[f64]) -> f64 {
    const MAX_POINTS: usize = 8192; // Max points for theil-sen to keep memory pressure manageable
                                    // with longer wordlists. Kicks in stride sampling if a status
                                    // code family has more than 8192 members in it.
    let step = xs.len().div_ceil(MAX_POINTS).max(1);
    let points: Vec<(f64, f64)> = (0..xs.len())
        .step_by(step)
        .map(|i| (xs[i], ys[i]))
        .collect();
    let mut slopes = Vec::with_capacity(points.len() * (points.len() - 1) / 2);
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let dx = points[j].0 - points[i].0;
            if dx != 0.0 {
                slopes.push((points[j].1 - points[i].1) / dx);
            }
        }
    }
    if slopes.is_empty() {
        // All sampled payload lengths are equal meaning nothing to regress out
        return 0.0;
    }
    slopes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    slopes[slopes.len() / 2]
}

/// Response length with the reflected payload component removed.
fn adjusted_length(feature: &ResponseFeatures, regression: Option<(f64, f64)>) -> f64 {
    match regression {
        Some((_r, slope)) => feature.response_length as f64 - slope * payload_len(feature),
        None => feature.response_length as f64,
    }
}

/// Bare media type for column lookup: "; ..." parameters stripped, case
/// kept.
fn bare_media_type(content_type: &str) -> String {
    content_type
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_string()
}

/// One-hot column index for a content type: known types first, then the
/// no-header and unknown columns.
fn content_type_column(content_type: &str) -> usize {
    let ct = bare_media_type(content_type);
    let mut known = 0usize;
    for line in KNOWN_CONTENT_TYPES.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if line.trim() == ct.as_str() {
            return known;
        }
        known += 1;
    }
    if ct.is_empty() { known } else { known + 1 }
}

/// One status code family scaled to its z-space. This is the single source
/// of truth for family scaling: the per-family DBSCAN matrix and the outlier
/// distance pass both build their scaled rows through here, so the two cant
/// diverge.
struct FamilyScaled {
    /// Rows in the given response order; columns: length, (ttfb), words,
    /// lines, one-hot content-type block.
    matrix: Array2<f64>,
    /// The fitted scaler, for follow-up math on the same z-space
    /// (e.g. `find_representative`).
    scaler: Scaler,
}

fn family_scaled_matrix(
    features: &[&ResponseFeatures],
    include_timing: bool,
    regression: Option<(f64, f64)>,
    status: u16,
    ttfb_ref: Option<(u16, f64, f64)>,
    jitter_ms: f64,
) -> FamilyScaled {
    let mut data = features_to_array(features, include_timing, regression);

    let mut scaler = Scaler::fit(&data);
    if include_timing {
        // TTFB column: when this family is the baseline's, scale against
        // the baseline's own jitter so normal endpoint wobble is absorbed
        // and only real timing anomalies stand out.
        if let Some((ref_status, ref_mean, ref_sigma)) = ttfb_ref {
            if ref_status == status {
                scaler.set_scale(1, ref_mean, ref_sigma);
            }
        }
        // Set the scaler's TTFB stddev to the jitter floor so millisecond
        // wobble isn't flagged as timing outliers. No-ops when the data is
        // naturally more variable than the floor.
        scaler.std[1] = scaler.std[1].max(jitter_ms);
    }
    scaler.transform(&mut data);

    FamilyScaled {
        matrix: data,
        scaler,
    }
}

/// Column index of a feature in the scaled matrix layout:
/// 0=length, 1=TTFB (only when timing is enabled), 2=words, 3=lines, then
/// the one-hot content-type block. Shared with the outlier distance pass so
/// the components map to the same columns DBSCAN clustered in.
fn feature_column(name: &str, include_timing: bool) -> usize {
    match name {
        "length" => 0,
        "ttfb" if include_timing => 1,
        "words" => {
            if include_timing {
                2
            } else {
                1
            }
        }
        "lines" => {
            if include_timing {
                3
            } else {
                2
            }
        }
        _ => panic!("unknown feature column: {}", name),
    }
}

/// For each outlier, the per-feature distance to its nearest inlier
/// neighbour, measured in the outlier's family's z-scored feature space -
/// the exact space DBSCAN clustered in, via `family_scaled_matrix`.
///
/// Must run after the max_clusters demotion: candidates are members of
/// surviving clusters only. When the outlier's family has no surviving
/// cluster, the candidates fall back to all same-status responses except
/// the outlier itself.
fn outlier_distances(
    features: &[ResponseFeatures],
    clusters: &[(usize, &Cluster)],
    outliers: &[usize],
    include_timing: bool,
    ttfb_ref: Option<(u16, f64, f64)>,
    jitter_ms: f64,
) -> HashMap<usize, OutlierDistance> {
    let mut distances: HashMap<usize, OutlierDistance> = HashMap::new();
    if features.is_empty() || outliers.is_empty() {
        return distances;
    }

    // All response indices grouped by status code, run-wide order kept.
    let responses_by_status: BTreeMap<u16, Vec<usize>> =
        features
            .iter()
            .enumerate()
            .fold(BTreeMap::new(), |mut acc, (i, f)| {
                acc.entry(f.status_code).or_default().push(i);
                acc
            });

    // Response indices of surviving cluster members, grouped by family
    // status code. Cluster members are clones of the run-wide features, so
    // membership is recovered by value equality; if two responses are
    // feature-identical, either index is equivalent (their scaled rows are
    // identical). Clustering is per status code, so every member of a
    // cluster shares one status: features[0].status_code is a safe family
    // tag (clusters are never empty).
    // (response index, cluster id) pairs of surviving cluster members.
    let mut members_by_status: BTreeMap<u16, Vec<(usize, usize)>> = BTreeMap::new();
    for (cluster_id, cluster) in clusters {
        let family_status = cluster.features[0].status_code;
        for member in &cluster.features {
            if let Some(response_index) = features.iter().position(|f| f == member) {
                members_by_status
                    .entry(family_status)
                    .or_default()
                    .push((response_index, *cluster_id));
            }
        }
    }

    // Outliers grouped by their own family's status code, so each family's
    // scaler is fitted exactly once.
    let mut outliers_by_status: BTreeMap<u16, Vec<usize>> = BTreeMap::new();
    for &outlier in outliers {
        let status = features[outlier].status_code;
        outliers_by_status.entry(status).or_default().push(outlier);
    }

    for (status, outlier_indices) in &outliers_by_status {
        // Same population the per-family DBSCAN path scaled: every response
        // with this status, so the fitted scaler is identical to the one
        // clustering used.
        let family_indices = &responses_by_status[status];
        let family_features: Vec<&ResponseFeatures> =
            family_indices.iter().map(|&i| &features[i]).collect();
        let regression = reflection_regression(&family_features);
        let scaled = family_scaled_matrix(
            &family_features,
            include_timing,
            regression,
            *status,
            ttfb_ref,
            jitter_ms,
        )
        .matrix;

        // Row (a coordinate inside this family's scaled matrix) for every
        // response index of the family.
        let row_of: HashMap<usize, usize> = family_indices
            .iter()
            .enumerate()
            .map(|(r, &i)| (i, r))
            .collect();

        // Candidates: surviving inlier members of this family, or - when the
        // family has no surviving cluster - all same-status responses.
        let members = members_by_status.get(status).filter(|m| !m.is_empty());

        for &outlier in outlier_indices {
            let Some(&row) = row_of.get(&outlier) else {
                continue;
            };

            // (response index, cluster id of the candidate or None, total squared distance)
            let mut best: Option<(usize, Option<usize>, f64)> = None;

            let scan = |candidate: usize,
                        cluster: Option<usize>,
                        best: &mut Option<(usize, Option<usize>, f64)>| {
                if candidate == outlier {
                    return;
                }
                let Some(candidate_row) = row_of.get(&candidate) else {
                    return;
                };
                let mut total_sq = 0.0;
                for j in 0..scaled.ncols() {
                    let d = scaled[[row, j]] - scaled[[*candidate_row, j]];
                    total_sq += d * d;
                }
                if best.is_none_or(|(_, _, best_sq)| total_sq < best_sq) {
                    *best = Some((candidate, cluster, total_sq));
                }
            };

            match members {
                Some(members) => {
                    for &(candidate, cluster_id) in members {
                        scan(candidate, Some(cluster_id), &mut best);
                    }
                }
                None => {
                    for &candidate in family_indices {
                        scan(candidate, None, &mut best);
                    }
                }
            }

            let Some((nearest, nearest_cluster, _total_sq)) = best else {
                continue;
            };
            let nearest_row = row_of[&nearest];

            // Squared components in the same column order as the scaled
            // matrix; the one-hot block is summed into a single value.
            let component_sq = |j: usize| {
                let d = scaled[[row, j]] - scaled[[nearest_row, j]];
                d * d
            };
            let first_ct_column = feature_column("lines", include_timing) + 1;
            let content_type_sq: f64 = (first_ct_column..scaled.ncols()).map(component_sq).sum();

            distances.insert(
                outlier,
                OutlierDistance {
                    nearest_entry: nearest,
                    nearest_cluster,
                    length_sq: component_sq(feature_column("length", include_timing)),
                    ttfb_sq: include_timing
                        .then(|| component_sq(feature_column("ttfb", include_timing))),
                    words_sq: component_sq(feature_column("words", include_timing)),
                    lines_sq: component_sq(feature_column("lines", include_timing)),
                    content_type_sq,
                },
            );
        }
    }

    distances
}

/// Dense feature matrix: [length, (ttfb), words, lines, one-hot content
/// type columns]; the TTFB column is omitted when timing is disabled. Keep
/// in sync with `find_representative`.
fn features_to_array(
    features: &[&ResponseFeatures],
    include_timing: bool,
    regression: Option<(f64, f64)>,
) -> Array2<f64> {
    let n_continuous = (if include_timing {
        CONTINUOUS_FEATURES
    } else {
        CONTINUOUS_FEATURES_NO_TIMING
    }) + content_type_columns();
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
        data[[i, col]] = if squash {
            0.0
        } else {
            feature.response_words as f64
        };
        col += 1;
        data[[i, col]] = if squash {
            0.0
        } else {
            feature.response_lines as f64
        };
        col += 1;
        data[[i, col + content_type_column(&feature.content_type)]] = 1.0;
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
        Some((_r, _slope)) => {
            features
                .iter()
                .map(|f| adjusted_length(f, regression))
                .sum::<f64>()
                / features.len() as f64
        }
        None => centroid.avg_response_length,
    };

    // Column layout mirrors `features_to_array`.
    let mut centroid_vals: Vec<f64> = if include_timing {
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

    // One-hot content-type columns, same layout as `features_to_array`.
    // The centroid takes the cluster's mode type; clusters are effectively
    // single-type, so this is a formality.
    let mut ct_counts: HashMap<String, usize> = HashMap::new();
    for f in features {
        *ct_counts.entry(f.content_type.clone()).or_default() += 1;
    }
    let mode_ct = ct_counts
        .iter()
        .max_by(|a, b| a.1.cmp(b.1).then_with(|| a.0.cmp(b.0)))
        .map(|(ct, _)| ct.clone())
        .unwrap_or_default();
    centroid_vals.extend(std::iter::repeat(0.0).take(content_type_columns()));
    let ct_index = centroid_vals.len() - content_type_columns() + content_type_column(&mode_ct);
    centroid_vals[ct_index] = 1.0;

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
        let mut raw: Vec<f64> = if include_timing {
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
        raw.extend(std::iter::repeat(0.0).take(content_type_columns()));
        let ct_index =
            raw.len() - content_type_columns() + content_type_column(&feature.content_type);
        raw[ct_index] = 1.0;

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

pub fn analyze_clusters(result: &DbscanResult, ttfb_ref: Option<(u16, f64, f64)>) -> Vec<String> {
    let mut observations = Vec::new();

    if let Some((_status, mean, sigma)) = ttfb_ref {
        observations.push(format!(
            "TTFB scaled against baseline jitter (mean {:.1} ms, std {:.1} ms).",
            mean, sigma
        ));
    }

    for (status, r) in &result.reflection {
        // Slope is the reflection factor: response bytes per payload byte (3.0
        // means the input lands in the output three times).
        match result.reflection_slopes.get(status) {
            Some(&slope) => observations.push(format!(
                "Reflection detected on status {} (rho = {:.2}, slope = {:.1})",
                status, r, slope
            )),
            None => observations.push(format!(
                "Reflection detected on status {} (rho = {:.2})",
                status, r
            )),
        }
    }

    if result.clusters.len() == 1 && result.noise_count == 0 {
        observations.push(
            "[.] All responses belong to a single cluster - Responses are consistent.".to_string(),
        );
    }
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

        let mut content_types: HashMap<String, u64> = HashMap::new();
        cluster
            .features
            .iter()
            .for_each(|f| *content_types.entry(f.content_type.clone()).or_insert(0) += 1);

        let mut content_types_v: Vec<(String, u64)> = content_types.into_iter().collect();
        content_types_v.sort_by(|a, b| b.1.cmp(&a.1));
        let top_content_types_str: String = content_types_v
            .iter()
            .take(5)
            .map(|(k, v)| format!("{}: {}", k, v))
            .collect::<Vec<_>>()
            .join(", ");

        if content_types_v.len() == 1 {
            observations.push(format!(
                "    Unique Content-Types: {}",
                content_types_v.len()
            ));
        }

        observations.push(format!("    Content-Types: {}", top_content_types_str));

        let lengths: Vec<u64> = cluster
            .features
            .iter()
            .map(|f| f.response_length as u64)
            .collect();
        let ttfbs: Vec<u64> = cluster
            .features
            .iter()
            .map(|f| f.time_to_first_byte_ms)
            .collect();

        observations.push(format!("    Length: {} bytes", describe_range(&lengths)));

        if let Some(&slope) = result.reflection_slopes.get(&cluster.centroid.mode_status_code) {
            // Reflecting family: also show the reflection residual (length with the
            // fitted reflection subtracted), on top of the raw lengths above. A
            // perfectly uniform residual means every member wa reflected
            // verbatim; any spread is members that were not, behaviour that
            // may sit inside the cluster without reaching the outlier bar.
            let residuals: Vec<f64> = cluster
                .features
                .iter()
                .map(|f| f.response_length as f64 - slope * payload_len(f))
                .collect();
            let min = residuals.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = residuals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if max - min < 1.0 {
                observations.push(format!(
                    "    Reflection residual: {} bytes (uniform)",
                    min.round()
                ));
            } else {
                let mean = residuals.iter().sum::<f64>() / residuals.len() as f64;
                observations.push(format!(
                    "    Reflection residual: mean {}, min {}, max {} bytes",
                    mean.round(),
                    min.round(),
                    max.round()
                ));
            }
        }
        observations.push(format!("    TTFB:   {} ms", describe_range(&ttfbs)));

        for payload in &cluster.sample_payloads {
            observations.push(format!("    Sample payload: '{}'", payload));
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

    fn feat(
        status: u16,
        len: usize,
        ttfb: u64,
        words: usize,
        lines: usize,
        payload: &str,
    ) -> ResponseFeatures {
        ResponseFeatures {
            request_id: "00000000-0000-0000-0000-000000000000".into(),
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

    fn feat_ct(
        status: u16,
        len: usize,
        ttfb: u64,
        words: usize,
        lines: usize,
        payload: &str,
        ct: &str,
    ) -> ResponseFeatures {
        let mut f = feat(status, len, ttfb, words, lines, payload);
        f.content_type = ct.to_string();
        f
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
            assert!(
                c.features.iter().all(|f| f.status_code == status),
                "cluster {} mixes status codes",
                c.id
            );
        }
        assert_eq!(result.clusters[0].centroid.mode_status_code, 200);
        assert_eq!(result.clusters[1].centroid.mode_status_code, 500);

        // Outlier is the far 200, addressed by GLOBAL index
        assert_eq!(result.outliers, vec![6]);
        assert_eq!(result.noise_count, 1);
        assert_eq!(features[6].payload.as_deref(), Some("far"));

        // Sample payloads come from the cluster's real members
        assert!(
            result.clusters[0]
                .sample_payloads
                .iter()
                .all(|p| p.starts_with("bulk-"))
        );
        assert!(
            result.clusters[1]
                .sample_payloads
                .iter()
                .all(|p| p.starts_with("err-"))
        );
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
        for (i, ttfb) in [96u64, 98, 100, 101, 103, 105, 107, 109]
            .into_iter()
            .enumerate()
        {
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

    #[test]
    fn content_type_split_forms_two_clusters() {
        // Identical responses except content type: html vs json must split.
        let mut features = Vec::new();
        for i in 0..6 {
            features.push(feat_ct(
                200,
                1000,
                100,
                50,
                5,
                &format!("html-{}", i),
                "text/html",
            ));
            features.push(feat_ct(
                200,
                1000,
                100,
                50,
                5,
                &format!("json-{}", i),
                "application/json",
            ));
        }
        let result = perform_clustering(&features, 1.0, 3, 6, true, None, 50.0);

        assert_eq!(result.clusters.len(), 2);
        assert_eq!(result.noise_count, 0);
        let sizes: Vec<usize> = result.clusters.iter().map(|c| c.features.len()).collect();
        assert_eq!(sizes, vec![6, 6]);

        // Deterministic tie-breaker: equal-size clusters must always get the
        // same ids (HashMap iteration order must not leak through).
        for _ in 0..25 {
            let again = perform_clustering(&features, 1.0, 3, 6, true, None, 50.0);
            assert_eq!(
                again.clusters
                    .iter()
                    .map(|c| (c.id, c.features[0].content_type.clone()))
                    .collect::<Vec<_>>(),
                result.clusters
                    .iter()
                    .map(|c| (c.id, c.features[0].content_type.clone()))
                    .collect::<Vec<_>>(),
                "equal-size cluster ids must be stable across runs"
            );
        }
    }

    #[test]
    fn rare_content_type_is_an_outlier() {
        // One json response in a sea of html: too few to form a cluster,
        // must be flagged as noise.
        let mut features = Vec::new();
        for i in 0..8 {
            features.push(feat_ct(
                200,
                1000,
                100,
                50,
                5,
                &format!("html-{}", i),
                "text/html",
            ));
        }
        features.push(feat_ct(200, 1000, 100, 50, 5, "json", "application/json"));
        let result = perform_clustering(&features, 1.0, 3, 6, true, None, 50.0);

        assert_eq!(result.clusters.len(), 1);
        assert_eq!(result.clusters[0].features.len(), 8);
        assert_eq!(result.outliers, vec![8]);
    }

    #[test]
    fn content_type_matching_strips_parameters() {
        // "text/html; charset=UTF-8" lands in the text/html column.
        let mut features = Vec::new();
        for i in 0..6 {
            features.push(feat_ct(
                200,
                1000,
                100,
                50,
                5,
                &format!("a-{}", i),
                "text/html; charset=UTF-8",
            ));
            features.push(feat_ct(
                200,
                1000,
                100,
                50,
                5,
                &format!("b-{}", i),
                "text/html",
            ));
        }
        let result = perform_clustering(&features, 1.0, 3, 6, true, None, 50.0);

        assert_eq!(result.clusters.len(), 1);
        assert_eq!(result.clusters[0].features.len(), 12);
        assert_eq!(result.noise_count, 0);
    }

    #[test]
    fn non_canonical_case_content_type_is_unknown() {
        // "Text/HTML" deliberately misses the canonical column; the six
        // variants form their own cluster via the unknown column and the
        // json response is the outlier.
        let mut features = Vec::new();
        for i in 0..6 {
            features.push(feat_ct(
                200,
                1000,
                100,
                50,
                5,
                &format!("weird-{}", i),
                "Text/HTML; charset=UTF-8",
            ));
            features.push(feat_ct(
                200,
                1000,
                100,
                50,
                5,
                &format!("html-{}", i),
                "text/html",
            ));
        }
        features.push(feat_ct(200, 1000, 100, 50, 5, "json", "application/json"));
        let result = perform_clustering(&features, 1.0, 3, 6, true, None, 50.0);

        assert_eq!(result.clusters.len(), 2);
        let sizes: Vec<usize> = result.clusters.iter().map(|c| c.features.len()).collect();
        assert_eq!(sizes, vec![6, 6]);
        assert_eq!(result.outliers, vec![12]);
    }

    #[test]
    fn length_dominant_outlier_distance() {
        // The far 200 (response index 6) stands out on length alone.
        let features = mixed_features();
        let result = perform_clustering(&features, 1.0, 3, 6, true, None, 0.0);
        assert_eq!(result.outliers, vec![6]);

        let d = &result.outlier_distances[&6];
        // Nearest neighbour is one of the six identical bulk 200s.
        assert!(
            d.nearest_entry < 6,
            "nearest neighbour must be a bulk 200, got {}",
            d.nearest_entry
        );
        assert_eq!(features[d.nearest_entry].status_code, 200);
        // The recorded neighbour cluster actually contains the neighbour.
        let cluster_id = d
            .nearest_cluster
            .expect("nearest neighbour must belong to a cluster");
        assert!(
            result.clusters[cluster_id]
                .features
                .iter()
                .any(|f| f == &features[d.nearest_entry]),
            "nearest neighbour (entry {}) must be a member of cluster {}",
            d.nearest_entry,
            cluster_id
        );
        // Length dominates the per-feature breakdown.
        let rest = d.words_sq + d.lines_sq + d.content_type_sq + d.ttfb_sq.unwrap_or(0.0);
        assert!(
            d.length_sq > rest,
            "length_sq {} should dominate rest {}",
            d.length_sq,
            rest
        );
    }

    #[test]
    fn content_type_dominant_outlier_distance() {
        // One json response in a sea of html (response index 8): only the
        // one-hot content-type block differs.
        let mut features = Vec::new();
        for i in 0..8 {
            features.push(feat_ct(
                200,
                1000,
                100,
                50,
                5,
                &format!("html-{}", i),
                "text/html",
            ));
        }
        features.push(feat_ct(200, 1000, 100, 50, 5, "json", "application/json"));
        let result = perform_clustering(&features, 1.0, 3, 6, true, None, 50.0);
        assert_eq!(result.outliers, vec![8]);

        let d = &result.outlier_distances[&8];
        assert!(d.nearest_entry < 8);
        // The one-hot swap is the only difference: a non-trivial z-distance
        // (the two columns have different sigmas, so it is not exactly 2).
        assert!(d.content_type_sq > 1.0);
        let rest = d.length_sq + d.words_sq + d.lines_sq + d.ttfb_sq.unwrap_or(0.0);
        assert!(
            d.content_type_sq > rest,
            "content_type_sq {} should dominate rest {}",
            d.content_type_sq,
            rest
        );
    }

    #[test]
    fn words_dominant_outlier_distance() {
        // One response with a very different word count (response index 8).
        let mut features = Vec::new();
        for i in 0..8 {
            features.push(feat(200, 1000, 100, 50, 5, &format!("w-{}", i)));
        }
        features.push(feat(200, 1000, 100, 950, 5, "many-words"));
        let result = perform_clustering(&features, 1.0, 3, 6, true, None, 50.0);
        assert_eq!(result.outliers, vec![8]);

        let d = &result.outlier_distances[&8];
        assert!(d.nearest_entry < 8);
        let rest = d.length_sq + d.lines_sq + d.content_type_sq + d.ttfb_sq.unwrap_or(0.0);
        assert!(
            d.words_sq > rest,
            "words_sq {} should dominate rest {}",
            d.words_sq,
            rest
        );
    }

    #[test]
    fn demoted_family_falls_back_to_same_status() {
        // max_clusters=1 demotes the 500 cluster: the demoted 500s become
        // outliers whose family has no surviving cluster, so the fallback
        // candidate set is all same-status responses.
        let features = mixed_features();
        let result = perform_clustering(&features, 1.0, 3, 1, true, None, 0.0);
        assert_eq!(result.outliers, vec![6, 7, 8, 9]);

        // Each demoted 500's nearest neighbour is one of the other two
        // identical 500s, at zero distance.
        for &o in &[7, 8, 9] {
            let d = &result.outlier_distances[&o];
            assert_eq!(
                features[d.nearest_entry].status_code, 500,
                "demoted 500 {} should match a 500, got {}",
                o, d.nearest_entry
            );
            let total = d.length_sq
                + d.words_sq
                + d.lines_sq
                + d.content_type_sq
                + d.ttfb_sq.unwrap_or(0.0);
            assert!(total == 0.0, "identical 500s must be at zero distance");
            // Fallback neighbours come from the unclustered family: no cluster.
            assert!(d.nearest_cluster.is_none());
        }

        // The 200 outlier still matches a surviving 200 cluster member.
        let d = &result.outlier_distances[&6];
        assert_eq!(features[d.nearest_entry].status_code, 200);
        assert!(d.length_sq > 0.0);
        assert!(d.nearest_cluster.is_some());
    }

    #[test]
    fn offset_family_outlier_distance() {
        // ensure distances for all status code families (not just the main one, 0,)
        // function as expected
        let mut features = Vec::new();
        for i in 0..4 {
            features.push(feat(200, 500, 100, 40, 4, &format!("ok-{}", i)));
        }
        for i in 0..4 {
            features.push(feat(500, 200, 40, 20, 2, &format!("err-{}", i)));
        }
        // Same-length payload as the other 500s, so the family is not
        // reflective and the length outlier is judged on raw length.
        features.push(feat(500, 9000, 40, 20, 2, "far50")); // response index 8

        let result = perform_clustering(&features, 1.0, 3, 6, true, None, 0.0);
        assert_eq!(result.outliers, vec![8]);

        let d = &result.outlier_distances[&8];
        assert!(
            (4..8).contains(&d.nearest_entry),
            "500 outlier's nearest neighbour must be a 500 cluster member, got {}",
            d.nearest_entry
        );
        assert!(d.length_sq > 0.0);
    }

    #[test]
    fn no_header_and_unknown_content_types_separate() {
        // html, missing header, and an unknown type: three groups.
        let mut features = Vec::new();
        for i in 0..5 {
            features.push(feat_ct(
                200,
                1000,
                100,
                50,
                5,
                &format!("html-{}", i),
                "text/html",
            ));
            features.push(feat_ct(200, 1000, 100, 50, 5, &format!("none-{}", i), ""));
            features.push(feat_ct(
                200,
                1000,
                100,
                50,
                5,
                &format!("gif-{}", i),
                "image/gif",
            ));
        }
        let result = perform_clustering(&features, 1.0, 3, 6, true, None, 50.0);

        assert_eq!(result.clusters.len(), 3);
        assert_eq!(result.noise_count, 0);
    }

    /// y = 600 + 3x, with the payload itself providing the length.
    fn reflected_features(count: usize) -> Vec<ResponseFeatures> {
        (1..=count)
            .map(|x| feat(200, 600 + 3 * x, 100, 100, 10, &"a".repeat(x)))
            .collect()
    }

    fn refs(features: &[ResponseFeatures]) -> Vec<&ResponseFeatures> {
        features.iter().collect()
    }

    #[test]
    fn reflection_is_detected_and_slope_recovered() {
        let features = reflected_features(50);
        let (rho, slope) = reflection_regression(&refs(&features)).expect("detected");
        assert!(rho > REFLECTION_CORRELATION_THRESHOLD);
        assert!((slope - 3.0).abs() < 0.1);
    }

    #[test]
    fn giant_responses_do_not_disable_reflection() {
        let mut features = reflected_features(200);
        // Two ~250KB error dumps on 1-byte payloads.
        features.push(feat(200, 250_000, 100, 50, 40, "1"));
        features.push(feat(200, 249_000, 100, 50, 40, "2"));
        let (rho, slope) = reflection_regression(&refs(&features)).expect("still detected");
        assert!(rho > REFLECTION_CORRELATION_THRESHOLD);
        assert!((slope - 3.0).abs() < 0.5);
    }

    #[test]
    fn length_capped_reflection_is_still_detected() {
        // Payloads beyond 255 chars reflect at a constant length.
        let features: Vec<ResponseFeatures> = (1..=300)
            .map(|x| feat(200, 600 + 3 * x.min(255), 100, 100, 10, &"a".repeat(x)))
            .collect();
        let (rho, slope) = reflection_regression(&refs(&features)).expect("still detected");
        assert!(rho > REFLECTION_CORRELATION_THRESHOLD);
        assert!((slope - 3.0).abs() < 0.5);
    }

    #[test]
    fn constant_length_is_not_reflection() {
        let features: Vec<ResponseFeatures> = (1..=20)
            .map(|x| feat(200, 500, 100, 100, 10, &"a".repeat(x)))
            .collect();
        assert!(reflection_regression(&refs(&features)).is_none());
    }

    #[test]
    fn reflecting_cluster_reports_uniform_residual() {
        let features = reflected_features(30);
        let result = perform_clustering(&features, 1.0, 3, 6, true, None, 0.0);
        let obs = analyze_clusters(&result, None);
        assert!(
            obs.iter().any(|l| l.contains("Reflection residual: 600 bytes (uniform)")),
            "expected uniform reflection residual line, got: {:?}",
            obs
        );
    }

    #[test]
    fn reflecting_cluster_reports_residual_spread() {
        let mut features = reflected_features(30);
        // One member echoes 200 bytes short of the line. Wide enough
        // tolerance that it stays inside the cluster, where the
        // residual spread gives it away.
        features.push(feat(200, 700, 100, 100, 10, &"a".repeat(100)));
        let result = perform_clustering(&features, 6.0, 3, 6, true, None, 0.0);
        let obs = analyze_clusters(&result, None);
        assert!(
            obs.iter()
                .any(|l| l.starts_with("    Reflection residual: mean") && l.contains("min 400")),
            "expected residual spread line, got: {:?}",
            obs
        );
    }
}
