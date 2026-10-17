use crate::baseline::BaselineResult;
use crate::clustering;
use crate::fuzzing::FuzzingResult;

pub fn print_summary(baseline: &BaselineResult, fuzzing: &FuzzingResult) {
    println!();
    println!("{}", "[+] WEBFUZZY RESULTS");

    print_baseline_summary(baseline);
    print_fuzzing_summary(fuzzing);

    println!();
}

pub fn print_baseline_summary(baseline: &BaselineResult) {
    println!();
    println!("{}", "[+] BASELINE ANALYSIS");

    if baseline.request_count == 0 {
        println!("  Skipped");
        return;
    }

    println!("  Requests: {}", baseline.request_count);
    println!(
        "  Status code consistency: {:.1}%",
        baseline.status_code_consistency * 100.0
    );
    println!(
        "  Response length consistency: {:.1}%",
        baseline.response_length_consistency * 100.0
    );
    println!("  Average TTFB: {:?}", baseline.average_ttfb);

    if baseline.stable {
        println!("  Status: {}", "Stable");
    } else {
        println!("  Status: {}", "Unstable");
    }
}

pub fn print_fuzzing_summary(fuzzing: &FuzzingResult) {
    println!();
    println!("{}", "[+] OUTLIER-BASED FUZZING");

    if fuzzing.total_requests == 0 {
        println!("  Skipped");
        return;
    }

    println!("  Total requests: {}", fuzzing.total_requests);
    println!("  Successful: {}", fuzzing.successful_requests);
    println!("  Failed: {}", fuzzing.failed_requests);
    match fuzzing.payload_length_correlation {
        Some(r) => {
            if r > 0.8 {
                println!(
                    "  Payload-response correlation (HIGH - likely payload reflection): {r:.4}"
                );
            } else if r > 0.5 {
                println!(
                    "  Payload-response correlation (MED - check for reflection and length-based clusters): {r:.4}"
                );
            } else {
                println!("  Payload-response correlation (LOW - reflection unlikely): {r:.4}");
            }
        }
        None => println!("  Payload-response correlation: N/A - No reflection"),
    }
    println!();
    println!("  Status code distribution:");
    let mut status_codes: Vec<_> = fuzzing.status_codes.iter().collect();
    status_codes.sort_by_key(|&(_, &count)| std::cmp::Reverse(count));
    for (code, count) in &status_codes {
        let percentage = **count as f64 / fuzzing.total_requests as f64 * 100.0;
        println!("    {code}: {count} ({percentage:.1}%)");
    }

    println!();
    let cluster_observations =
        clustering::analyze_clusters(&fuzzing.clustering_result, fuzzing.ttfb_baseline_ref);
    for observation in &cluster_observations {
        println!("  {}", observation);
    }

    if fuzzing.clustering_result.noise_count > 0 {
        println!();
        let show = fuzzing.clustering_result.noise_count.min(10);
        println!(
            "  Outlier payloads ({show} of {}):",
            fuzzing.clustering_result.noise_count
        );
        for &idx in fuzzing.clustering_result.outliers.iter().take(10) {
            if let Some(feature) = fuzzing.response_features.get(idx) {
                let payload = feature.payload.as_deref().unwrap_or("");
                println!("    '{}'", truncate(payload, 40));
            }
        }
    }
}

pub fn print_outlier_details(fuzzing: &FuzzingResult) {
    if fuzzing.clustering_result.noise_count == 0 {
        return;
    }

    println!();
    println!("[+] OUTLIER DETAILS");
    println!("{}", "-".repeat(19));

    for &outlier_idx in &fuzzing.clustering_result.outliers {
        if let Some(feature) = fuzzing.response_features.get(outlier_idx) {
            println!(
                "  Sequence {outlier_idx} (payload: '{}')",
                feature.payload.as_deref().unwrap_or("")
            );
            println!("    Status: {}", feature.status_code);

            // Whether this status family has any surviving cluster, i.e. any
            // true inliers to measure against. A family with fewer responses
            // than DBSCAN's min_samples can never form a cluster. Handles the
            // case where a fuzz run has <3 responses in a certain status code
            // e.g., the fuzz run only saw 2 401 responses.
            let family_has_inliers = fuzzing.clustering_result.clusters.iter().any(|c| {
                c.features
                    .first()
                    .is_some_and(|f| f.status_code == feature.status_code)
            });

            // Per-feature distance to the nearest inlier neighbour, in sigma
            // units of the family's z-space; '*' marks the dominant feature.
            let distance = family_has_inliers
                .then(|| {
                    fuzzing
                        .clustering_result
                        .outlier_distances
                        .get(&outlier_idx)
                        .map(feature_distances)
                })
                .flatten();
            let dominant = distance.as_ref().and_then(|d| {
                d.iter()
                    .max_by(|a, b| a.3.total_cmp(&b.3))
                    .map(|(name, ..)| *name)
            });

            // "[dist X.XXσ]" next to a feature line; '*' on the dominant
            // feature. Empty when the feature has no distance (timing
            // disabled) or no neighbour was found.
            let dist_tag = |sigma: Option<f64>, name: &'static str| -> String {
                match sigma {
                    Some(s) if s > 0.0 => {
                        let star = if Some(name) == dominant { "*" } else { "" };
                        format!(" [dist {s:.2}\u{3c3}{star}]")
                    }
                    _ => String::new(),
                }
            };
            let sigma_of = |name: &'static str| -> Option<f64> {
                distance
                    .as_ref()
                    .and_then(|d| d.iter().find(|(n, _, _, _)| *n == name))
                    .and_then(|(_, _, sigma, _)| *sigma)
            };

            println!(
                "    Length: {} bytes{}",
                feature.response_length,
                dist_tag(sigma_of("length"), "length")
            );
            println!(
                "    TTFB: {}ms{}",
                feature.time_to_first_byte_ms,
                dist_tag(sigma_of("ttfb"), "ttfb")
            );
            println!(
                "    Words: {}{}",
                feature.response_words,
                dist_tag(sigma_of("words"), "words")
            );
            println!(
                "    Lines: {}{}",
                feature.response_lines,
                dist_tag(sigma_of("lines"), "lines")
            );
            println!(
                "    Content-Type: {}{}",
                feature.content_type,
                dist_tag(sigma_of("content type"), "content type")
            );

            if let Some(distance) = &distance {
                let od = &fuzzing.clustering_result.outlier_distances[&outlier_idx];
                let nearest_entry = od.nearest_entry;
                // Which surviving cluster the neighbour sits in; omitted for
                // same-status fallback neighbours and pre-field reports.
                let cluster_tag = od
                    .nearest_cluster
                    .map(|c| format!(" (cluster {c})"))
                    .unwrap_or_default();
                let total: f64 = distance.iter().map(|(_, _, _, sq)| *sq).sum::<f64>().sqrt();
                let dominant_name = dominant
                    .and_then(|name| distance.iter().find(|(n, ..)| *n == name))
                    .map(|(_, label, ..)| *label)
                    .unwrap_or("none");
                println!(
                    "    Nearest inlier neighbour: Payload {nearest_entry}{cluster_tag}, total distance {total:.2}\u{3c3} (dominant: {dominant_name})"
                );
            } else if !family_has_inliers {
                let family_size = fuzzing
                    .response_features
                    .iter()
                    .filter(|f| f.status_code == feature.status_code)
                    .count();
                println!(
                    "    No inlier neighbours: status {} has only {family_size} response(s)",
                    feature.status_code
                );
            }
        }
    }
}

/// Per-feature sigma distances of one outlier: (feature name, display label,
/// sigma distance or None, squared component) in the scaled-matrix column
/// order. A None sigma means the feature was absent (timing disabled).
fn feature_distances(
    d: &clustering::OutlierDistance,
) -> Vec<(&'static str, &'static str, Option<f64>, f64)> {
    vec![
        ("length", "Length", Some(d.length_sq.sqrt()), d.length_sq),
        (
            "ttfb",
            "TTFB",
            d.ttfb_sq.map(|sq| sq.sqrt()),
            d.ttfb_sq.unwrap_or(0.0),
        ),
        ("words", "Words", Some(d.words_sq.sqrt()), d.words_sq),
        ("lines", "Lines", Some(d.lines_sq.sqrt()), d.lines_sq),
        (
            "content type",
            "Content-Type",
            Some(d.content_type_sq.sqrt()),
            d.content_type_sq,
        ),
    ]
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let end = s.floor_char_boundary(max_len);
        format!("{}...", &s[..end])
    }
}
