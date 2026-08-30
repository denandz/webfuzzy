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
        Some(r) => println!("  Payload-response correlation: {r:.4}"),
        None => println!("  Payload-response correlation: N/A"),
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
                println!("    '{}'", truncate(payload, 70));
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
                truncate(&feature.payload.clone().unwrap_or_default(), 40)
            );
            println!("    Status: {}", feature.status_code);
            println!("    Length: {} bytes", feature.response_length);
            println!("    TTFB: {}ms", feature.time_to_first_byte_ms);
        }
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let end = s.floor_char_boundary(max_len);
        format!("{}...", &s[..end])
    }
}
