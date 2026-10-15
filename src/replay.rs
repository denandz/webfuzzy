//! Offline re-analysis of a previous run's report file
//! (`report_*.json`). Re-computes the DBSCAN clustering from the recorded
//! responses and displays the results without sending any requests.

use std::error::Error;
use std::path::Path;

use crate::cli::Args;
use crate::clustering;
use crate::fuzzing::{self, FuzzingResult};
use crate::logging::Report;
use crate::reporting;

/// Read an end-of-run report file (`report_*.json`) from a previous run.
pub fn load_report(path: &Path) -> Result<Report, Box<dyn Error>> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("could not read '{}': {}", path.display(), e))?;
    let report: Report = serde_json::from_slice(&bytes).map_err(|e| {
        format!("'{}' is not a valid webfuzzy report file: {}", path.display(), e)
    })?;
    Ok(report)
}

/// Re-compute the clustering from a stored report.
///
/// Response features are taken verbatim from the recorded run, and the
/// baseline TTFB reference is rebuilt from the recorded baseline (so the
/// current `--timing-jitter` floor applies). The current `--max-clusters`
/// and `--disable-timing` settings shape the re-computed result.
pub fn reanalyze(report: &Report, args: &Args) -> Result<FuzzingResult, Box<dyn Error>> {
    let stored = report.fuzzing.as_ref().ok_or_else(|| -> Box<dyn Error> {
        "report contains no fuzzing data (it was written by an older webfuzzy \
         version) - only runs recorded with fuzzing features can be re-analyzed"
            .into()
    })?;
    if stored.response_features.is_empty() {
        return Err(
            "report recorded no successful fuzzing responses to cluster".into(),
        );
    }

    let ttfb_ref = (!args.disable_timing)
        .then(|| {
            report.baseline.as_ref().and_then(|b| {
                clustering::ttfb_baseline_ref(&b.status_codes, &b.ttfbs, args.timing_jitter)
            })
        })
        .flatten();

    let clustering_result = clustering::perform_clustering(
        &stored.response_features,
        fuzzing::CLUSTER_TOLERANCE,
        fuzzing::CLUSTER_MIN_SAMPLES,
        args.max_clusters,
        !args.disable_timing,
        ttfb_ref,
        args.timing_jitter,
    );

    Ok(FuzzingResult {
        total_requests: stored.total_requests,
        successful_requests: stored.successful_requests,
        failed_requests: stored.failed_requests,
        status_codes: stored.status_codes.clone(),
        clustering_result,
        response_features: stored.response_features.clone(),
        payload_length_correlation: stored.payload_length_correlation,
        ttfb_baseline_ref: ttfb_ref,
    })
}

/// Read one file, re-compute, print. No network I/O.
pub fn run_replay(args: &Args, path: &Path) -> Result<(), Box<dyn Error>> {
    println!("{}", "Webfuzzy - offline re-analysis");
    println!("{}", "=".repeat(40));
    println!("Report: {}", path.display());

    let report = load_report(path)?;
    let info = &report.scan_info;
    println!("Target: {}", info.target_url);
    println!("Method: {}", info.method);
    println!("Run ID: {}", info.run_id);
    println!(
        "Scanned: {} -> {}",
        info.start_time,
        info.end_time.as_deref().unwrap_or("incomplete")
    );
    println!();

    let result = reanalyze(&report, args)?;

    if let Some(stored) = &report.fuzzing {
        println!(
            "[.] Original run: {} cluster(s), {} outlier(s) - recomputed with current settings: {} cluster(s), {} outlier(s)",
            stored.clustering_result.clusters.len(),
            stored.clustering_result.noise_count,
            result.clustering_result.clusters.len(),
            result.clustering_result.noise_count,
        );
        println!();
    }

    if args.disable_timing {
        println!("[.] Timing analysis disabled - TTFB excluded from clustering");
        println!();
    }

    reporting::print_fuzzing_summary(&result);
    reporting::print_outlier_details(&result);
    Ok(())
}
