//! Offline re-analysis of a previous run's report file
//! (`report_*.json` or its xz-compressed form `report_*.json.xz`).
//! Re-computes the DBSCAN clustering from the recorded responses and
//! displays the results without sending any requests.

use std::error::Error;
use std::io::Read;
use std::path::Path;

use crate::cli::Args;
use crate::clustering;
use crate::fuzzing::{self, FuzzingResult};
use crate::logging::Report;
use crate::reporting;

/// Read an end-of-run report file from a previous run.
///
/// Accepts both plain `report_*.json` files and the xz-compressed
/// `report_*.json.xz` form that large reports are automatically saved as.
/// Compression is transparent: the JSON is parsed directly out of the
/// decoded stream.
pub fn load_report(path: &Path) -> Result<Report, Box<dyn Error>> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("could not read '{}': {}", path.display(), e))?;

    let mut reader: Box<dyn Read> = if path.extension().is_some_and(|e| e == "xz") {
        Box::new(xz2::read::XzDecoder::new(file))
    } else {
        Box::new(file)
    };

    let report: Report = serde_json::from_reader(&mut reader).map_err(|e| {
        format!(
            "'{}' is not a valid webfuzzy report file: {}",
            path.display(),
            e
        )
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
        return Err("report recorded no successful fuzzing responses to cluster".into());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http_client::ResponseFeatures;
    use std::path::PathBuf;
    use std::collections::HashMap;

    /// Scratch dir removed on drop (avoids a `tempfile` dev-dependency).
    struct TmpDir(PathBuf);
    impl TmpDir {
        fn new(tag: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("webfuzzy_test_{}_{}", tag, std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            TmpDir(path)
        }
    }
    impl Drop for TmpDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn sample_report() -> Report {
        let features = ResponseFeatures {
            status_code: 200,
            content_type: "text/html".into(),
            response_length: 100,
            time_to_first_byte_ms: 42,
            response_words: 10,
            response_lines: 2,
            content_length_minus_payload: 0,
            payload: Some("x".into()),
        };
        let fuzzing = crate::fuzzing::FuzzingResult {
            total_requests: 1,
            successful_requests: 1,
            failed_requests: 0,
            status_codes: [(200u16, 1)].into(),
            clustering_result: clustering::DbscanResult {
                clusters: Vec::new(),
                outliers: Vec::new(),
                noise_count: 1,
                reflection: Vec::new(),
                outlier_distances: HashMap::new(),
            },
            response_features: vec![features],
            payload_length_correlation: None,
            ttfb_baseline_ref: None,
        };
        Report {
            summary: crate::logging::ScanSummary {
                total_requests: 1,
                total_responses: 1,
                total_errors: 0,
                baseline_stable: true,
            },
            version: "test".into(),
            scan_info: crate::logging::ScanInfo {
                tool_version: "test".into(),
                run_id: "test".into(),
                target_url: "https://example.com".into(),
                method: "GET".into(),
                start_time: "2026-01-01T00:00:00Z".into(),
                end_time: Some("2026-01-01T00:01:00Z".into()),
                arguments: Vec::new(),
            },
            baseline: None,
            fuzzing: Some(fuzzing),
        }
    }

    #[test]
    fn load_report_plain_json() {
        let dir = TmpDir::new("plain");
        let path = dir.0.join("report_test.json");
        std::fs::write(&path, serde_json::to_vec(&sample_report()).unwrap()).unwrap();

        let report = load_report(&path).unwrap();
        assert_eq!(report.fuzzing.as_ref().unwrap().total_requests, 1);
    }

    #[test]
    fn load_report_json_xz() {
        let dir = TmpDir::new("xz");
        let path = dir.0.join("report_test.json.xz");
        // Compress the same way logging.rs::compress_xz does: the xz
        // encoder is a Read stream that compresses on the fly.
        let plain_path = dir.0.join("report_test.json");
        let plain = serde_json::to_vec(&sample_report()).unwrap();
        std::fs::write(&plain_path, &plain).unwrap();
        let mut enc = xz2::read::XzEncoder::new(std::fs::File::open(&plain_path).unwrap(), 6);
        let mut out = std::fs::File::create(&path).unwrap();
        std::io::copy(&mut enc, &mut out).unwrap();
        out.sync_all().unwrap();

        let report = load_report(&path).unwrap();
        assert_eq!(report.fuzzing.as_ref().unwrap().total_requests, 1);
        assert_eq!(report.scan_info.target_url, "https://example.com");
    }

    #[test]
    fn load_report_rejects_invalid_file() {
        let dir = TmpDir::new("invalid");
        let path = dir.0.join("report_test.json");
        std::fs::write(&path, "not json").unwrap();
        assert!(load_report(&path).is_err());
    }
}
