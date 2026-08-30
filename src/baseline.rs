use std::collections::HashMap;
use std::time::Duration;

use crate::cli::Args;
use crate::http_client::{HttpClient, RequestResult};
use crate::marker;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BaselineResult {
    pub request_count: usize,
    pub stable: bool,
    pub status_code_consistency: f64,
    pub response_length_consistency: f64,
    pub average_ttfb: Duration,
    pub status_codes: HashMap<u16, usize>,
    pub response_lengths: Vec<usize>,
    pub ttfbs: Vec<u64>,
    pub outliers: Vec<usize>,
    pub request_results: Vec<RequestResult>,
}

pub async fn run_baseline(
    client: &HttpClient,
    args: &Args,
) -> Result<BaselineResult, Box<dyn std::error::Error>> {
    if args.skip_baseline {
        return Ok(BaselineResult {
            request_count: 0,
            stable: true,
            status_code_consistency: 1.0,
            response_length_consistency: 1.0,
            average_ttfb: Duration::from_millis(0),
            status_codes: HashMap::new(),
            response_lengths: Vec::new(),
            ttfbs: Vec::new(),
            outliers: Vec::new(),
            request_results: Vec::new(),
        });
    }

    // The baseline exercises the original request: every marker span
    // (§...§) - in the URL, body, or header keys/values - is collapsed
    // to its inner content, so no markers are sent.
    let headers = marker::collapse_headers(&args.parse_headers(), &args.marker.as_bytes());
    let marker_bytes = args.marker.as_bytes().to_vec();

    let body: Option<Vec<u8>> = args
        .data
        .as_ref()
        .map(|s| marker::collapse_spans(s.as_bytes(), &marker_bytes));

    // The original value inside the first span (URL or body) is what the
    // baseline "injects"; record its byte length so feature extraction can
    // subtract it from the response, mirroring the fuzzing stage.
    let baseline_payload_len = args
        .url
        .as_ref()
        .and_then(|u| marker::first_span_value(u.as_bytes(), &marker_bytes))
        .or_else(|| {
            args.data
                .as_ref()
                .and_then(|d| marker::first_span_value(d.as_bytes(), &marker_bytes))
        })
        .or_else(|| {
            headers
                .iter()
                .find_map(|(k, v)| {
                    marker::first_span_value(k.as_bytes(), &marker_bytes)
                        .or_else(|| marker::first_span_value(v.as_bytes(), &marker_bytes))
                })
        })
        .map(|span| span.len())
        .unwrap_or(0);

    println!(
        "{}",
        format!("Running baseline with {} requests...", args.baseline_count)
    );

    let mut status_codes = HashMap::new();
    let mut response_lengths = Vec::new();
    let mut ttfbs = Vec::new();
    let mut request_results = Vec::new();

    let mut successful = 0;

    for i in 0..args.baseline_count {
        let url = prepare_url(args);
        let req_body = body.as_deref();

        match client
            .send_request(&url, args.method.as_deref().unwrap(), &headers, req_body)
            .await
        {
            Ok((request, response)) => {
                successful += 1;
                status_codes
                    .entry(response.status_code)
                    .and_modify(|e| *e += 1)
                    .or_insert(1);
                response_lengths.push(response.body.len());
                ttfbs.push(response.time_to_first_byte.as_millis() as u64);
                request_results.push(RequestResult {
                    request,
                    response: Some(response),
                    error: None,
                    sequence: i,
                    payload_byte_len: baseline_payload_len,
                });
            }
            Err((_, e)) => {
                if args.verbose {
                    eprintln!("  [!] baseline request {} failed: {}", i, e);
                }
            }
        }
    }

    // Fraction of successful requests that share the most common status code.
    let status_code_consistency = if successful > 0 {
        let most_common_count = status_codes.values().max().copied().unwrap_or(0);
        most_common_count as f64 / successful as f64
    } else {
        0.0
    };

    let response_length_consistency = if response_lengths.len() > 1 {
        let mean = response_lengths.iter().sum::<usize>() as f64 / response_lengths.len() as f64;
        let variance = response_lengths
            .iter()
            .map(|&l| (l as f64 - mean).powi(2))
            .sum::<f64>()
            / response_lengths.len() as f64;
        let std_dev = variance.sqrt();
        let cv = std_dev / mean;
        1.0 - cv.min(1.0)
    } else if response_lengths.is_empty() {
        0.0
    } else {
        1.0
    };

    let average_ttfb = if !ttfbs.is_empty() {
        let sum: u64 = ttfbs.iter().sum();
        Duration::from_millis(sum / ttfbs.len() as u64)
    } else {
        Duration::from_millis(0)
    };

    let mut outliers = Vec::new();
    let mean_length =
        response_lengths.iter().sum::<usize>() as f64 / response_lengths.len().max(1) as f64;

    for (i, &length) in response_lengths.iter().enumerate() {
        let deviation = (length as f64 - mean_length).abs();
        if deviation > mean_length * 0.1 {
            outliers.push(i);
        }
    }

    // TODO this is silly, stability should be all the same status code and reasonably similar
    // response lengths, with some level of similarity on timing.
    let stable =
        successful > 0 && status_code_consistency > 0.95 && response_length_consistency > 0.9;

    if stable {
        println!("{}", "Baseline: Endpoint appears stable");
    } else {
        println!(
            "{}",
            "Baseline: Endpoint shows instability - results may vary"
        );
        if status_code_consistency < 0.95 {
            println!(
                "  Status code consistency: {:.1}%",
                status_code_consistency * 100.0
            );
        }
        if response_length_consistency < 0.9 {
            println!(
                "  Response length consistency: {:.1}%",
                response_length_consistency * 100.0
            );
        }
    }

    println!("  Average TTFB: {:?}", average_ttfb);

    Ok(BaselineResult {
        request_count: successful,
        stable,
        status_code_consistency,
        response_length_consistency,
        average_ttfb,
        status_codes,
        response_lengths,
        ttfbs,
        outliers,
        request_results,
    })
}

/// Returns the baseline URL: every marker span (`§...§`) collapsed to its
/// inner content, so the original parameter value is sent without markers.
fn prepare_url(args: &Args) -> String {
    let url = args.url.clone().unwrap_or_default();
    crate::marker::collapse_spans_str(&url, &args.marker)
}

pub fn analyze_baseline_stability(baseline: &BaselineResult) -> Vec<String> {
    let mut observations = Vec::new();

    if baseline.request_count == 0 {
        observations.push("No baseline requests were made (baseline skipped).".to_string());
        return observations;
    }

    if baseline.stable {
        observations.push(format!(
            "Baseline is stable: {} requests with {:.1}% status code consistency and {:.1}% response length consistency.",
            baseline.request_count,
            baseline.status_code_consistency * 100.0,
            baseline.response_length_consistency * 100.0
        ));
    } else {
        if baseline.status_code_consistency < 0.95 {
            observations.push(format!(
                "Status codes are inconsistent ({:.1}% consistency). The endpoint may be returning different responses.",
                baseline.status_code_consistency * 100.0
            ));
        }
        if baseline.response_length_consistency < 0.9 {
            observations.push(format!(
                "Response lengths are inconsistent ({:.1}% consistency). The endpoint may have dynamic content.",
                baseline.response_length_consistency * 100.0
            ));
        }
    }

    if !baseline.outliers.is_empty() {
        observations.push(format!(
            "Detected {} response(s) with significant length deviation from the mean.",
            baseline.outliers.len()
        ));
    }

    observations
}
