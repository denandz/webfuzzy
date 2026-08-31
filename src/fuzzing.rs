use std::sync::Arc;

use indicatif::{ProgressBar, ProgressStyle};
use tokio::sync::Semaphore;

use crate::baseline::BaselineResult;
use crate::cli::Args;
use crate::marker;
use crate::clustering::{self, DbscanResult};
use crate::http_client::{HttpClient, RequestResult, ResponseFeatures};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FuzzingResult {
    pub total_requests: usize,
    pub successful_requests: usize,
    pub failed_requests: usize,
    pub status_codes: std::collections::HashMap<u16, usize>,
    pub clustering_result: DbscanResult,
    pub response_features: Vec<ResponseFeatures>,
    /// Pearson correlation between input payload length and response content length.
    /// `None` when fewer than 2 successful requests are available.
    pub payload_length_correlation: Option<f64>,
    /// Baseline TTFB reference (status, mean, sigma) used to scale the TTFB
    /// feature for the baseline's status family; `None` when unavailable.
    #[serde(default)]
    pub ttfb_baseline_ref: Option<(u16, f64, f64)>,
}

pub async fn run_fuzzing(
    client: &HttpClient,
    args: &Args,
    baseline: &BaselineResult,
) -> Result<FuzzingResult, Box<dyn std::error::Error>> {
    let url = args.url.clone().ok_or("URL is required for fuzzing")?;
    let wordlist = load_wordlist(args)?;
    if wordlist.is_empty() {
        return Ok(create_empty_result());
    }

    println!(
        "Running outlier-based fuzzing with {} payloads...",
        wordlist.len()
    );

    let progress: Option<Arc<ProgressBar>>;
    let pb = ProgressBar::new(wordlist.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{pos}/{len}] {msg}")
            .unwrap(),
    );
    progress = Some(Arc::new(pb));

    let mut status_codes = std::collections::HashMap::new();
    let mut successful = 0;
    let mut failed = 0;
    let mut request_results = Vec::new();

    let headers = args.parse_headers();
    let marker_bytes = args.marker.as_bytes().to_vec();
    let body_template_bytes = args.data.as_ref().map(|s| s.as_bytes().to_vec());

    // Fuzzing requires at least one marker span (§...§) in the URL, body,
    // or a header key/value template; without it there is nowhere to
    // inject the payloads.
    let url_has_span = marker::has_span(url.as_bytes(), &marker_bytes);
    let body_has_span = body_template_bytes
        .as_ref()
        .map_or(false, |b| marker::has_span(b, &marker_bytes));
    let header_has_span = marker::headers_have_span(&headers, &marker_bytes);
    if !url_has_span && !body_has_span && !header_has_span {
        return Err(format!(
            "no payload span found in URL, body, or headers: wrap the original value in '{}' pairs, e.g. ?id=§1§ or -H 'X-Api-Key: §key§'",
            args.marker
        )
        .into());
    }

    // NOTE: header key/value fuzzing and hyper's parsing logic.
    // Payload bytes injected into headers are validated by the `http`
    // crate when `send_request` builds the request:
    // - A header NAME must be a valid RFC 9110 token (ASCII: letters,
    //   digits, !#$%&'*+-.^_`|~). If a payload makes the key invalid,
    //   `Request::builder` enters an error state and `.body()` fails, so
    //   the request is never sent and is logged as a failure.
    // - CR (0x0D) and LF (0x0A) are rejected
    //   True CRLF injection would require a raw-socket client.
    // Header names are lowercased on the wire (h2 requires it; hyper's
    // h1 encoder emits the HeaderMap's normalized lowercase form).
    // There are ways to have non-cannonicalized headers with h1, which is
    // worth investigating https://github.com/hyperium/hyper/issues/1492

    if args.threads <= 1 {
        for (i, payload) in wordlist.iter().enumerate() {
            let payload_display = String::from_utf8_lossy(payload).to_string();
            let url_with_payload =
                marker::inject_span_str(&url, &args.marker, &payload_display);
            let body_bytes = body_template_bytes
                .as_ref()
                .map(|template| marker::inject_span(template, &marker_bytes, payload));
            let headers_with_payload =
                marker::inject_headers(&headers, &marker_bytes, payload);

            let result = client
                .send_request(
                    &url_with_payload,
                    args.method.as_deref().unwrap(),
                    &headers_with_payload,
                    body_bytes.as_deref(),
                )
                .await;

            let request_result = match result {
                Ok((request, response)) => RequestResult {
                    request,
                    response: Some(response),
                    error: None,
                    sequence: i,
                    payload_byte_len: payload.len(),
                },
                Err((request, e)) => RequestResult {
                    request,
                    response: None,
                    error: Some(e),
                    sequence: i,
                    payload_byte_len: payload.len(),
                },
            };

            if let Some(ref response) = request_result.response {
                *status_codes.entry(response.status_code).or_insert(0) += 1;
                successful += 1;
                if let Some(ref pb) = progress {
                    let truncated = truncate_payload(&payload_display, 30);
                    pb.set_message(format!(
                        "{:>3} {:>4}ms {}",
                        response.status_code,
                        response.total_time.as_millis(),
                        truncated
                    ));
                    pb.inc(1);
                }
            } else {
                failed += 1;
                if let Some(ref pb) = progress {
                    let truncated = truncate_payload(&payload_display, 30);
                    pb.set_message(format!("{} ERR", truncated));
                    pb.inc(1);
                }
            }

            request_results.push(request_result);
        }
    } else {
        let semaphore = Arc::new(Semaphore::new(args.threads));
        let mut handles = Vec::new();

        for (i, payload) in wordlist.iter().enumerate() {
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let client_clone = client.clone();
            let url = url.clone();
            let method = args.method.clone().unwrap();
            let body_template = body_template_bytes.clone();
            let payload_clone = payload.clone();
            let headers_clone = headers.clone();
            let marker = marker_bytes.clone();
            let marker_str = args.marker.clone();
            let pb_clone = progress.clone();

            let handle = tokio::spawn(async move {
                let payload_display = String::from_utf8_lossy(&payload_clone).to_string();
                let url_with_payload =
                    marker::inject_span_str(&url, &marker_str, &payload_display);
                let body_bytes = body_template
                    .as_ref()
                    .map(|template| marker::inject_span(template, &marker, &payload_clone));
                let headers_with_payload =
                    marker::inject_headers(&headers_clone, &marker, &payload_clone);

                let result = client_clone
                    .send_request(
                        &url_with_payload,
                        &method,
                        &headers_with_payload,
                        body_bytes.as_deref(),
                    )
                    .await;

                drop(permit);

                match result {
                    Ok((request, response)) => {
                        if let Some(ref pb) = pb_clone {
                            let truncated = truncate_payload(&payload_display, 30);
                            pb.set_message(format!(
                                "{:>3} {:>4}ms {}",
                                response.status_code,
                                response.total_time.as_millis(),
                                truncated
                            ));
                            pb.inc(1);
                        }
                        Some(RequestResult {
                            request,
                            response: Some(response),
                            error: None,
                            sequence: i,
                            payload_byte_len: payload_clone.len(),
                        })
                    }
                    Err((request, e)) => {
                        if let Some(ref pb) = pb_clone {
                            let truncated = truncate_payload(&payload_display, 30);
                            pb.set_message(format!("{} ERR", truncated));
                            pb.inc(1);
                        }
                        Some(RequestResult {
                            request,
                            response: None,
                            error: Some(e),
                            sequence: i,
                            payload_byte_len: payload_clone.len(),
                        })
                    }
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            if let Ok(Some(result)) = handle.await {
                if let Some(ref response) = result.response {
                    *status_codes.entry(response.status_code).or_insert(0) += 1;
                    successful += 1;
                } else {
                    failed += 1;
                }
                request_results.push(result);
            }
        }
    }

    request_results.sort_by_key(|a| a.sequence);

    let response_features: Vec<ResponseFeatures> = request_results
        .iter()
        .enumerate()
        .filter_map(|(_, r)| {
            r.response.as_ref().map(|resp| {
                let mut features = ResponseFeatures::from(resp);
                features.content_length_minus_payload =
                    resp.body.len() as i64 - r.payload_byte_len as i64;
                features.payload =
                    Some(String::from_utf8_lossy(&wordlist[r.sequence]).into_owned());
                features
            })
        })
        .collect();

    if let Some(ref pb) = progress {
        pb.finish_and_clear();
    }

    println!(
        "{}",
        format!(
            "[+] Fuzzing complete: {successful} successful, {failed} failed requests (+ {} baseline)",
            baseline.request_count
        )
    );
    println!("{}", "[.] Analyzing features...");
    if args.disable_timing {
        println!("[.] Timing analysis disabled - TTFB excluded from clustering");
    }

    let ttfb_ref = (!args.disable_timing)
        .then(|| {
            clustering::ttfb_baseline_ref(
                &baseline.status_codes,
                &baseline.ttfbs,
                args.timing_jitter,
            )
        })
        .flatten();

    let clustering_result = clustering::perform_clustering(
        &response_features,
        CLUSTER_TOLERANCE,
        CLUSTER_MIN_SAMPLES,
        args.max_clusters,
        !args.disable_timing,
        ttfb_ref,
        args.timing_jitter,
    );

    for r in &baseline.request_results {
        if let Some(ref resp) = r.response {
            *status_codes.entry(resp.status_code).or_insert(0) += 1;
        }
    }

    // Collect (payload_length, response_length) pairs for correlation.
    let pairs: Vec<(f64, f64)> = request_results
        .iter()
        .filter_map(|r| {
            r.response
                .as_ref()
                .map(|resp| (r.payload_byte_len as f64, resp.body.len() as f64))
        })
        .collect();
    let payload_length_correlation = pearson_correlation(&pairs);

    Ok(FuzzingResult {
        total_requests: baseline.request_count + request_results.len(),
        successful_requests: baseline.request_results.len() + successful,
        failed_requests: failed,
        status_codes,
        clustering_result,
        response_features,
        payload_length_correlation,
        ttfb_baseline_ref: ttfb_ref,
    })
}

fn truncate_payload(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let end = s.floor_char_boundary(max);
        format!("{}...", &s[..end])
    }
}

/// DBSCAN tolerance for outlier clustering (in z-scored feature space).
/// This should probably be a cli argument with a sensible default...
pub const CLUSTER_TOLERANCE: f64 = 0.5;
/// DBSCAN min_samples for outlier clustering.
pub const CLUSTER_MIN_SAMPLES: usize = 3;

static DEFAULT_WORDLIST: &[u8] = include_bytes!("../wordlists/CrazyDoIsDiscountFuzzList.txt");

fn load_wordlist(args: &Args) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    let bytes = if let Some(ref path) = args.wordlist {
        std::fs::read(path)?
    } else {
        DEFAULT_WORDLIST.to_vec()
    };

    // Lines are taken verbatim: leading/trailing spaces and tabs are part
    // of the payload. Only the line terminator is stripped (a single
    // trailing `\r` for CRLF files), and empty lines are skipped.
    let words: Vec<Vec<u8>> = bytes
        .split(|&b| b == b'\n')
        .map(|line| {
            let end = if line.last() == Some(&b'\r') {
                line.len() - 1
            } else {
                line.len()
            };
            &line[..end]
        })
        .filter(|line| !line.is_empty())
        .map(|line| line.to_vec())
        .collect();
    Ok(words)
}

fn create_empty_result() -> FuzzingResult {
    FuzzingResult {
        total_requests: 0,
        successful_requests: 0,
        failed_requests: 0,
        status_codes: std::collections::HashMap::new(),
        clustering_result: DbscanResult {
            clusters: Vec::new(),
            outliers: Vec::new(),
            noise_count: 0,
            reflection: Vec::new(),
        },
        response_features: Vec::new(),
        payload_length_correlation: None,
        ttfb_baseline_ref: None,
    }
}

/// Compute Pearson correlation coefficient between two sets of values.
/// Returns `None` if fewer than 2 data points or if either variable has zero variance.
fn pearson_correlation(pairs: &[(f64, f64)]) -> Option<f64> {
    let n = pairs.len() as f64;
    if n < 2.0 {
        return None;
    }

    let sum_x: f64 = pairs.iter().map(|&(x, _)| x).sum();
    let sum_y: f64 = pairs.iter().map(|&(_, y)| y).sum();
    let sum_xy: f64 = pairs.iter().map(|&(x, y)| x * y).sum();
    let sum_x2: f64 = pairs.iter().map(|&(x, _)| x * x).sum();
    let sum_y2: f64 = pairs.iter().map(|&(_, y)| y * y).sum();

    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

    if denominator == 0.0 {
        return None;
    }

    Some(numerator / denominator)
}
