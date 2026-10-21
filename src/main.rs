use clap::Parser;
use webfuzzy::baseline;
use webfuzzy::cli::{Args, Config};
use webfuzzy::fuzzing;
use webfuzzy::http_client::HttpClient;
use webfuzzy::logging::{AuditLogger, ScanInfo};
use webfuzzy::replay;
use webfuzzy::reporting;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // parse arguments and generate a config struct
    let args = Args::parse();
    let config = Config::new(args);
    let args = &config.args;
    let method = args.method.as_deref().unwrap();

    // Offline mode: re-compute clustering from a previous run's report
    // file without sending any requests. No audit logger - the input file
    // is only ever read.
    if let Some(ref report_path) = args.report {
        return replay::run_replay(args, report_path);
    }

    let url = args
        .url
        .clone()
        .ok_or("either --url (live scan) or --report <file> (offline re-analysis) is required")?;

    println!("{}", "Webfuzzy - Web Request Fuzzer");
    println!("{}", "=".repeat(40));
    println!("Target: {url}");
    println!("Method: {}", method);
    println!("Run ID: {}", config.run_id);
    println!("Threads: {}", config.args.threads);
    println!();

    let audit_logger = AuditLogger::new(&args.output, &config.run_id);
    let http_client = HttpClient::new(args, audit_logger.clone())?;

    // Probe with the original request (marker spans collapsed), like the baseline.
    let headers = webfuzzy::marker::collapse_headers(&args.parse_headers(), args.marker.as_bytes());
    let probe_body: Option<Vec<u8>> = args
        .data
        .as_ref()
        .map(|s| webfuzzy::marker::collapse_spans(s.as_bytes(), args.marker.as_bytes()));
    let probe_url = webfuzzy::marker::collapse_spans_str(&url, &args.marker);
    let initial_request = match http_client
        .send_request(&probe_url, method, &headers, probe_body.as_deref())
        .await
    {
        Ok((request, response)) => {
            println!("{}", format!("Initial request: {} {}", method, probe_url));
            println!("Status: {}", response.status_code);
            println!(
                "Content-Type: {}",
                response
                    .headers
                    .iter()
                    .find(|(k, _)| k == "content-type")
                    .map(|(_, v)| v.as_str())
                    .unwrap_or("unknown")
            );
            println!("Length: {} bytes", response.body.len());
            println!("TTFB: {}ms", response.time_to_first_byte.as_millis());
            println!();

            Some((request, response))
        }
        Err((_, e)) => {
            eprintln!("{}", format!("[!] Initial request failed: {e}"));
            None
        }
    };

    // initial request failed, no point continuing
    if initial_request.is_none() {
        audit_logger.close();
        return Ok(());
    }

    // run baselines to determine request stability
    println!("{}", "[+] STAGE 1: BASELINE");
    println!("{}", "-".repeat(21));

    let baseline_results = baseline::run_baseline(&http_client, args).await?;

    reporting::print_baseline_summary(&baseline_results);
    println!();

    // TODO if the request isnt stable (timing, status code, etc) then prompt the user to continue

    // Run the fuzzing and determine reseponse clusters
    println!("{}", "[+] STAGE 2: FUZZING");
    println!("{}", "-".repeat(20));

    let fuzz_result = fuzzing::run_fuzzing(&http_client, args, &baseline_results).await?;
    reporting::print_fuzzing_summary(&fuzz_result);
    reporting::print_outlier_details(&fuzz_result);

    // End-of-run report: scan metadata, baseline, and the full fuzzing
    // result so the run can be re-analyzed offline with `--report`.
    let scan_info = ScanInfo {
        tool_version: env!("CARGO_PKG_VERSION").to_string(),
        run_id: config.run_id.clone(),
        target_url: url.clone(),
        method: method.to_string(),
        start_time: config.started_at.clone(),
        end_time: Some(chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)),
        arguments: std::env::args().collect(),
    };
    match audit_logger.save_report(&scan_info, &baseline_results, &fuzz_result) {
        Ok(path) => println!("\n[+] Report saved to '{}'", path.display()),
        Err(e) => eprintln!("[!] failed to save report: {e}"),
    }

    audit_logger.close();
    Ok(())
}
