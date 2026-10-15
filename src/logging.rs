use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::baseline::BaselineResult;
use crate::fuzzing::FuzzingResult;
use crate::http_client::{HttpRequest, HttpResponse};

/// Real-time jsonl logs bigger than this (1MB) are xz-compressed on close.
const XZ_THRESHOLD: u64 = 1_000_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub summary: ScanSummary,
    pub version: String,
    pub scan_info: ScanInfo,
    pub baseline: Option<BaselineResult>,
    /// Full fuzzing stage result (response features + clustering) so a
    /// previous run can be re-analyzed offline (`--report`) without
    /// re-sending requests.
    #[serde(default)]
    pub fuzzing: Option<FuzzingResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanInfo {
    pub tool_version: String,
    /// Unique identifier for this execution (see `cli::Config::run_id`).
    #[serde(default)]
    pub run_id: String,
    pub target_url: String,
    pub method: String,
    pub start_time: String,
    pub end_time: Option<String>,
    pub arguments: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanSummary {
    pub total_requests: u64,
    pub total_responses: u64,
    pub total_errors: u64,
    pub baseline_stable: bool,
}

#[derive(Clone)]
pub struct AuditLogger {
    inner: Arc<Mutex<AuditLoggerInner>>,
}

struct AuditLoggerInner {
    output_dir: PathBuf,
    /// Per-run filename suffix (run hash) preventing same-second
    /// filename collisions between runs.
    file_id: String,
    /// Path of the real-time jsonl request log.
    log_path: PathBuf,
    log_file: Option<File>,
    total_requests: u64,
    total_responses: u64,
    total_errors: u64,
}

impl AuditLogger {
    /// Create a logger for one execution.
    ///
    /// `run_id` (the run hash) is embedded in every output filename so that
    /// runs starting within the same second can never truncate each other's
    /// logs. `File::create` truncates, and the timestamp alone only has
    /// second precision.
    ///
    /// The `.jsonl` file is written per-request-and-reponse (unbuffered `File`,
    /// one syscall per entry) and is the definitive real-time record of every
    /// request/response sent. The end-of-run `.json` files are convenience snapshots.
    pub fn new(output_dir: &PathBuf, run_id: &str) -> Self {
        std::fs::create_dir_all(output_dir).ok();

        // run ID is used to create the filenames
        let id = run_id.to_string();

        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let log_path = output_dir.join(format!("audit_{}_{id}.jsonl", timestamp));
        // Opened read+write: close() re-reads the file to xz-compress it
        // when it exceeds 1MB.
        let log_file = match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&log_path)
        {
            Ok(f) => Some(f),
            Err(e) => {
                eprintln!(
                    "[!] could not create audit log '{}': {} - raw request data will NOT be recorded",
                    log_path.display(),
                    e
                );
                None
            }
        };

        Self {
            inner: Arc::new(Mutex::new(AuditLoggerInner {
                output_dir: output_dir.clone(),
                file_id: id,
                log_path,
                log_file,
                total_requests: 0,
                total_responses: 0,
                total_errors: 0,
            })),
        }
    }

    /// Log an http request
    pub fn log_request(&self, request: &HttpRequest) {
        let mut inner = self.inner.lock().unwrap();
        inner.total_requests += 1;

        if let Some(ref mut file) = inner.log_file {
            let entry = serde_json::json!({
                "type": "request",
                "timestamp": Utc::now().to_rfc3339(),
                "data": request
            });

            if let Err(e) = writeln!(file, "{}", entry) {
                eprintln!("[!] failed to write audit log entry: {}", e);
            }
        }
    }

    /// Log an http response
    pub fn log_response(&self, response: &HttpResponse) {
        let mut inner = self.inner.lock().unwrap();
        inner.total_responses += 1;

        if let Some(ref mut file) = inner.log_file {
            let entry = serde_json::json!({
                "type": "response",
                "timestamp": Utc::now().to_rfc3339(),
                "data": response
            });

            if let Err(e) = writeln!(file, "{}", entry) {
                eprintln!("[!] failed to write audit log entry: {}", e);
            }
        }
    }

    /// Log an error
    pub fn log_error(&self, id: &String, err: &String) {
        let mut inner = self.inner.lock().unwrap();
        inner.total_errors += 0;

        if let Some(ref mut file) = inner.log_file {
            let entry = serde_json::json!({
                "type": "error",
                "request_id": id,
                "timestamp": Utc::now().to_rfc3339(),
                "data": err
            });

            if let Err(e) = writeln!(file, "{}", entry) {
                eprintln!("  [!] failed to write audit log entry: {}", e);
            }
        }
    }

    /// Path of the real-time jsonl request log for this run.
    pub fn jsonl_path(&self) -> PathBuf {
        self.inner.lock().unwrap().log_path.clone()
    }

    /// Close the real-time log. XZ-compressed if it's larger than 1MB
    /// and the original is removed. `jsonl_path` ends up pointing at the
    /// `.xz` file.
    pub fn close(&self) {
        let mut inner = self.inner.lock().unwrap();
        let Some(file) = inner.log_file.take() else {
            return;
        };
        let len = file.metadata().map(|m| m.len()).unwrap_or(0);
        if len <= XZ_THRESHOLD {
            return;
        }

        let xz_path = inner.log_path.with_extension("jsonl.xz");
        match compress_xz(file, &xz_path) {
            Ok(()) => {
                if let Err(e) = std::fs::remove_file(&inner.log_path) {
                    eprintln!(
                        "[!] compressed audit log but could not remove '{}': {} (both files kept)",
                        inner.log_path.display(),
                        e
                    );
                }
                eprintln!(
                    "[*] audit log was {} bytes (>1MB), saved as '{}'",
                    len,
                    xz_path.display()
                );
                inner.log_path = xz_path;
            }
            Err(e) => eprintln!(
                "[!] failed to compress audit log '{}': {} - keeping the uncompressed file",
                inner.log_path.display(),
                e
            ),
        }
    }

    /// Write the single end-of-run report file: scan metadata (including the
    /// run hash), baseline, and the full fuzzing result (features +
    /// clusters) in one JSON document. The report is a summary snapshot; the
    /// raw per-request record lives exclusively in the real-time audit jsonl
    /// (`jsonl_path`).
    pub fn save_report(
        &self,
        scan_info: &ScanInfo,
        baseline: &BaselineResult,
        fuzzing: &FuzzingResult,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let inner = self.inner.lock().unwrap();

        let summary = ScanSummary {
            total_requests: inner.total_requests,
            total_responses: inner.total_responses,
            total_errors: inner.total_errors,
            baseline_stable: baseline.stable,
        };

        let report = Report {
            summary,
            version: env!("CARGO_PKG_VERSION").to_string(),
            scan_info: scan_info.clone(),
            baseline: Some(baseline.clone()),
            fuzzing: Some(fuzzing.clone()),
        };

        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let output_path = inner
            .output_dir
            .join(format!("report_{}_{}.json", timestamp, inner.file_id));
        let json = serde_json::to_string_pretty(&report)?;
        let len = json.len().try_into().unwrap_or(0);
        std::fs::write(&output_path, json)?;

        if len <= XZ_THRESHOLD {
            return Ok(output_path);
        }

        let xz_path = output_path.with_extension("json.xz");
        match compress_xz_path(&output_path, &xz_path) {
            Ok(()) => {
                if let Err(e) = std::fs::remove_file(&output_path) {
                    eprintln!(
                        "[!] compressed report but could not remove '{}': {} (both files kept)",
                        inner.log_path.display(),
                        e
                    );
                }
                eprintln!(
                    "[*] report log was {} bytes (>1MB), saved as '{}'",
                    len,
                    xz_path.display()
                );
                Ok(xz_path)
            }
            Err(e) => {
                eprintln!(
                    "[!] failed to compress report '{}': {} - keeping the uncompressed file",
                    inner.log_path.display(),
                    e
                );
                Err(Box::new(e))
            }
        }
    }
}

/// Stream `file` through xz (preset 6) into `xz_path`.
fn compress_xz(file: File, xz_path: &Path) -> std::io::Result<()> {
    // The log fd's offset sits at the end after the writes; restart at 0.
    let mut file = file;
    file.seek(SeekFrom::Start(0))?;
    let mut encoder = xz2::read::XzEncoder::new(file, 6);
    let mut out = File::create(xz_path)?;
    std::io::copy(&mut encoder, &mut out)?;
    out.sync_all()
}

fn compress_xz_path(in_path: &Path, xz_path: &Path) -> std::io::Result<()> {
    let file = File::open(in_path)?;
    compress_xz(file, xz_path)
}
