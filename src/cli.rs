use clap::Parser;
use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// Args follow curl flags where possible.
#[derive(Parser, Serialize, Deserialize, Clone, Debug)]
#[command(
    name = "webfuzzy",
    about = "Web security testing tool for vulnerability hunting against HTTP endpoints",
    version,
    long_about = "Webfuzzy is a fuzzer for web application security assessments.\nThe tool identifies input-based vulnerabilities through fuzzing and outlier-analysis with DBScan clustering."
)]
pub struct Args {
    /// Target URL
    #[arg(short, long, required = true)]
    pub url: Option<String>,

    /// HTTP headers in format 'Name: Value' (repeatable). Wrap the original
    /// value in marker pairs to fuzz it, in the key ('§X-Api§-Key: v') and/or
    /// the value ('X-Api-Key: §key§').
    #[arg(short = 'H', long = "header", action = clap::ArgAction::Append)]
    pub headers: Vec<String>,

    /// Request body/data (for POST/PUT/etc)
    #[arg(short, long)]
    pub data: Option<String>,

    /// HTTP method (GET, POST, PUT, DELETE, etc.)
    #[arg(short = 'X', long = "request", default_value = "GET")]
    pub method: String,

    /// Number of baseline requests to send
    #[arg(long, default_value = "10")]
    pub baseline_count: usize,

    /// Fuzzing wordlist file path. Defaults to built-in wordlist.
    #[arg(long, short = 'w')]
    pub wordlist: Option<PathBuf>,

    /// Output directory for logs
    #[arg(long, short = 'o', default_value = ".")]
    pub output: PathBuf,

    /// Payload span delimiter (default: §). Wrap the original value of a
    /// fuzzable parameter in a marker pair, e.g. `?id=§1§`: baseline
    /// requests send the original value without markers (`?id=1`) and
    /// fuzzing runs replace the whole span (markers and original value)
    /// with the payload
    #[arg(long, default_value = "§")]
    pub marker: String,

    /// Number of concurrent requests for fuzzing (lower values improve timing analysis accuracy)
    #[arg(long, default_value = "5")]
    pub threads: usize,

    /// Request timeout in seconds
    #[arg(long, default_value = "30")]
    pub timeout: u64,

    /// Skip baseline testing
    #[arg(long)]
    pub skip_baseline: bool,

    /// Verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// Maximum number of clusters to report
    #[arg(long, default_value = "6")]
    pub max_clusters: usize,

    /// Disable timing analysis
    #[arg(long, default_value = "false")]
    pub disable_timing: bool,
}

impl Args {
    pub fn parse_headers(&self) -> Vec<(String, String)> {
        let mut headers: Vec<(String, String)> = self
            .headers
            .iter()
            .filter_map(|h| {
                let (key, value) = h.split_once(':')?;
                Some((key.trim().to_string(), value.trim().to_string()))
            })
            .collect();

        if self.data.is_some()
            && !headers
                .iter()
                .any(|(k, _)| k.eq_ignore_ascii_case("content-type"))
        {
            headers.push((
                "Content-Type".to_string(),
                "application/x-www-form-urlencoded".to_string(),
            ));
        }

        headers
    }
}

/// Runtime configuration
///
/// Wraps the parsed CLI arguments plus a per-execution `run_id` that
/// uniquely identifies a single run of the tool.
#[derive(Serialize, Deserialize)]
pub struct Config {
    /// Unique identifier for this execution (8 random lowercase hex chars).
    pub run_id: String,
    /// RFC 3339 timestamp (millisecond precision) of when the run started.
    pub started_at: String,
    /// Full runtime flags and parameters parsed from the command line.
    pub args: Args,
}

impl Config {
    /// Build a config from parsed CLI args, generating a fresh run id.
    pub fn new(args: Args) -> Self {
        let started_at = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
        Self {
            run_id: Self::generate_run_id(),
            started_at,
            args,
        }
    }

    /// Generate a random 8-hex-char run id.
    pub fn generate_run_id() -> String {
        // 4 bytes will produce an 8-character hex string
        let num: u32 = rand::rng().random();
        format!("{:08x}", num)
    }
}
