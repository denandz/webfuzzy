# webfuzzy

A web security testing tool for vulnerability hunting against HTTP endpoints. Webfuzzy bridges automated analysis and manual testing by speeding up outlier-based fuzzing with DBSCAN clustering.

Webfuzzy is still a work-in-progress, and needs a few more features before its fully ready. Currently limited to a single payload position per-run.

Build with `cargo build`. Quick start:

```
cargo run -- --url https://target/api/v1/get?id=§1§
```

Section sign (§) characters can be added to the command line with `ctrl+shift+u a7` on most Linux systems. 

# Usage

```
Web security testing tool for vulnerability hunting against HTTP endpoints

Usage: webfuzzy [OPTIONS] --url <URL>

Options:
  -u, --url <URL>
          Target URL
  -H, --header <HEADERS>
          HTTP headers in format 'Name: Value' (repeatable). Wrap the original value in marker pairs to fuzz it, in the key ('§X-Api§-Key: v') and/or the value ('X-Api-Key: §key§')
  -d, --data <DATA>
          Request body/data (for POST/PUT/etc)
  -X, --request <METHOD>
          HTTP method (GET, POST, PUT, DELETE, etc.) [default: GET]
      --baseline-count <BASELINE_COUNT>
          Number of baseline requests to send [default: 10]
  -w, --wordlist <WORDLIST>
          Fuzzing wordlist file path. Defaults to built-in wordlist
  -o, --output <OUTPUT>
          Output directory for logs [default: .]
      --marker <MARKER>
          Payload span delimiter (default: §). Wrap the original value of a fuzzable parameter in a marker pair, e.g. `?id=§1§`: baseline requests send the original value without markers (`?id=1`) and fuzzing runs replace the whole span (markers and original value) with the payload [default: §]
      --threads <THREADS>
          Number of concurrent requests for fuzzing (lower values improve timing analysis accuracy) [default: 5]
      --timeout <TIMEOUT>
          Request timeout in seconds [default: 30]
      --skip-baseline
          Skip baseline testing
  -v, --verbose
          Verbose output
      --max-clusters <MAX_CLUSTERS>
          Maximum number of clusters to report [default: 6]
      --disable-timing
          Disable timing analysis
  -h, --help
          Print help (see more with '--help')
  -V, --version
          Print version
```

# Architecture

- DBSCAN clustering of response data to classify endpoint behaviour quickly, and find outliers (noise). - Curl argument interop. Arguments are as-close-as-possible to `curl`, so any tool that offers a `copy as curl` function (browsers, testing tools) can easily be pasted into a webfuzzy run
- Sensible defaults, aiming for minimal runtime configuration.
- Auditability. All HTTP requests and responses sent/recieved by webfuzzy are written to a `jsonl` file in the current directory by default. If the user needs to investigate some condition such as a server crash, this log contains all data sent/recieved by the `http_client.rs` HTTP client.

# Clustering

WebFuzzy uses DBSCAN to cluster response data into groups, with the aim of quickly identifying how a target API or application behaves.

Here is a quick PNG that simplifies the concept with only two features (two dimensions):

![](_img/DBSCAN.png)

The above shows a stable 200 response, a 404 response that has variable content lengths (likely reflecting the input payload), and a timing outlier that warrants further investigation.

Current features used for clustering are:

- Status Code - applied categorically, clustering is run per-status code since a difference of HTTP/200 and HTTP/201 is a very meaningful difference for an API. Euclidian distance between response data points for HTTP/200 and HTTP/201, for example, shouldn't be clustered together.
- Response content length in bytes.
- Time-to-first byte timing data.
- Response content word count.
- Response content line count.
- Response content type, one-hot encoded. One column per common response type (the list lives in `wordlists/content_types.txt`, one type per line), plus one column for missing content-type headers and one for types not on the list.

These raw values are not directly comparable (bytes vs milliseconds vs counts), so each status family is z-scored against its own mean and standard deviation before DBSCAN runs (think [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)). That also does the right thing for the one-hot content-type columns: a type used by only a few responses sits many sigma away from the bulk, so a handful of `application/json` responses among `text/html` split into two clusters, and a single weird content-length is an outlier. Content-type matching ignores parameters (`text/html; charset=utf-8` matches the `text/html` column) but is case-sensitive on purpose: a server answering `Text/HTML` is weird and we want to see that.

Two further adjustments keep the length and timing features from producing false outliers:

- TTFB jitter scaling. Timing on a fast network is dominated by millisecond-scale wobble. The family that matches the baseline's mode status is scaled against the baseline's own mean and standard deviation (a control group of un-injected requests), and every family's TTFB standard deviation is floored at `--timing-jitter` (default 50 ms). TLDR: by default we don't really cluster on timing thats less than ~50ms of wobbly.
- Reflection regression. An endpoint that reflects the input payload (think reflected XSS) makes response length track payload length, which fragments the clusters into one point per payload length. When the Pearson correlation between payload length and response length within a family reaches 0.8, the length feature is replaced by the regression residual (`content_length - slope * payload_length`); the slope covers decoders as well, where the reflected length is a fraction of the payload length. Word and line counts are driven by the reflected payload too, and the server's exact mapping is unknown, so they are squashed to zero for that family.

A deliberate side effect of the reflection handling: payloads whose encoding the server transforms (heavy percent-encoding, charset tricks) decode to a different fraction than the population average and keep a residual, surfacing as small clusters or outliers. This is intentional! These are the payloads worth a human look since they indicate some form of encoding/decoding behaviour that's worth investigating!

The goal is 'weird handling of data, or odd encoding' gets its own cluster or gets flagged as an outlier.

# License

Webfuzzy is released under BSD 3-Clause. See [LICENSE](https://github.com/denandz/webfuzzy/blob/main/LICENSE).
