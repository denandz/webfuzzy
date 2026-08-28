use crate::encoding::{percent_encode_path, percent_encode_query};
use http_body_util::{BodyExt, Full};
use hyper::Request;
use hyper::body::Bytes;
use hyper_util::client::legacy::Client as HyperClient;
use hyper_util::rt::TokioExecutor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use std::time::{Duration, Instant};

use crate::cli::Args;
use crate::logging::AuditLogger;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpRequest {
    pub request_id: String,
    pub method: String,
    pub url: String,
    pub headers: HashMap<String, String>,
    pub body: Option<String>,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpResponse {
    pub request_id: String,
    pub status_code: u16,
    pub headers: HashMap<String, String>,
    pub body: String,
    pub content_length: usize,
    pub response_words: usize,
    pub response_lines: usize,
    pub time_to_first_byte: Duration,
    pub total_time: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestResult {
    pub request: HttpRequest,
    pub response: Option<HttpResponse>,
    pub error: Option<String>,
    pub sequence: usize,
    #[serde(default)]
    pub payload_byte_len: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFeatures {
    pub status_code: u16,
    pub content_type: String,
    pub response_length: usize,
    pub time_to_first_byte_ms: u64,
    pub response_words: usize,
    pub response_lines: usize,
    pub content_length_minus_payload: i64,
    pub payload: Option<String>,
}

impl From<&HttpResponse> for ResponseFeatures {
    fn from(response: &HttpResponse) -> Self {
        Self {
            status_code: response.status_code,
            content_type: response
                .headers
                .get("content-type")
                .cloned()
                .unwrap_or_default(),
            response_length: response.body.len(),
            time_to_first_byte_ms: response.time_to_first_byte.as_millis() as u64,
            response_words: response.response_words,
            response_lines: response.response_lines,
            // Populated later by the caller once payload byte length is known.
            content_length_minus_payload: 0,
            payload: None,
        }
    }
}

/// Hyper HTTP client wrapper. Sends raw `http::Uri` without usual URL normalization.
///
/// Unlike `reqwest`, hyper does NOT normalize dot-segments (`../`) in paths.
/// The URI is sent on the wire exactly as constructed.
pub struct HttpClient {
    client: HyperClient<HttpConnector, Full<Bytes>>,
    args: Args,
    audit_logger: AuditLogger,
    request_counter: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl Clone for HttpClient {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            args: self.args.clone(),
            audit_logger: self.audit_logger.clone(),
            request_counter: self.request_counter.clone(),
        }
    }
}

/// Type alias for our HTTP(S) connector.
/// Uses `hyper-rustls` for TLS, with optional insecure mode.
type HttpConnector =
    hyper_rustls::HttpsConnector<hyper_util::client::legacy::connect::HttpConnector>;

impl HttpClient {
    pub fn new(args: &Args, audit_logger: AuditLogger) -> Result<Self, Box<dyn std::error::Error>> {
        // Install ring as the default crypto provider for rustls
        rustls::crypto::ring::default_provider()
            .install_default()
            .ok();

        // Build HTTP connector
        let mut http = hyper_util::client::legacy::connect::HttpConnector::new();
        http.set_nodelay(true);
        http.enforce_http(false); // Allow non-HTTP URIs to pass through

        // Build rustls config
        let tls_builder = rustls::ClientConfig::builder();
        // Insecure: skip certificate verification entirely
        let no_verify = NoVerifier {};
        let tls_config = tls_builder
            .dangerous()
            .with_custom_certificate_verifier(std::sync::Arc::new(no_verify))
            .with_no_client_auth();

        // Wrap HTTP connector with HTTPS support
        let https: HttpConnector = hyper_rustls::HttpsConnectorBuilder::new()
            .with_tls_config(tls_config)
            .https_or_http()
            .enable_all_versions()
            .wrap_connector(http);

        // Build hyper client
        let builder = hyper_util::client::legacy::Client::builder(TokioExecutor::new());
        let client = builder.build(https);

        Ok(Self {
            client,
            args: args.clone(),
            audit_logger,
            request_counter: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(1)),
        })
    }

    pub async fn send_request(
        &self,
        url: &str,
        method: &str,
        headers: &[(String, String)],
        body: Option<&[u8]>,
    ) -> Result<(HttpRequest, HttpResponse), (HttpRequest, String)> {
        // build the HttpRequest
        let timestamp = chrono::Utc::now().to_rfc3339();
        let mut request = HttpRequest {
            request_id: uuid::Uuid::new_v4().to_string(),
            method: method.to_string(),
            url: url.to_string(),
            headers: headers.iter().cloned().collect(),
            body: body.map(|b| String::from_utf8_lossy(b).to_string()),
            timestamp,
        };

        // Use url::Url to properly percent-encode query parameters (preserving #,
        // spaces, etc.) while keeping the path raw (no dot-segment normalization).
        let uri = match build_uri(url) {
            Ok(uri) => uri,
            Err(e) => return Err((request, e.to_string())),
        };

        request.url = uri.to_string();

        let mut builder = Request::builder().method(method).uri(uri.clone());

        let mut header_map: HashMap<String, String> = HashMap::new();
        for (name, value) in headers {
            builder = builder.header(
                name.as_str(),
                hyper::header::HeaderValue::from_bytes(value.as_bytes()).unwrap_or_else(|_| {
                    let fallback = String::from_utf8_lossy(value.as_bytes()).to_string();
                    if self.args.verbose {
                        eprintln!(
                            "  [WARN] header {} fallback (invalid bytes): \"{}\" -> \"{}\"",
                            name,
                            String::from_utf8_lossy(value.as_bytes()),
                            fallback,
                        );
                    }
                    hyper::header::HeaderValue::from_str(&fallback)
                        .unwrap_or(hyper::header::HeaderValue::from_static(""))
                }),
            );
            header_map.insert(name.clone(), value.clone());
        }

        if !header_map.contains_key("User-Agent") {
            builder = builder.header(
                "User-Agent",
                format!("webfuzzy/{}", env!("CARGO_PKG_VERSION")),
            );
        }
        //builder = builder.header("Host", uri.host().unwrap_or(""));

        let body_bytes: Vec<u8> = body.unwrap_or(&[]).to_vec();
        let hyper_req = match builder.body(Full::from(Bytes::from(body_bytes))) {
            Ok(hyper_req) => hyper_req,
            Err(e) => return Err((request, e.to_string())),
        };
        
        self.audit_logger.log_request(&request);

        let start = Instant::now();

        // Execute with timeout
        let response_result = tokio::time::timeout(
            Duration::from_secs(self.args.timeout),
            self.client.request(hyper_req),
        )
        .await;

        let ttfb = start.elapsed();

        let response_result: Result<
            hyper::Response<hyper::body::Incoming>,
            hyper_util::client::legacy::Error,
        > = match response_result {
            Ok(Ok(res)) => Ok(res),
            Ok(Err(e)) => Err(e),
            Err(_) => {
                let io_err = std::io::Error::new(std::io::ErrorKind::TimedOut, "timeout");
                return Err((request, io_err.to_string()))
            }
        };

        match response_result {
            Ok(response) => {
                let status_code = response.status().as_u16();

                let response_headers: HashMap<String, String> = response
                    .headers()
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
                    .collect();

                // Collect body bytes
                let collected = match response.into_body().collect().await {
                    Ok(collected) => collected,
                    Err(e) => return Err((request, e.to_string())),
                };
                let response_text = String::from_utf8_lossy(&collected.to_bytes()).to_string();
                let total_time = start.elapsed();

                let words = response_text.split_whitespace().count();
                let lines = response_text.lines().count();
                let content_length = response_text.len();

                let response = HttpResponse {
                    request_id: request.request_id.clone(),
                    status_code,
                    headers: response_headers,
                    body: response_text,
                    content_length,
                    response_words: words,
                    response_lines: lines,
                    time_to_first_byte: ttfb,
                    total_time,
                };

                self.audit_logger.log_response(&response);

                Ok((request, response))
            }
            Err(e) => {
                let error_msg = e.to_string();

                self.audit_logger.log_error(&request.request_id, &error_msg);

                Err((request, error_msg))
            }
        }
    }

    pub fn audit_logger(&self) -> &AuditLogger {
        &self.audit_logger
    }
}

/// Build an `http::Uri` from a URL string.
///
/// - Preserves path raw (no dot-segment normalization, a/../../../q is left as a literal path)
/// - Percent-encodes the path and query with the bare-minimum RFC 9110
///   encoding: only bytes that would make the request target invalid are
///   encoded, so payloads (e.g. `!'`) reach the wire unmodified
fn build_uri(url: &str) -> Result<http::Uri, String> {
    // Extract scheme + authority from original string (no normalization)
    let (scheme_authority, original_path, raw_query) = match url.find("://") {
        Some(scheme_end) => {
            let after_scheme = &url[scheme_end + 3..];
            let path_start = after_scheme
                .find('/')
                .map(|i| i + scheme_end + 3)
                .unwrap_or(url.len());
            let scheme_authority = &url[..path_start];

            let (original_path, raw_query) = if path_start < url.len() {
                let rest = &url[path_start..];
                if let Some(q_pos) = rest.find('?') {
                    (&rest[..q_pos], Some(&rest[q_pos + 1..]))
                } else {
                    (rest, None)
                }
            } else {
                ("/", None)
            };

            (scheme_authority, original_path, raw_query)
        }
        None => return Err(format!("Invalid URL '{}': missing scheme", url)),
    };

    // Validate that the scheme:authority is well-formed (not used further;
    // the URI is reconstructed from the original string, not this parse).
    url::Url::parse(scheme_authority).map_err(|e| format!("Invalid URL '{}': {}", url, e))?;

    // Percent-encode path and query with the bare-minimum encoding so
    // payload bytes are not rewritten beyond protocol requirements.
    let encoded_path = percent_encode_path(original_path);

    let uri_string = match raw_query {
        Some(query) => format!(
            "{}{}?{}",
            scheme_authority,
            encoded_path,
            percent_encode_query(query)
        ),
        None => format!("{}{}", scheme_authority, encoded_path),
    };

    http::Uri::from_str(&uri_string).map_err(|e| format!("Invalid URI '{}': {}", uri_string, e))
}

/// No-op certificate verifier for insecure mode.
#[derive(Debug)]
struct NoVerifier {}

impl rustls::client::danger::ServerCertVerifier for NoVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls_pki_types::CertificateDer<'_>,
        _intermediates: &[rustls_pki_types::CertificateDer<'_>],
        _server_name: &rustls_pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls_pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls_pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls_pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        // Return common signature schemes
        vec![
            rustls::SignatureScheme::RSA_PKCS1_SHA256,
            rustls::SignatureScheme::RSA_PKCS1_SHA384,
            rustls::SignatureScheme::RSA_PKCS1_SHA512,
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::ECDSA_NISTP521_SHA512,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA384,
            rustls::SignatureScheme::RSA_PSS_SHA512,
            rustls::SignatureScheme::ED25519,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_payload_bytes_pass_through() {
        let uri = build_uri("http://example.com/test?foo=bar!').drop();--").unwrap();
        assert_eq!(
            uri.to_string(),
            "http://example.com/test?foo=bar!').drop();--"
        );
    }

    #[test]
    fn query_illegal_bytes_are_encoded() {
        // space and non-ASCII (the § marker) must be pct-encoded to keep
        // the request line legal; existing %XX is preserved verbatim
        let uri = build_uri("http://example.com/test?foo=%C2%A7bar%C2%A7 baz%41%").unwrap();
        assert_eq!(
            uri.to_string(),
            "http://example.com/test?foo=%C2%A7bar%C2%A7%20baz%41%25"
        );
    }

    #[test]
    fn fragment_hash_in_query_is_encoded_not_dropped() {
        let uri = build_uri("http://example.com/?a=1#x").unwrap();
        assert_eq!(uri.to_string(), "http://example.com/?a=1%23x");
    }

    #[test]
    fn path_is_preserved_raw() {
        let uri = build_uri("http://example.com/a/../../../q?x=1").unwrap();
        assert_eq!(uri.to_string(), "http://example.com/a/../../../q?x=1");
    }
}
