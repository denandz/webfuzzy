# webfuzzy

A web security testing tool for vulnerability hunting against HTTP endpoints. Webfuzzy bridges automated analysis and manual testing by speeding up outlier-based fuzzing with DBSCAN clustering.

# Architecture

- Auditability. All HTTP requests and responses sent/recieved by webfuzzy are written to a `jsonl` file in the current directory by default. If the user needs to investigate some condition such as a server crash, this log contains all data sent/recieved by the `http_client.rs` HTTP client.

# License

Webfuzzy is released under BSD 3-Clause. See [LICENSE](https://github.com/denandz/webfuzzy/blob/main/LICENSE).
