//! Payload span markers.
//!
//! The marker (default: `§`, the section sign) delimits the original
//! value of a fuzzable parameter: For example, in
//! `example.com/api/endpoint?id=§1§` the original value is `1`.
//!
//! - **Baseline**: every span is collapsed to its inner content, so the
//!   request is sent with the original parameter data and no markers
//!   (`?id=1`).
//! - **Fuzzing**: the first span markers *and* original data included
//!   is replaced by the payload (`?id=<payload>`); any further spans are
//!   collapsed to their inner content.
//!
//! All functions operate on raw bytes, so multi-byte markers (`§` is
//! `C2 A7` in UTF-8) and binary body templates work identically. A
//! trailing marker without a closing pair is left as a literal.

/// Byte ranges `(start, end)` of every complete marker span in `template`,
/// left to right, non-overlapping. A span runs from one marker to the next,
/// markers included; the content between them is the original value.
pub fn find_spans(template: &[u8], marker: &[u8]) -> Vec<(usize, usize)> {
    if marker.is_empty() {
        return Vec::new();
    }
    let mut spans = Vec::new();
    let mut i = 0;
    while let Some(rel) = find_byte_seq(&template[i..], marker) {
        let open = i + rel;
        match find_byte_seq(&template[open + marker.len()..], marker) {
            Some(rel2) => {
                let end = open + marker.len() + rel2 + marker.len();
                spans.push((open, end));
                i = end;
            }
            None => break, // unmatched trailing marker
        }
    }
    spans
}

/// `true` when the template contains at least one complete marker span.
pub fn has_span(template: &[u8], marker: &[u8]) -> bool {
    !find_spans(template, marker).is_empty()
}

/// Inner content (original value) of the first marker span, if any.
pub fn first_span_value<'a>(template: &'a [u8], marker: &[u8]) -> Option<&'a [u8]> {
    find_spans(template, marker)
        .first()
        .map(|(s, e)| &template[s + marker.len()..*e - marker.len()])
}

/// Replace every marker span with its inner content (markers removed).
pub fn collapse_spans(template: &[u8], marker: &[u8]) -> Vec<u8> {
    build(template, marker, None)
}

/// Replace the first marker span with `payload`; collapse any further
/// spans to their inner content. Templates without a span are returned
/// unchanged.
pub fn inject_span(template: &[u8], marker: &[u8], payload: &[u8]) -> Vec<u8> {
    build(template, marker, Some(payload))
}

/// String convenience wrappers over the byte-level functions.
pub fn collapse_spans_str(template: &str, marker: &str) -> String {
    String::from_utf8_lossy(&collapse_spans(template.as_bytes(), marker.as_bytes())).into_owned()
}

pub fn inject_span_str(template: &str, marker: &str, payload: &str) -> String {
    String::from_utf8_lossy(&inject_span(
        template.as_bytes(),
        marker.as_bytes(),
        payload.as_bytes(),
    ))
    .into_owned()
}

fn build(template: &[u8], marker: &[u8], first_payload: Option<&[u8]>) -> Vec<u8> {
    let spans = find_spans(template, marker);
    let mut out = Vec::with_capacity(template.len());
    let mut prev = 0;
    for (n, (start, end)) in spans.iter().enumerate() {
        out.extend_from_slice(&template[prev..*start]);
        match (n, first_payload) {
            (0, Some(payload)) => out.extend_from_slice(payload),
            _ => out.extend_from_slice(&template[start + marker.len()..*end - marker.len()]),
        }
        prev = *end;
    }
    out.extend_from_slice(&template[prev..]);
    out
}

fn find_byte_seq(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    (0..=haystack.len() - needle.len())
        .find(|&i| &haystack[i..i + needle.len()] == needle)
}
