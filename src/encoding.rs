use std::fmt::Write;

/// Percent-encode bytes that are illegal in a URI path, preserving
/// existing valid `%XX` sequences.
///
/// Only the bare minimum required by RFC 9110 is encoded (path segments
/// allow `pchar` plus `/`): control bytes, space, DEL, non-ASCII bytes,
/// and a `%` that does not begin a valid `%XX` sequence. Sub-delims such
/// as `!'` are sent as-is: this tool is a fuzzer, so payloads must not be
/// rewritten beyond what keeps the request line legal, or non-decoding
/// servers would see a different payload than intended (false negatives).
pub fn percent_encode_path(path: &str) -> String {
    encode(path, false)
}

/// Like `percent_encode_path`, but for a query string (`query = *( pchar /
/// "/" / "?" )`): `?` is additionally legal, and `#` must be encoded since
/// it would otherwise start a fragment.
pub fn percent_encode_query(query: &str) -> String {
    encode(query, true)
}

fn encode(input: &str, in_query: bool) -> String {
    let mut result = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'%' {
            // Preserve valid %XX sequences verbatim (wire fidelity); a
            // lone/malformed % would make the request target invalid, so
            // it must be encoded.
            if i + 2 < bytes.len() && is_hex(bytes[i + 1]) && is_hex(bytes[i + 2]) {
                result.push('%');
                result.push(bytes[i + 1] as char);
                result.push(bytes[i + 2] as char);
                i += 3;
            } else {
                result.push_str("%25");
                i += 1;
            }
        } else if is_safe_byte(bytes[i], in_query) {
            result.push(bytes[i] as char);
            i += 1;
        } else {
            write!(result, "%{:02X}", bytes[i]).ok();
            i += 1;
        }
    }

    result
}

/// Check if a byte is a valid hex digit.
fn is_hex(b: u8) -> bool {
    b.is_ascii_hexdigit()
}

/// Bytes legal verbatim in a path segment (plus `/`); in a query, `?`
/// is also legal.
fn is_safe_byte(b: u8, in_query: bool) -> bool {
    matches!(b,
        b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9'
        | b'-' | b'.' | b'_' | b'~'
        | b'!' | b'$' | b'&' | b'\'' | b'(' | b')' | b'*' | b'+'
        | b',' | b';' | b'=' | b':' | b'@'
        | b'/')
        || (in_query && b == b'?')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_preserves_sub_delims() {
        // sub-delims and other legal query bytes must go out literally
        assert_eq!(percent_encode_query("foo=!'"), "foo=!'");
        assert_eq!(
            percent_encode_query("a=1&b=$()*,;=+:@/?.~_-"),
            "a=1&b=$()*,;=+:@/?.~_-"
        );
    }

    #[test]
    fn query_encodes_only_illegal_bytes() {
        assert_eq!(percent_encode_query("a b"), "a%20b"); // space
        assert_eq!(percent_encode_query("a#b"), "a%23b"); // fragment delim
        assert_eq!(percent_encode_query("a\tb"), "a%09b"); // control
        assert_eq!(percent_encode_query("a§b"), "a%C2%A7b"); // non-ASCII
        assert_eq!(percent_encode_query("100%"), "100%25"); // bare %
    }

    #[test]
    fn query_preserves_valid_pct_encoded() {
        // existing %XX is kept verbatim (not double-encoded)
        assert_eq!(percent_encode_query("a=%41b"), "a=%41b");
        assert_eq!(percent_encode_query("a=%zz"), "a=%25zz");
    }

    #[test]
    fn path_encoding_unchanged() {
        assert_eq!(percent_encode_path("/a/§b?c"), "/a/%C2%A7b%3Fc"); // ? delimits the query
        assert_eq!(percent_encode_path("/!'"), "/!'");
        assert_eq!(percent_encode_path("/100%"), "/100%25");
    }
}
