//! Latin-1 / UTF-8 handling without mandatory decoding.

/// Source encoding of the CSV body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    Utf8,
    Latin1,
}

/// Cheap detection: sample validated as UTF-8; an incomplete multibyte
/// sequence at the sample edge still counts as UTF-8, real invalid bytes mean
/// Latin-1.
pub fn detect(data: &[u8]) -> Encoding {
    let sample = if data.len() > 65_536 {
        &data[..65_536]
    } else {
        data
    };
    match core::str::from_utf8(sample) {
        Ok(_) => Encoding::Utf8,
        Err(e) if e.error_len().is_none() => Encoding::Utf8,
        Err(_) => Encoding::Latin1,
    }
}

/// Append Latin-1 bytes as UTF-8 text.
pub fn latin1_to_string(src: &[u8], out: &mut String) {
    out.reserve(src.len());
    for &b in src {
        out.push(b as char);
    }
}

/// Decode a raw field into `out` honouring `enc`.
pub fn decode_text(raw: &[u8], enc: Encoding, out: &mut String) {
    match enc {
        Encoding::Utf8 => match core::str::from_utf8(raw) {
            Ok(s) => out.push_str(s),
            Err(_) => latin1_to_string(raw, out),
        },
        Encoding::Latin1 => latin1_to_string(raw, out),
    }
}

/// Borrow raw bytes as `&str` when already UTF-8.
pub fn as_utf8(raw: &[u8]) -> Option<&str> {
    core::str::from_utf8(raw).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_utf8() {
        assert_eq!(detect("ção".as_bytes()), Encoding::Utf8);
    }

    #[test]
    fn detects_latin1() {
        assert_eq!(detect(&[0xE7, 0xE3, 0xF5]), Encoding::Latin1);
    }

    #[test]
    fn latin1_roundtrip() {
        let mut s = String::new();
        latin1_to_string(&[0xE7, 0xE3, b'o'], &mut s);
        assert_eq!(s, "ção");
    }
}
