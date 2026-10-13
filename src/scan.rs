//! Zero-copy record and field iteration over raw CSV bytes.
//!
//! Everything borrows from the input slice. No allocation happens on the parse
//! path; quote/escape materialisation is opt-in via `Field::decode_into`.

use crate::simd::{self, Backend, BlockMasks};

pub const MAX_COLS: usize = 64;

/// Field-scan window: 16 bytes (one narrow SIMD load).
const FIELD_WINDOW: usize = 16;

/// CSV dialect: delimiter, quote, record terminator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dialect {
    pub quote: u8,
    pub sep: u8,
    pub nl: u8,
}

impl Dialect {
    /// TSE open data: `;` delimited, `"` quoted, CRLF tolerated.
    pub const TSE: Dialect = Dialect {
        quote: b'"',
        sep: b';',
        nl: b'\n',
    };

    pub const fn new(quote: u8, sep: u8, nl: u8) -> Self {
        Dialect { quote, sep, nl }
    }
}

impl Default for Dialect {
    fn default() -> Self {
        Dialect::TSE
    }
}

/// Per-block result: delimiter positions that sit outside quoted fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScanBlock {
    pub quote: u64,
    pub sep_outside: u64,
    pub nl_outside: u64,
    pub n: usize,
}

/// Quote-aware block scanner. Carries quote parity across blocks and records.
pub struct BlockScanner {
    pub backend: Backend,
    dialect: Dialect,
    in_quotes: bool,
}

impl BlockScanner {
    pub fn new(dialect: Dialect) -> Self {
        Self::with_backend(Backend::detect(), dialect)
    }

    pub fn with_backend(backend: Backend, dialect: Dialect) -> Self {
        BlockScanner {
            backend,
            dialect,
            in_quotes: false,
        }
    }

    pub fn in_quotes(&self) -> bool {
        self.in_quotes
    }

    /// Scan one chunk of at most `simd::BLOCK` bytes. Masks are zero beyond
    /// `chunk.len()`. Advances quote state as if the whole chunk was consumed;
    /// call `rewind_to` when a delimiter cut the chunk short.
    #[inline]
    pub fn scan(&mut self, chunk: &[u8]) -> ScanBlock {
        debug_assert!(chunk.len() <= simd::BLOCK);
        let m: BlockMasks =
            self.backend
                .scan(chunk, self.dialect.quote, self.dialect.sep, self.dialect.nl);
        let p = simd::prefix_xor(m.quote);
        let carry: u64 = if self.in_quotes { u64::MAX } else { 0 };
        // `inside[i]` = quote-state immediately before byte i (garbage past n).
        let inside = (p << 1) ^ carry;
        let parity = ((p >> 63) & 1) == 1;
        self.in_quotes ^= parity;
        ScanBlock {
            quote: m.quote,
            sep_outside: m.sep & !inside,
            nl_outside: m.nl & !inside,
            n: chunk.len(),
        }
    }

    /// Roll quote state back to just after byte `consumed` of the last scanned
    /// chunk (state was advanced over the full chunk by `scan`).
    #[inline]
    pub fn rewind_to(&mut self, quote: u64, consumed: usize, n: usize) {
        debug_assert!(consumed <= n);
        if consumed >= n {
            return;
        }
        let bits = n - consumed;
        let mut rest = quote >> consumed;
        if bits < 64 {
            rest &= (1u64 << bits) - 1;
        }
        self.in_quotes ^= (rest.count_ones() & 1) == 1;
    }
}

fn trim_cr(bytes: &[u8]) -> &[u8] {
    match bytes.split_last() {
        Some((b'\r', rest)) => rest,
        _ => bytes,
    }
}

/// One CSV record (a full line, quotes preserved, CR stripped).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Record<'a> {
    bytes: &'a [u8],
    start: usize,
    end: usize,
}

impl<'a> Record<'a> {
    /// Wrap pre-sliced record bytes (no newline expected).
    pub fn from_raw(bytes: &'a [u8]) -> Self {
        Record {
            bytes,
            start: 0,
            end: bytes.len(),
        }
    }

    pub fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Byte offset of record start within the scanned buffer.
    pub fn start(&self) -> usize {
        self.start
    }

    /// Byte offset just past the record terminator.
    pub fn end(&self) -> usize {
        self.end
    }

    pub fn fields(&self, dialect: Dialect) -> FieldIter<'a> {
        FieldIter::new(self.bytes, dialect)
    }

    /// Split into at most `MAX_COLS` field slices. Returns field count.
    pub fn split(&self, dialect: Dialect, buf: &mut [&'a [u8]; MAX_COLS]) -> usize {
        let mut n = 0;
        for field in self.fields(dialect) {
            if n == MAX_COLS {
                break;
            }
            buf[n] = field.as_bytes();
            n += 1;
        }
        n
    }
}

/// Iterator over raw records. BOM-aware, CRLF-tolerant, quote-aware.
pub struct RecordIter<'a> {
    data: &'a [u8],
    pos: usize,
    scanner: BlockScanner,
}

impl<'a> RecordIter<'a> {
    pub fn new(data: &'a [u8], dialect: Dialect) -> Self {
        Self::with_backend(data, Backend::detect(), dialect)
    }

    pub fn with_backend(data: &'a [u8], backend: Backend, dialect: Dialect) -> Self {
        let pos = if data.starts_with(&[0xEF, 0xBB, 0xBF]) {
            3
        } else {
            0
        };
        RecordIter {
            data,
            pos,
            scanner: BlockScanner::with_backend(backend, dialect),
        }
    }

    fn finish_record(&mut self, start: usize, end: usize) -> Record<'a> {
        self.pos = end + 1;
        Record {
            bytes: trim_cr(&self.data[start..end]),
            start,
            end: end + 1,
        }
    }
}

impl<'a> Iterator for RecordIter<'a> {
    type Item = Record<'a>;

    fn next(&mut self) -> Option<Record<'a>> {
        if self.pos >= self.data.len() {
            return None;
        }
        let start = self.pos;
        let mut i = start;
        while self.data.len() - i >= simd::BLOCK {
            let blk = self.scanner.scan(&self.data[i..i + simd::BLOCK]);
            if blk.nl_outside != 0 {
                let idx = blk.nl_outside.trailing_zeros() as usize;
                self.scanner.rewind_to(blk.quote, idx + 1, blk.n);
                return Some(self.finish_record(start, i + idx));
            }
            i += simd::BLOCK;
        }
        if i < self.data.len() {
            let blk = self.scanner.scan(&self.data[i..]);
            if blk.nl_outside != 0 {
                let idx = blk.nl_outside.trailing_zeros() as usize;
                self.scanner.rewind_to(blk.quote, idx + 1, blk.n);
                return Some(self.finish_record(start, i + idx));
            }
        }
        self.pos = self.data.len();
        Some(Record {
            bytes: trim_cr(&self.data[start..]),
            start,
            end: self.data.len(),
        })
    }
}

/// One CSV field: raw slice plus quote-aware accessors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Field<'a> {
    raw: &'a [u8],
}

impl<'a> Field<'a> {
    pub fn from_raw(raw: &'a [u8]) -> Self {
        Field { raw }
    }

    /// Raw bytes including surrounding quotes when present.
    pub fn as_bytes(&self) -> &'a [u8] {
        self.raw
    }

    pub fn is_quoted(&self) -> bool {
        self.raw.len() >= 2 && self.raw[0] == b'"' && self.raw[self.raw.len() - 1] == b'"'
    }

    /// Borrowed value when no escape processing is required (`""` absent).
    pub fn as_borrowed(&self) -> Option<&'a [u8]> {
        if self.is_quoted() {
            let inner = &self.raw[1..self.raw.len() - 1];
            if memchr(b'"', inner).is_some() {
                None
            } else {
                Some(inner)
            }
        } else if memchr(b'"', self.raw).is_some() {
            None
        } else {
            Some(self.raw)
        }
    }

    /// Borrowed UTF-8 value when possible.
    pub fn as_str(&self) -> Option<&'a str> {
        self.as_borrowed()
            .and_then(|b| core::str::from_utf8(b).ok())
    }

    /// Fully decoded bytes (strips outer quotes, collapses `""`). Uses `out`
    /// as scratch; only place that may allocate.
    pub fn decode_into(&self, out: &mut Vec<u8>) {
        out.clear();
        match self.as_borrowed() {
            Some(b) => out.extend_from_slice(b),
            None => {
                let inner = if self.is_quoted() {
                    &self.raw[1..self.raw.len() - 1]
                } else {
                    self.raw
                };
                let mut i = 0;
                while i < inner.len() {
                    if inner[i] == b'"' && i + 1 < inner.len() && inner[i + 1] == b'"' {
                        out.push(b'"');
                        i += 2;
                    } else {
                        out.push(inner[i]);
                        i += 1;
                    }
                }
            }
        }
    }

    /// Decode into a stack buffer. Returns written length.
    pub fn decode_into_slice(&self, buf: &mut [u8]) -> Option<usize> {
        match self.as_borrowed() {
            Some(b) if b.len() <= buf.len() => {
                buf[..b.len()].copy_from_slice(b);
                Some(b.len())
            }
            Some(_) => None,
            None => {
                let inner = if self.is_quoted() {
                    &self.raw[1..self.raw.len() - 1]
                } else {
                    self.raw
                };
                let mut n = 0;
                let mut i = 0;
                while i < inner.len() {
                    if n >= buf.len() {
                        return None;
                    }
                    if inner[i] == b'"' && i + 1 < inner.len() && inner[i + 1] == b'"' {
                        buf[n] = b'"';
                        n += 1;
                        i += 2;
                    } else {
                        buf[n] = inner[i];
                        n += 1;
                        i += 1;
                    }
                }
                Some(n)
            }
        }
    }

    pub fn to_u64(&self) -> Option<u64> {
        let mut buf = [0u8; 32];
        let n = self.decode_into_slice(&mut buf)?;
        parse_u64(&buf[..n])
    }

    pub fn to_u32(&self) -> Option<u32> {
        let mut buf = [0u8; 32];
        let n = self.decode_into_slice(&mut buf)?;
        parse_u32(&buf[..n])
    }

    /// Brazilian decimal (`1.234,56`) as integer cents.
    pub fn to_cents(&self) -> Option<u64> {
        let mut buf = [0u8; 32];
        let n = self.decode_into_slice(&mut buf)?;
        parse_cents(&buf[..n])
    }
}

fn memchr(needle: u8, hay: &[u8]) -> Option<usize> {
    hay.iter().position(|&b| b == needle)
}

/// Iterator over fields of one record. Trailing empty field preserved.
pub struct FieldIter<'a> {
    data: &'a [u8],
    pos: usize,
    scanner: BlockScanner,
    finished: bool,
}

impl<'a> FieldIter<'a> {
    pub fn new(data: &'a [u8], dialect: Dialect) -> Self {
        Self::with_backend(data, Backend::detect(), dialect)
    }

    pub fn with_backend(data: &'a [u8], backend: Backend, dialect: Dialect) -> Self {
        FieldIter {
            data,
            pos: 0,
            scanner: BlockScanner::with_backend(backend, dialect),
            finished: false,
        }
    }

    /// Byte offset where the next field starts.
    pub fn pos(&self) -> usize {
        self.pos
    }
}

impl<'a> Iterator for FieldIter<'a> {
    type Item = Field<'a>;

    fn next(&mut self) -> Option<Field<'a>> {
        if self.finished {
            return None;
        }
        let start = self.pos;
        let mut i = start;
        // Small windows: fields are short, scanning the whole record tail per
        // field would re-walk bytes over and over.
        while self.data.len() - i >= FIELD_WINDOW {
            let blk = self.scanner.scan(&self.data[i..i + FIELD_WINDOW]);
            if blk.sep_outside != 0 {
                let idx = blk.sep_outside.trailing_zeros() as usize;
                self.scanner.rewind_to(blk.quote, idx + 1, blk.n);
                self.pos = i + idx + 1;
                return Some(Field {
                    raw: &self.data[start..i + idx],
                });
            }
            i += FIELD_WINDOW;
        }
        if i < self.data.len() {
            let blk = self.scanner.scan(&self.data[i..]);
            if blk.sep_outside != 0 {
                let idx = blk.sep_outside.trailing_zeros() as usize;
                self.scanner.rewind_to(blk.quote, idx + 1, blk.n);
                self.pos = i + idx + 1;
                return Some(Field {
                    raw: &self.data[start..i + idx],
                });
            }
        }
        self.finished = true;
        self.pos = self.data.len();
        Some(Field {
            raw: &self.data[start..],
        })
    }
}

/// Unsigned integer parse over raw bytes (no alloc, no `str` round-trip).
pub fn parse_u64(bytes: &[u8]) -> Option<u64> {
    if bytes.is_empty() {
        return None;
    }
    let mut acc: u64 = 0;
    for &b in bytes {
        if !b.is_ascii_digit() {
            return None;
        }
        acc = acc.checked_mul(10)?.checked_add((b - b'0') as u64)?;
    }
    Some(acc)
}

pub fn parse_u32(bytes: &[u8]) -> Option<u32> {
    parse_u64(bytes).and_then(|v| u32::try_from(v).ok())
}

/// Parse `1.234,56` / `1234,56` / `1234` style money as cents.
pub fn parse_cents(bytes: &[u8]) -> Option<u64> {
    if bytes.is_empty() {
        return None;
    }
    let mut int_part: u64 = 0;
    let mut frac: u64 = 0;
    let mut frac_digits = 0u32;
    let mut seen_comma = false;
    for &b in bytes {
        match b {
            b'.' if !seen_comma => {}
            b',' if !seen_comma => seen_comma = true,
            b'0'..=b'9' if !seen_comma => {
                int_part = int_part.checked_mul(10)?.checked_add((b - b'0') as u64)?;
            }
            b'0'..=b'9' if frac_digits < 2 => {
                frac = frac * 10 + (b - b'0') as u64;
                frac_digits += 1;
            }
            b'0'..=b'9' => {}
            _ => return None,
        }
    }
    if seen_comma && frac_digits == 1 {
        frac *= 10;
    }
    int_part.checked_mul(100)?.checked_add(frac)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn records(data: &[u8]) -> Vec<Vec<u8>> {
        RecordIter::new(data, Dialect::TSE)
            .map(|r| r.as_bytes().to_vec())
            .collect()
    }

    #[test]
    fn splits_plain_records() {
        assert_eq!(
            records(b"a;b\nc;d\n"),
            vec![b"a;b".to_vec(), b"c;d".to_vec()]
        );
    }

    #[test]
    fn tolerates_crlf_and_missing_final_newline() {
        assert_eq!(
            records(b"a;b\r\nc;d"),
            vec![b"a;b".to_vec(), b"c;d".to_vec()]
        );
    }

    #[test]
    fn quoted_newline_stays_inside_record() {
        let data = b"\"a\nb\";c\nd\n";
        assert_eq!(records(data), vec![b"\"a\nb\";c".to_vec(), b"d".to_vec()]);
    }

    #[test]
    fn fields_split_on_unquoted_sep() {
        let rec = Record {
            bytes: b"a;\"b;c\";d",
            start: 0,
            end: 0,
        };
        let got: Vec<_> = rec
            .fields(Dialect::TSE)
            .map(|f| f.as_bytes().to_vec())
            .collect();
        assert_eq!(got, vec![b"a".to_vec(), b"\"b;c\"".to_vec(), b"d".to_vec()]);
    }

    #[test]
    fn trailing_empty_field_preserved() {
        let rec = Record {
            bytes: b"a;",
            start: 0,
            end: 0,
        };
        let got = rec.fields(Dialect::TSE).count();
        assert_eq!(got, 2);
    }

    #[test]
    fn escaped_quotes_decode() {
        let f = Field::from_raw(b"\"a\"\"b\"");
        assert!(f.is_quoted());
        assert_eq!(f.as_borrowed(), None);
        let mut out = Vec::new();
        f.decode_into(&mut out);
        assert_eq!(out, b"a\"b");
    }

    #[test]
    fn quoted_plain_borrows() {
        let f = Field::from_raw(b"\"abc\"");
        assert_eq!(f.as_borrowed(), Some(&b"abc"[..]));
        assert_eq!(f.as_str(), Some("abc"));
    }

    #[test]
    fn number_parsing() {
        assert_eq!(Field::from_raw(b"123").to_u32(), Some(123));
        assert_eq!(Field::from_raw(b"\"123\"").to_u32(), Some(123));
        assert_eq!(Field::from_raw(b"x").to_u32(), None);
        assert_eq!(Field::from_raw(b"1.234,56").to_cents(), Some(123_456));
        assert_eq!(Field::from_raw(b"1234,5").to_cents(), Some(123_450));
        assert_eq!(Field::from_raw(b"7").to_cents(), Some(700));
    }

    #[test]
    fn bom_is_skipped() {
        let mut data = vec![0xEF, 0xBB, 0xBF];
        data.extend_from_slice(b"col\nv\n");
        assert_eq!(records(&data), vec![b"col".to_vec(), b"v".to_vec()]);
    }
}
