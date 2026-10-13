//! Property-style check: SIMD scanner must match a naive byte-walk reference
//! on adversarial inputs (quotes, escapes, CRLF, chunk boundaries).

use fast_tse_parser::{BlockScanner, Dialect, RecordIter, MAX_COLS};

/// Reference: walk bytes, toggle in_quotes on `"`, split records on `\n` when
/// outside quotes, strip `\r`, split fields on `;` when outside quotes.
fn reference_records(data: &[u8]) -> Vec<Vec<Vec<u8>>> {
    let mut records = Vec::new();
    let mut record: Vec<u8> = Vec::new();
    let mut in_quotes = false;
    for &b in data {
        match b {
            b'"' => {
                in_quotes = !in_quotes;
                record.push(b);
            }
            b'\n' if !in_quotes => {
                if record.last() == Some(&b'\r') {
                    record.pop();
                }
                records.push(reference_fields(&record));
                record.clear();
            }
            _ => record.push(b),
        }
    }
    if !record.is_empty() {
        if record.last() == Some(&b'\r') {
            record.pop();
        }
        records.push(reference_fields(&record));
    }
    records
}

fn reference_fields(record: &[u8]) -> Vec<Vec<u8>> {
    let mut fields = Vec::new();
    let mut field: Vec<u8> = Vec::new();
    let mut in_quotes = false;
    for &b in record {
        match b {
            b'"' => {
                in_quotes = !in_quotes;
                field.push(b);
            }
            b';' if !in_quotes => {
                fields.push(std::mem::take(&mut field));
            }
            _ => field.push(b),
        }
    }
    fields.push(field);
    fields
}

fn parse_records(data: &[u8]) -> Vec<Vec<Vec<u8>>> {
    RecordIter::new(data, Dialect::TSE)
        .map(|rec| {
            let mut buf: [&[u8]; MAX_COLS] = [&[]; MAX_COLS];
            let n = rec.split(Dialect::TSE, &mut buf);
            buf[..n].iter().map(|f| f.to_vec()).collect()
        })
        .collect()
}

fn rng_next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn gen_input(state: &mut u64, len: usize) -> Vec<u8> {
    let alphabet: &[u8] = b"ab;\"\n\r12";
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let r = rng_next(state) as usize;
        match r % 10 {
            // Bias toward runs of plain bytes and occasional quoted blobs.
            0..=5 => {
                let c = alphabet[r / 10 % 2];
                for _ in 0..1 + (r % 5) {
                    out.push(c);
                }
            }
            6 => out.push(b'"'),
            7 => out.push(b';'),
            8 => out.push(b'\n'),
            _ => out.push(b'\r'),
        }
    }
    out
}

#[test]
fn matches_naive_reference() {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for case in 0..200 {
        let len = 1 + (rng_next(&mut state) as usize % 300);
        let data = gen_input(&mut state, len);
        let expect = reference_records(&data);
        let got = parse_records(&data);
        assert_eq!(
            got,
            expect,
            "case {case} diverged on input {:?}",
            String::from_utf8_lossy(&data)
        );
    }
}

#[test]
fn long_input_crosses_chunk_boundaries() {
    let mut state = 42u64;
    for case in 0..20 {
        let len = 100 + (rng_next(&mut state) as usize % 5000);
        let data = gen_input(&mut state, len);
        assert_eq!(
            parse_records(&data),
            reference_records(&data),
            "case {case}"
        );
    }
}

#[test]
fn scanner_state_rewinds_consistently() {
    // Byte-by-byte scanning (chunk=1) must agree with whole-chunk scanning.
    let mut state = 7u64;
    for _ in 0..100 {
        let data = gen_input(&mut state, 200);
        let mut whole = BlockScanner::new(Dialect::TSE);
        for chunk in data.chunks(64) {
            whole.scan(chunk);
        }
        let mut one = BlockScanner::new(Dialect::TSE);
        for &b in data.iter() {
            one.scan(&[b]);
        }
        assert_eq!(whole.in_quotes(), one.in_quotes());
    }
}
