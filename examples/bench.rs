//! Throughput bench for TSE CSV parsing.
//!
//! Usage: `cargo run --release --example bench -- <tse-file.csv> [iterations]`

use fast_tse_parser::{ConsultaCand, Dialect, Mmap, RecordIter};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .unwrap_or_else(|| "consulta_cand_2022_BR.csv".to_string());
    let iters: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);

    let map = match Mmap::open(&path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("open {path}: {e}");
            std::process::exit(1);
        }
    };
    let bytes = map.as_bytes();
    let mib = bytes.len() as f64 / (1024.0 * 1024.0);
    println!("file {path}: {mib:.2} MiB");

    let mut best = f64::MAX;
    let mut rows = 0usize;
    for _ in 0..iters {
        let t = Instant::now();
        rows = RecordIter::new(bytes, Dialect::TSE).count();
        let dt = t.elapsed().as_secs_f64();
        best = best.min(dt);
    }
    println!(
        "records (raw): {rows} in {best:.4}s -> {:.2} MiB/s, {:.0} rows/s",
        mib / best,
        rows as f64 / best
    );

    let mut best = f64::MAX;
    let mut n = 0usize;
    for _ in 0..iters {
        let t = Instant::now();
        if let Ok(doc) = ConsultaCand::parse(bytes) {
            n = doc.iter().count();
        }
        let dt = t.elapsed().as_secs_f64();
        best = best.min(dt);
    }
    println!(
        "typed rows (no accessors): {n} in {best:.4}s -> {:.2} MiB/s, {:.0} rows/s",
        mib / best,
        n as f64 / best
    );

    let mut best = f64::MAX;
    let mut named = 0usize;
    for _ in 0..iters {
        let t = Instant::now();
        if let Ok(doc) = ConsultaCand::parse(bytes) {
            named = 0;
            for cand in doc.iter() {
                if cand.sq_candidato().is_some() && cand.nm_candidato().is_some() {
                    named += 1;
                }
            }
        }
        let dt = t.elapsed().as_secs_f64();
        best = best.min(dt);
    }
    println!(
        "typed consulta_cand rows: {named} in {best:.4}s -> {:.2} MiB/s, {:.0} rows/s",
        mib / best,
        named as f64 / best
    );

    if let Ok(doc) = ConsultaCand::parse(bytes) {
        let t = Instant::now();
        let mut cycles = Vec::new();
        doc.election_cycles(|f| {
            if let Some(y) = f.to_u32() {
                cycles.push(y);
            }
        });
        println!(
            "cycle discovery: {:?} in {:.4}ms",
            cycles,
            t.elapsed().as_secs_f64() * 1e3
        );
    }
}
