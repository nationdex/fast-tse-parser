# fast-tse-parser

SIMD / zero-copy CSV parser for Brazilian TSE open electoral data
(`consulta_cand`, `votacao_secao`, `bem_candidato`).

Semicolon-delimited Latin-1/UTF-8 rows, hundreds of thousands to millions per
file. Built for MindVote-style election-cycle discovery and candidate asset
sync where the parse budget is milliseconds.

## What it does

- **Memory-mapped streaming** — parse straight from the OS page cache, no
  copying into heap buffers.
- **SIMD kernels** — AVX2 (x86-64), SSE2 (x86), NEON (aarch64), scalar
  fallback. Runtime detection, one 64-bit mask per interesting byte.
- **Zero allocations on the parse path** — every record and field borrows from
  the map. Decoding to owned text is opt-in.
- **Quote-aware** — embedded `;`, newlines inside quoted fields, `""` escapes,
  CRLF/LF, UTF-8 BOM, Latin-1 or UTF-8 sources.
- **Layout-agnostic columns** — typed accessors resolve by header name, so
  minor schema drift between election years does not break parsing.

## Usage

```rust
use fast_tse_parser::{ConsultaCand, Mmap};

let map = Mmap::open("consulta_cand_2022_BR.csv")?;
let doc = ConsultaCand::parse(map.as_bytes())?;

// Election cycle discovery: distinct ANO_ELEICAO, zero alloc.
doc.election_cycles(|ano| println!("cycle {}", ano.to_u32().unwrap_or_default()));

for cand in doc.iter() {
    if cand.sq_candidato() == Some(1001) {
        println!("{:?} {:?}", cand.nm_candidato().and_then(|f| f.as_str()), cand.sg_uf());
    }
}
```

Raw layer (any TSE CSV, no typed schema):

```rust
use fast_tse_parser::{Dialect, RecordIter};

for rec in RecordIter::new(bytes, Dialect::TSE) {
    for field in rec.fields(Dialect::TSE) {
        let raw = field.as_bytes(); // zero-copy
    }
}
```

`VotacaoSecao` / `BemCandidato` wrap the same table engine
(`total_votes()`, `total_cents()`, per-row accessors).

## Measured (200 MiB synthetic `consulta_cand`, 1.87M rows, release, x86-64)

| path                          | throughput      | rate        |
|-------------------------------|-----------------|-------------|
| raw record iteration          | ~3.7 GiB/s      | ~35M rows/s |
| typed rows (2 column lookups) | ~0.5 GiB/s      | ~4.7M rows/s|
| election cycle discovery      | ~140 ms         | full file   |

```
cargo run --release --example bench -- consulta_cand_2022_BR.csv
```

## Design notes

- One pass per level: `RecordIter` walks the map in 64-byte SIMD blocks and
  emits line slices; `FieldIter` splits each record with 16-byte windows.
  Quote state is a prefix-XOR bitmask over `"`, so delimiters inside quoted
  fields are ignored without byte-wise parsing.
- Column lookups use FNV-1a hashes of header names in a fixed 64-slot array
  (no `HashMap`, no allocation). Typed accessors hold `const` hashes.
- `Row` carries a cursor cache: sequential column reads scan the record once.
- Numbers parse from stack buffers (`u32`/`u64`, `1.234,56` money to cents).
- Safety: the only `unsafe` is inside the SIMD kernels and the `mmap` call.

## Tests

- Unit tests per module (mask kernels, quote machine, number parsing, encodings).
- Property tests: random adversarial CSV vs naive byte-walk reference.
- Format tests for all three TSE tables, Latin-1 decoding, escapes, CRLF.

```
cargo test
cargo clippy --all-targets
```
