//! # fast-tse-parser
//!
//! SIMD / zero-copy CSV parser for Brazilian TSE open electoral data
//! (`consulta_cand`, `votacao_secao`, `bem_candidato`).
//!
//! - Memory-mapped streaming over semicolon-delimited Latin-1/UTF-8 rows
//! - AVX2 / SSE2 / NEON kernels with scalar fallback
//! - Zero allocation on the parse path: every field borrows from the map
//! - Quote-aware splitting (embedded `;`, newlines, `""` escapes)
//!
//! ```no_run
//! use fast_tse_parser::{ConsultaCand, Mmap};
//!
//! let map = Mmap::open("consulta_cand_2022_BR.csv")?;
//! let doc = ConsultaCand::parse(map.as_bytes())?;
//! doc.election_cycles(|ano| {
//!     let year = ano.to_u32().unwrap_or_default();
//!     println!("cycle {year}");
//! });
//! for cand in doc.iter().take(3) {
//!     println!("{:?} {:?}", cand.nm_candidato().and_then(|f| f.as_str()), cand.sg_uf());
//! }
//! # Ok::<(), fast_tse_parser::Error>(())
//! ```

#![deny(unsafe_op_in_unsafe_fn)]

pub mod formats;
pub mod header;
pub mod latin1;
pub mod mmap;
pub mod scan;
pub mod simd;

pub use formats::{
    Bem, BemCandidato, Candidato, ConsultaCand, Row, TseTable, VotacaoSecao, VotoSecao,
};
pub use header::Header;
pub use latin1::Encoding;
pub use mmap::Mmap;
pub use scan::{BlockScanner, Dialect, Field, FieldIter, Record, RecordIter, ScanBlock, MAX_COLS};

use std::fmt;

#[derive(Debug)]
pub enum Error {
    EmptyInput,
    Io(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::EmptyInput => write!(f, "empty TSE input"),
            Error::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
