//! Zero-allocation header -> column index map (FNV-1a over fixed array).

use crate::scan::{Dialect, FieldIter, Record, MAX_COLS};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[inline]
pub const fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    let mut i = 0;
    while i < bytes.len() {
        h ^= bytes[i] as u64;
        h = h.wrapping_mul(FNV_PRIME);
        i += 1;
    }
    h
}

/// Column names hashed at parse time. Lookup is a linear scan over at most
/// `MAX_COLS` hashes — no heap, no `HashMap`.
#[derive(Clone, Debug)]
pub struct Header {
    hashes: [u64; MAX_COLS],
    len: u16,
}

impl Header {
    pub fn parse(record: Record<'_>, dialect: Dialect) -> Header {
        let mut h = Header {
            hashes: [0; MAX_COLS],
            len: 0,
        };
        for field in FieldIter::new(record.as_bytes(), dialect) {
            if h.len as usize == MAX_COLS {
                break;
            }
            h.hashes[h.len as usize] = fnv1a(field.as_bytes());
            h.len += 1;
        }
        h
    }

    pub fn len(&self) -> u16 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn index_of(&self, name: &str) -> Option<u16> {
        self.index_of_hash(fnv1a(name.as_bytes()))
    }

    pub fn index_of_hash(&self, target: u64) -> Option<u16> {
        (0..self.len).find(|&i| self.hashes[i as usize] == target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scan::Record;

    #[test]
    fn resolves_columns() {
        let rec = Record::from_raw(b"ANO_ELEICAO;SG_UF;QT_VOTOS");
        let h = Header::parse(rec, Dialect::TSE);
        assert_eq!(h.len(), 3);
        assert_eq!(h.index_of("SG_UF"), Some(1));
        assert_eq!(h.index_of("QT_VOTOS"), Some(2));
        assert_eq!(h.index_of("NOPE"), None);
    }
}
