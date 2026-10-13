//! Memory-mapped file access (read-only, page-cache friendly streaming).

use crate::Error;
use std::fs::File;
use std::path::Path;

pub struct Mmap {
    inner: memmap2::Mmap,
}

impl Mmap {
    pub fn open(path: impl AsRef<Path>) -> Result<Mmap, Error> {
        let file = File::open(path)?;
        // SAFETY: read-only mapping of an immutable-at-parse-time file.
        let inner = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Mmap { inner })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.inner
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}
