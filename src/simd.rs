//! SIMD kernels producing 64-bit interesting-byte masks per 64-byte block.
//!
//! One block scan reports positions of three sentinel bytes (quote, field
//! separator, record separator) as `u64` bitmasks. Higher layers combine these
//! with a prefix-XOR quote-state machine to ignore delimiters inside quoted
//! fields.

pub const BLOCK: usize = 64;

/// Prefix XOR: bit `i` of the result is the XOR of mask bits `0..=i`.
#[inline(always)]
pub fn prefix_xor(mut m: u64) -> u64 {
    m ^= m << 1;
    m ^= m << 2;
    m ^= m << 4;
    m ^= m << 8;
    m ^= m << 16;
    m ^= m << 32;
    m
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockMasks {
    pub quote: u64,
    pub sep: u64,
    pub nl: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Avx2,
    Sse2,
    Neon,
    Scalar,
}

impl BlockMasks {
    #[inline(always)]
    fn merge_at(&mut self, m: BlockMasks, off: usize) {
        self.quote |= m.quote << off;
        self.sep |= m.sep << off;
        self.nl |= m.nl << off;
    }
}

impl Backend {
    pub fn detect() -> Backend {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                return Backend::Avx2;
            }
            return Backend::Sse2;
        }
        #[cfg(target_arch = "aarch64")]
        {
            return Backend::Neon;
        }
        #[allow(unreachable_code)]
        Backend::Scalar
    }

    /// Scan `data` (len <= `BLOCK`). Masks are relative to `data[0]` and zero
    /// beyond `data.len()`. Full 64-byte chunks take the wide kernel, 16-byte
    /// chunks a narrow SIMD kernel, only sub-16 tails go scalar.
    #[inline]
    pub fn scan(self, data: &[u8], quote: u8, sep: u8, nl: u8) -> BlockMasks {
        debug_assert!(data.len() <= BLOCK);
        let mut out = BlockMasks::default();
        let mut off = 0usize;
        while data.len() - off >= BLOCK {
            let m = unsafe { self.scan64(data.as_ptr().add(off), quote, sep, nl) };
            out.merge_at(m, off);
            off += BLOCK;
        }
        while data.len() - off >= 16 {
            let m = unsafe { self.scan16(data.as_ptr().add(off), quote, sep, nl) };
            out.merge_at(m, off);
            off += 16;
        }
        if off < data.len() {
            out.merge_at(scalar::scan(&data[off..], quote, sep, nl), off);
        }
        out
    }

    #[inline]
    unsafe fn scan64(self, ptr: *const u8, quote: u8, sep: u8, nl: u8) -> BlockMasks {
        unsafe {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if self == Backend::Avx2 {
                return x86::scan64_avx2(ptr, quote, sep, nl);
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            return x86::scan64_sse2(ptr, quote, sep, nl);
            #[cfg(target_arch = "aarch64")]
            return aarch64::scan64_neon(ptr, quote, sep, nl);
            #[allow(unreachable_code)]
            scalar::scan(core::slice::from_raw_parts(ptr, BLOCK), quote, sep, nl)
        }
    }

    #[inline]
    unsafe fn scan16(self, ptr: *const u8, quote: u8, sep: u8, nl: u8) -> BlockMasks {
        unsafe {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            return x86::scan16_sse2(ptr, quote, sep, nl);
            #[cfg(target_arch = "aarch64")]
            return aarch64::scan16_neon(ptr, quote, sep, nl);
            #[allow(unreachable_code)]
            scalar::scan(core::slice::from_raw_parts(ptr, 16), quote, sep, nl)
        }
    }
}

mod scalar {
    use super::BlockMasks;

    #[inline]
    pub(super) fn scan(data: &[u8], quote: u8, sep: u8, nl: u8) -> BlockMasks {
        let mut m = BlockMasks::default();
        for (i, &b) in data.iter().enumerate() {
            let bit = 1u64 << i;
            if b == quote {
                m.quote |= bit;
            } else if b == sep {
                m.sep |= bit;
            } else if b == nl {
                m.nl |= bit;
            }
        }
        m
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::BlockMasks;
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    #[inline(always)]
    fn merge(lo: u64, hi: u64) -> u64 {
        lo | (hi << 32)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn mask32_avx2(v: __m256i, k: __m256i) -> u64 {
        let cmp = _mm256_cmpeq_epi8(v, k);
        (_mm256_movemask_epi8(cmp) as u32) as u64
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn scan64_avx2(ptr: *const u8, quote: u8, sep: u8, nl: u8) -> BlockMasks {
        unsafe {
            let v0 = _mm256_loadu_si256(ptr as *const __m256i);
            let v1 = _mm256_loadu_si256(ptr.add(32) as *const __m256i);

            let q = _mm256_set1_epi8(quote as i8);
            let s = _mm256_set1_epi8(sep as i8);
            let n = _mm256_set1_epi8(nl as i8);

            BlockMasks {
                quote: merge(mask32_avx2(v0, q), mask32_avx2(v1, q)),
                sep: merge(mask32_avx2(v0, s), mask32_avx2(v1, s)),
                nl: merge(mask32_avx2(v0, n), mask32_avx2(v1, n)),
            }
        }
    }

    #[target_feature(enable = "sse2")]
    unsafe fn mask16_sse2(v: __m128i, k: __m128i) -> u64 {
        let cmp = _mm_cmpeq_epi8(v, k);
        (_mm_movemask_epi8(cmp) as u32) as u64
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn scan16_sse2(ptr: *const u8, quote: u8, sep: u8, nl: u8) -> BlockMasks {
        unsafe {
            let v = _mm_loadu_si128(ptr as *const __m128i);
            BlockMasks {
                quote: mask16_sse2(v, _mm_set1_epi8(quote as i8)),
                sep: mask16_sse2(v, _mm_set1_epi8(sep as i8)),
                nl: mask16_sse2(v, _mm_set1_epi8(nl as i8)),
            }
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn scan64_sse2(ptr: *const u8, quote: u8, sep: u8, nl: u8) -> BlockMasks {
        unsafe {
            let v0 = _mm_loadu_si128(ptr as *const __m128i);
            let v1 = _mm_loadu_si128(ptr.add(16) as *const __m128i);
            let v2 = _mm_loadu_si128(ptr.add(32) as *const __m128i);
            let v3 = _mm_loadu_si128(ptr.add(48) as *const __m128i);

            let q = _mm_set1_epi8(quote as i8);
            let s = _mm_set1_epi8(sep as i8);
            let n = _mm_set1_epi8(nl as i8);

            let quote = mask16_sse2(v0, q)
                | (mask16_sse2(v1, q) << 16)
                | (mask16_sse2(v2, q) << 32)
                | (mask16_sse2(v3, q) << 48);
            let sep = mask16_sse2(v0, s)
                | (mask16_sse2(v1, s) << 16)
                | (mask16_sse2(v2, s) << 32)
                | (mask16_sse2(v3, s) << 48);
            let nl = mask16_sse2(v0, n)
                | (mask16_sse2(v1, n) << 16)
                | (mask16_sse2(v2, n) << 32)
                | (mask16_sse2(v3, n) << 48);

            BlockMasks { quote, sep, nl }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::BlockMasks;
    use core::arch::aarch64::*;

    const BITS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn scan64_neon(ptr: *const u8, quote: u8, sep: u8, nl: u8) -> BlockMasks {
        unsafe {
            macro_rules! mask16 {
                ($v:expr, $k:expr, $bits:expr) => {{
                    let cm = vceqq_u8($v, $k);
                    let masked = vandq_u8(cm, $bits);
                    let lo = vaddv_u8(vget_low_u8(masked)) as u64;
                    let hi = vaddv_u8(vget_high_u8(masked)) as u64;
                    lo | (hi << 8)
                }};
            }

            let bits = vld1q_u8(BITS.as_ptr());
            let v0 = vld1q_u8(ptr);
            let v1 = vld1q_u8(ptr.add(16));
            let v2 = vld1q_u8(ptr.add(32));
            let v3 = vld1q_u8(ptr.add(48));

            let q = vdupq_n_u8(quote);
            let s = vdupq_n_u8(sep);
            let n = vdupq_n_u8(nl);

            let quote = mask16!(v0, q, bits)
                | (mask16!(v1, q, bits) << 16)
                | (mask16!(v2, q, bits) << 32)
                | (mask16!(v3, q, bits) << 48);
            let sep = mask16!(v0, s, bits)
                | (mask16!(v1, s, bits) << 16)
                | (mask16!(v2, s, bits) << 32)
                | (mask16!(v3, s, bits) << 48);
            let nl = mask16!(v0, n, bits)
                | (mask16!(v1, n, bits) << 16)
                | (mask16!(v2, n, bits) << 32)
                | (mask16!(v3, n, bits) << 48);

            BlockMasks { quote, sep, nl }
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn scan16_neon(ptr: *const u8, quote: u8, sep: u8, nl: u8) -> BlockMasks {
        unsafe {
            macro_rules! mask16 {
                ($v:expr, $k:expr, $bits:expr) => {{
                    let cm = vceqq_u8($v, $k);
                    let masked = vandq_u8(cm, $bits);
                    let lo = vaddv_u8(vget_low_u8(masked)) as u64;
                    let hi = vaddv_u8(vget_high_u8(masked)) as u64;
                    lo | (hi << 8)
                }};
            }
            let bits = vld1q_u8(BITS.as_ptr());
            let v = vld1q_u8(ptr);
            BlockMasks {
                quote: mask16!(v, vdupq_n_u8(quote), bits),
                sep: mask16!(v, vdupq_n_u8(sep), bits),
                nl: mask16!(v, vdupq_n_u8(nl), bits),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan_all(data: &[u8], quote: u8, sep: u8, nl: u8) -> BlockMasks {
        Backend::Scalar.scan(data, quote, sep, nl)
    }

    #[test]
    fn prefix_xor_folds_bits() {
        // bits 0 and 3 set -> parity 1 from bit 0..2? p[0]=1, p[1]=1, p[2]=1, p[3]=0
        let p = prefix_xor(0b1001);
        assert_eq!(p & 0b1111, 0b0111);
    }

    #[test]
    fn scalar_masks_match_reference() {
        let mut data = [0u8; 64];
        data[0] = b'"';
        data[31] = b';';
        data[32] = b'"';
        data[63] = b'\n';
        let m = scan_all(&data, b'"', b';', b'\n');
        assert_eq!(m.quote, (1 << 0) | (1 << 32));
        assert_eq!(m.sep, 1 << 31);
        assert_eq!(m.nl, 1 << 63);
    }

    #[test]
    fn backend_matches_scalar_on_full_block() {
        let data: Vec<u8> = (0..64u8)
            .map(|i| match i % 7 {
                0 => b'"',
                1 => b';',
                2 => b'\n',
                _ => b'a' + (i % 3),
            })
            .collect();
        let expect = scan_all(&data, b'"', b';', b'\n');
        for backend in [Backend::Avx2, Backend::Sse2, Backend::Neon, Backend::Scalar] {
            let got = backend.scan(&data, b'"', b';', b'\n');
            if got != expect {
                panic!("mismatch for {backend:?}: {got:?} != {expect:?}");
            }
        }
    }

    #[test]
    fn backends_match_scalar_on_every_length() {
        let mut state = 12345u64;
        for len in 0..=BLOCK {
            let data: Vec<u8> = (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    b"\";\nx"[(state % 4) as usize]
                })
                .collect();
            let expect = scan_all(&data, b'"', b';', b'\n');
            for backend in [Backend::Avx2, Backend::Sse2, Backend::Neon, Backend::Scalar] {
                assert_eq!(
                    backend.scan(&data, b'"', b';', b'\n'),
                    expect,
                    "len={len} backend={backend:?}"
                );
            }
        }
    }

    #[test]
    fn detect_is_scalar_or_simd() {
        let _ = Backend::detect();
    }
}
