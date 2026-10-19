#[cfg(test)]
const MAX_COMPARE_SIZE: usize = 256;

pub fn compare256_slice(src0: &[u8], src1: &[u8]) -> usize {
    let src0 = first_chunk::<_, 256>(src0).unwrap();
    let src1 = first_chunk::<_, 256>(src1).unwrap();

    compare256(src0, src1)
}

fn compare256(src0: &[u8; 256], src1: &[u8; 256]) -> usize {
    #[cfg(feature = "avx512")]
    #[cfg(target_arch = "x86_64")]
    if cfg!(target_feature = "avx512vl") && cfg!(target_feature = "avx512bw") {
        return unsafe { avx512::compare256(src0, src1) };
    }

    #[cfg(target_arch = "x86_64")]
    if crate::cpu_features::is_enabled_avx2_and_bmi2() {
        return unsafe { avx2::compare256(src0, src1) };
    }

    #[cfg(target_arch = "aarch64")]
    if crate::cpu_features::is_enabled_neon() {
        return unsafe { neon::compare256(src0, src1) };
    }

    #[cfg(all(target_arch = "loongarch64", feature = "lsx"))]
    if crate::cpu_features::is_enabled_lsx() {
        return unsafe { lsx::compare256(src0, src1) };
    }

    #[cfg(target_arch = "wasm32")]
    if crate::cpu_features::is_enabled_simd128() {
        return wasm32::compare256(src0, src1);
    }

    rust::compare256(src0, src1)
}

pub fn compare256_rle_slice(byte: u8, src: &[u8]) -> usize {
    rust::compare256_rle(byte, src)
}

#[inline]
pub const fn first_chunk<T, const N: usize>(slice: &[T]) -> Option<&[T; N]> {
    if slice.len() < N {
        None
    } else {
        // SAFETY: We explicitly check for the correct number of elements,
        //   and do not let the reference outlive the slice.
        Some(unsafe { &*(slice.as_ptr() as *const [T; N]) })
    }
}

mod rust {

    pub fn compare256(src0: &[u8; 256], src1: &[u8; 256]) -> usize {
        // only unrolls 4 iterations; zlib-ng unrolls 8
        src0.iter().zip(src1).take_while(|(x, y)| x == y).count()
    }

    // run-length encoding
    pub fn compare256_rle(byte: u8, src: &[u8]) -> usize {
        assert!(src.len() >= 256, "too short {}", src.len());

        let sv = u64::from_ne_bytes([byte; 8]);
        let mut len = 0;

        // this optimizes well because we statically limit the slice to 256 bytes.
        // the loop gets unrolled 4 times automatically.
        for chunk in src[..256].chunks_exact(8) {
            let mv = u64::from_le_bytes(chunk.try_into().unwrap());

            let diff = sv ^ mv;

            if diff > 0 {
                let match_byte = diff.trailing_zeros() / 8;
                return len + match_byte as usize;
            }

            len += 8
        }

        256
    }

    #[test]
    fn test_compare256() {
        let str1 = [b'a'; super::MAX_COMPARE_SIZE];
        let mut str2 = [b'a'; super::MAX_COMPARE_SIZE];

        for i in 0..str1.len() {
            str2[i] = 0;

            let match_len = compare256(&str1, &str2);
            assert_eq!(match_len, i);

            str2[i] = b'a';
        }
    }

    #[test]
    fn test_compare256_rle() {
        let mut string = [b'a'; super::MAX_COMPARE_SIZE];

        for i in 0..string.len() {
            string[i] = 0;

            let match_len = compare256_rle(b'a', &string);
            assert_eq!(match_len, i);

            string[i] = b'a';
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::{veorq_u8, vgetq_lane_u64, vld1q_u8, vreinterpretq_u64_u8};

    /// # Safety
    ///
    /// Behavior is undefined if the `neon` target feature is not enabled
    #[target_feature(enable = "neon")]
    pub unsafe fn compare256(src0: &[u8; 256], src1: &[u8; 256]) -> usize {
        // Compare 16 bytes at a time, and check each 8-byte half for a mismatch before loading
        // more. Most calls return within the first 16 bytes, so exiting early beats building a
        // 64-byte bitmask first.
        let src0 = src0.chunks_exact(16);
        let src1 = src1.chunks_exact(16);

        let mut len = 0;

        for (a, b) in src0.zip(src1) {
            unsafe {
                let a = vld1q_u8(a.as_ptr());
                let b = vld1q_u8(b.as_ptr());

                // Nonzero bytes mark a mismatch.
                let cmp = vreinterpretq_u64_u8(veorq_u8(a, b));

                // We reinterpreted the vector as 64-bit integers, so endianness matters: on
                // big-endian, lane 1 holds the first 8 bytes, most significant byte first.
                let (first, second) = if cfg!(target_endian = "big") {
                    (vgetq_lane_u64::<1>(cmp), vgetq_lane_u64::<0>(cmp))
                } else {
                    (vgetq_lane_u64::<0>(cmp), vgetq_lane_u64::<1>(cmp))
                };

                let first = first.to_le();
                if first != 0 {
                    return len + (first.trailing_zeros() / 8) as usize;
                }

                let second = second.to_le();
                if second != 0 {
                    return len + 8 + (second.trailing_zeros() / 8) as usize;
                }
            }

            len += 16;
        }

        256
    }

    #[test]
    fn test_compare256() {
        if crate::cpu_features::is_enabled_neon() {
            let str1 = [b'a'; super::MAX_COMPARE_SIZE];
            let mut str2 = [b'a'; super::MAX_COMPARE_SIZE];

            for i in 0..str1.len() {
                str2[i] = 0;

                let match_len = unsafe { compare256(&str1, &str2) };
                assert_eq!(match_len, i);

                str2[i] = b'a';
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use core::arch::x86_64::{
        __m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8,
    };

    /// # Safety
    ///
    /// Behavior is undefined if the `avx` target feature is not enabled
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "bmi2")]
    #[target_feature(enable = "bmi1")]
    pub unsafe fn compare256(src0: &[u8; 256], src1: &[u8; 256]) -> usize {
        let src0 = src0.chunks_exact(32);
        let src1 = src1.chunks_exact(32);

        let mut len = 0;

        unsafe {
            for (chunk0, chunk1) in src0.zip(src1) {
                let ymm_src0 = _mm256_loadu_si256(chunk0.as_ptr() as *const __m256i);
                let ymm_src1 = _mm256_loadu_si256(chunk1.as_ptr() as *const __m256i);

                // element-wise compare of the 8-bit elements
                let ymm_cmp = _mm256_cmpeq_epi8(ymm_src0, ymm_src1);

                // turn an 32 * 8-bit vector into a 32-bit integer.
                // a bit in the output is one if the corresponding element is non-zero.
                let mask = _mm256_movemask_epi8(ymm_cmp) as u32;

                if mask != 0xFFFFFFFF {
                    let match_byte = mask.trailing_ones();
                    return len + match_byte as usize;
                }

                len += 32;
            }
        }

        256
    }

    #[test]
    fn test_compare256() {
        if crate::cpu_features::is_enabled_avx2_and_bmi2() {
            let str1 = [b'a'; super::MAX_COMPARE_SIZE];
            let mut str2 = [b'a'; super::MAX_COMPARE_SIZE];

            for i in 0..str1.len() {
                str2[i] = 0;

                let match_len = unsafe { compare256(&str1, &str2) };
                assert_eq!(match_len, i);

                str2[i] = b'a';
            }
        }
    }
}

#[cfg(feature = "avx512")]
#[cfg(target_arch = "x86_64")]
mod avx512 {
    use core::arch::x86_64::{
        _mm512_cmpeq_epu8_mask, _mm512_loadu_si512, _mm_cmpeq_epu8_mask, _mm_loadu_si128,
    };

    /// # Safety
    ///
    /// Behavior is undefined if the `avx` target feature is not enabled
    #[target_feature(enable = "avx512vl")]
    #[target_feature(enable = "avx512bw")]
    pub unsafe fn compare256(src0: &[u8; 256], src1: &[u8; 256]) -> usize {
        // First do a 16byte round before increasing to 64bytes, this reduces the
        // penalty for the short matches, and those are usually the most common ones.
        // This requires us to overlap on the last round, giving a small penalty
        // on matches of 192+ bytes (Still faster than AVX2 though).

        unsafe {
            // 16 bytes
            let xmm_src0_0 = _mm_loadu_si128(src0.as_ptr().cast());
            let xmm_src1_0 = _mm_loadu_si128(src1.as_ptr().cast());
            let mask_0 = u32::from(_mm_cmpeq_epu8_mask(xmm_src0_0, xmm_src1_0)); // zero-extended to use __builtin_ctz
            if mask_0 != 0x0000FFFF {
                // There is potential for using __builtin_ctzg/__builtin_ctzs/_tzcnt_u16/__tzcnt_u16 here
                let match_byte = mask_0.trailing_ones();
                return match_byte as usize;
            }

            // 64 bytes
            let zmm_src0_1 = _mm512_loadu_si512(src0[16..].as_ptr().cast());
            let zmm_src1_1 = _mm512_loadu_si512(src1[16..].as_ptr().cast());
            let mask_1 = _mm512_cmpeq_epu8_mask(zmm_src0_1, zmm_src1_1);
            if mask_1 != 0xFFFFFFFFFFFFFFFF {
                let match_byte = mask_1.trailing_ones();
                return 16 + match_byte as usize;
            }

            // 64 bytes
            let zmm_src0_2 = _mm512_loadu_si512(src0[80..].as_ptr().cast());
            let zmm_src1_2 = _mm512_loadu_si512(src1[80..].as_ptr().cast());
            let mask_2 = _mm512_cmpeq_epu8_mask(zmm_src0_2, zmm_src1_2);
            if mask_2 != 0xFFFFFFFFFFFFFFFF {
                let match_byte = mask_2.trailing_ones();
                return 80 + match_byte as usize;
            }

            // 64 bytes
            let zmm_src0_3 = _mm512_loadu_si512(src0[144..].as_ptr().cast());
            let zmm_src1_3 = _mm512_loadu_si512(src1[144..].as_ptr().cast());
            let mask_3 = _mm512_cmpeq_epu8_mask(zmm_src0_3, zmm_src1_3);
            if mask_3 != 0xFFFFFFFFFFFFFFFF {
                let match_byte = mask_3.trailing_ones();
                return 144 + match_byte as usize;
            }

            // 64 bytes (overlaps the previous 16 bytes for fast tail processing)
            let zmm_src0_4 = _mm512_loadu_si512(src0[192..].as_ptr().cast());
            let zmm_src1_4 = _mm512_loadu_si512(src1[192..].as_ptr().cast());
            let mask_4 = _mm512_cmpeq_epu8_mask(zmm_src0_4, zmm_src1_4);
            if mask_4 != 0xFFFFFFFFFFFFFFFF {
                let match_byte = mask_4.trailing_ones();
                return 192 + match_byte as usize;
            }
        }

        256
    }

    #[test]
    fn test_compare256() {
        if cfg!(target_feature = "avx512vl") && cfg!(target_feature = "avx512bw") {
            let str1 = [b'a'; super::MAX_COMPARE_SIZE];
            let mut str2 = [b'a'; super::MAX_COMPARE_SIZE];

            for i in 0..str1.len() {
                str2[i] = 0;

                let match_len = unsafe { compare256(&str1, &str2) };
                assert_eq!(match_len, i);

                str2[i] = b'a';
            }
        }
    }
}

#[cfg(all(target_arch = "loongarch64", feature = "lsx"))]
mod lsx {
    use core::arch::loongarch64::{lsx_vld, lsx_vmskltz_b, lsx_vpickve2gr_w, lsx_vseq_b};

    #[target_feature(enable = "lsx")]
    pub fn compare256(src0: &[u8; 256], src1: &[u8; 256]) -> usize {
        let src0 = src0.chunks_exact(16);
        let src1 = src1.chunks_exact(16);

        let mut len = 0;

        unsafe {
            for (chunk0, chunk1) in src0.zip(src1) {
                let a = lsx_vld::<0>(chunk0.as_ptr() as *const i8);
                let b = lsx_vld::<0>(chunk1.as_ptr() as *const i8);

                // element-wise compare of the 8-bit elements
                // 0xff if equal, otherwise 0
                let v = lsx_vseq_b(a, b);

                // bootleg movemask implementation
                let mask = lsx_vpickve2gr_w::<0>(lsx_vmskltz_b(v));

                if mask != 0xFFFF {
                    let match_byte = mask.trailing_ones();
                    return len + match_byte as usize;
                }

                len += 16;
            }
        }

        256
    }

    #[test]
    fn test_compare256() {
        if crate::cpu_features::is_enabled_lsx() {
            let str1 = [b'a'; super::MAX_COMPARE_SIZE];
            let mut str2 = [b'a'; super::MAX_COMPARE_SIZE];

            for i in 0..str1.len() {
                str2[i] = 0;

                let match_len = unsafe { compare256(&str1, &str2) };
                assert_eq!(match_len, i);

                str2[i] = b'a';
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm32 {
    use core::arch::wasm32::{u8x16_bitmask, u8x16_eq, v128, v128_load};

    #[target_feature(enable = "simd128")]
    pub fn compare256(src0: &[u8; 256], src1: &[u8; 256]) -> usize {
        let src0 = src0.chunks_exact(16);
        let src1 = src1.chunks_exact(16);

        let mut len = 0;

        for (chunk0, chunk1) in src0.zip(src1) {
            // SAFETY: these are valid pointers to slice data.
            let v128_src0 = unsafe { v128_load(chunk0.as_ptr() as *const v128) };
            let v128_src1 = unsafe { v128_load(chunk1.as_ptr() as *const v128) };

            let v128_cmp = u8x16_eq(v128_src0, v128_src1);
            let mask = u8x16_bitmask(v128_cmp);

            if mask != 0xFFFF {
                let match_byte = mask.trailing_ones();
                return len + match_byte as usize;
            }

            len += 16;
        }

        256
    }

    #[test]
    fn test_compare256() {
        if crate::cpu_features::is_enabled_simd128() {
            let str1 = [b'a'; super::MAX_COMPARE_SIZE];
            let mut str2 = [b'a'; super::MAX_COMPARE_SIZE];

            for i in 0..str1.len() {
                str2[i] = 0;

                let match_len = unsafe { compare256(&str1, &str2) };
                assert_eq!(match_len, i);

                str2[i] = b'a';
            }
        }
    }
}
