//! SIMD-accelerated kernels for CPU tensor operations.
//!
//! This module provides optimized implementations using platform-specific SIMD
//! instructions (AVX2 on x86_64, NEON on ARM).

#![allow(dead_code, unreachable_code)]

/// Check if AVX2 is available (x86_64 only).
#[cfg(target_arch = "x86_64")]
pub fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx2() -> bool {
    false
}

/// Check if NEON is available (ARM only).
#[cfg(target_arch = "aarch64")]
pub fn has_neon() -> bool {
    // NEON is mandatory on aarch64
    true
}

#[cfg(not(target_arch = "aarch64"))]
pub fn has_neon() -> bool {
    false
}

// === SIMD-accelerated element-wise operations ===

/// SIMD-accelerated vector addition: out = a + b
pub fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: We've checked AVX2 is available
            unsafe { add_f32_avx2(a, b, out) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on aarch64
        unsafe { add_f32_neon(a, b, out) };
        return;
    }

    // Scalar fallback
    add_f32_scalar(a, b, out);
}

/// SIMD-accelerated vector multiplication: out = a * b
pub fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe { mul_f32_avx2(a, b, out) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { mul_f32_neon(a, b, out) };
        return;
    }

    mul_f32_scalar(a, b, out);
}

/// SIMD-accelerated fused multiply-add: out = a * b + c
pub fn fma_f32(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe { fma_f32_avx2(a, b, c, out) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { fma_f32_neon(a, b, c, out) };
        return;
    }

    fma_f32_scalar(a, b, c, out);
}

/// SIMD-accelerated sum reduction
pub fn sum_f32(a: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            return unsafe { sum_f32_avx2(a) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { sum_f32_neon(a) };
    }

    sum_f32_scalar(a)
}

// === Scalar implementations ===

fn add_f32_scalar(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

fn mul_f32_scalar(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

fn fma_f32_scalar(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] * b[i] + c[i];
    }
}

fn sum_f32_scalar(a: &[f32]) -> f32 {
    a.iter().sum()
}

// === AVX2 implementations (x86_64) ===

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_f32_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        let vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out_ptr.add(offset), vc);
    }

    // Handle remainder
    let start = chunks * 8;
    for i in 0..remainder {
        *out_ptr.add(start + i) = *a_ptr.add(start + i) + *b_ptr.add(start + i);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mul_f32_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        let vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(out_ptr.add(offset), vc);
    }

    let start = chunks * 8;
    for i in 0..remainder {
        *out_ptr.add(start + i) = *a_ptr.add(start + i) * *b_ptr.add(start + i);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn fma_f32_avx2(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        let vc = _mm256_loadu_ps(c_ptr.add(offset));
        let vr = _mm256_fmadd_ps(va, vb, vc);
        _mm256_storeu_ps(out_ptr.add(offset), vr);
    }

    let start = chunks * 8;
    for i in 0..remainder {
        *out_ptr.add(start + i) =
            *a_ptr.add(start + i) * *b_ptr.add(start + i) + *c_ptr.add(start + i);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_f32_avx2(a: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let a_ptr = a.as_ptr();
    let mut acc = _mm256_setzero_ps();

    for i in 0..chunks {
        let va = _mm256_loadu_ps(a_ptr.add(i * 8));
        acc = _mm256_add_ps(acc, va);
    }

    // Horizontal sum of 8 floats
    let low = _mm256_castps256_ps128(acc);
    let high = _mm256_extractf128_ps(acc, 1);
    let sum128 = _mm_add_ps(low, high);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);

    // Add remainder
    let start = chunks * 8;
    for i in 0..remainder {
        result += *a_ptr.add(start + i);
    }

    result
}

// === NEON implementations (aarch64) ===

#[cfg(target_arch = "aarch64")]
unsafe fn add_f32_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        let vc = vaddq_f32(va, vb);
        vst1q_f32(out_ptr.add(offset), vc);
    }

    let start = chunks * 4;
    for i in 0..remainder {
        *out_ptr.add(start + i) = *a_ptr.add(start + i) + *b_ptr.add(start + i);
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn mul_f32_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        let vc = vmulq_f32(va, vb);
        vst1q_f32(out_ptr.add(offset), vc);
    }

    let start = chunks * 4;
    for i in 0..remainder {
        *out_ptr.add(start + i) = *a_ptr.add(start + i) * *b_ptr.add(start + i);
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn fma_f32_neon(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        let vc = vld1q_f32(c_ptr.add(offset));
        let vr = vfmaq_f32(vc, va, vb);
        vst1q_f32(out_ptr.add(offset), vr);
    }

    let start = chunks * 4;
    for i in 0..remainder {
        *out_ptr.add(start + i) =
            *a_ptr.add(start + i) * *b_ptr.add(start + i) + *c_ptr.add(start + i);
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn sum_f32_neon(a: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let a_ptr = a.as_ptr();
    let mut acc = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let va = vld1q_f32(a_ptr.add(i * 4));
        acc = vaddq_f32(acc, va);
    }

    // Horizontal sum
    let mut result = vaddvq_f32(acc);

    // Add remainder
    let start = chunks * 4;
    for i in 0..remainder {
        result += *a_ptr.add(start + i);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_f32() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut out = vec![0.0; 9];

        add_f32(&a, &b, &mut out);
        assert_eq!(out, vec![10.0; 9]);
    }

    #[test]
    fn test_mul_f32() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 2.0, 2.0, 2.0];
        let mut out = vec![0.0; 4];

        mul_f32(&a, &b, &mut out);
        assert_eq!(out, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_sum_f32() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = sum_f32(&a);
        assert!((result - 55.0).abs() < 1e-6);
    }

    #[test]
    fn test_fma_f32() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 2.0, 2.0, 2.0];
        let c = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 4];

        fma_f32(&a, &b, &c, &mut out);
        // a*b + c = [3, 5, 7, 9]
        assert_eq!(out, vec![3.0, 5.0, 7.0, 9.0]);
    }
}
