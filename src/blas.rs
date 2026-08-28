//! Thin F32 GEMM backends.
//!
//! - **macOS:** Apple Accelerate `sgemm_`.
//! - **elsewhere:** `gemm` crate (NEON/AVX).

#![allow(clippy::too_many_arguments)]

/// `C = alpha * op(A) * op(B) + beta * C` with **column-major** storage (BLAS).
///
/// `transa`/`transb`: `b'N'` or `b'T'`. Leading dims `lda`/`ldb`/`ldc` follow BLAS.
///
/// # Safety
/// `a`, `b`, and `c` must be large enough for the BLAS leading dimensions
/// (`lda`/`ldb`/`ldc`) and the `m`/`n`/`k` extents. Slices must not alias
/// `c` with `a` or `b`.
#[cfg(target_os = "macos")]
pub unsafe fn sgemm(
    transa: u8,
    transb: u8,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: &[f32],
    lda: i32,
    b: &[f32],
    ldb: i32,
    beta: f32,
    c: &mut [f32],
    ldc: i32,
) {
    // SAFETY: caller guarantees dimensions match the slices.
    unsafe {
        ffi::sgemm_ffi(
            &(transa as core::ffi::c_char),
            &(transb as core::ffi::c_char),
            &m,
            &n,
            &k,
            &alpha,
            a.as_ptr(),
            &lda,
            b.as_ptr(),
            &ldb,
            &beta,
            c.as_mut_ptr(),
            &ldc,
        );
    }
}

#[cfg(target_os = "macos")]
mod ffi {
    use core::ffi::{c_char, c_float, c_int};

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        #[link_name = "sgemm_"]
        pub fn sgemm_ffi(
            transa: *const c_char,
            transb: *const c_char,
            m: *const c_int,
            n: *const c_int,
            k: *const c_int,
            alpha: *const c_float,
            a: *const c_float,
            lda: *const c_int,
            b: *const c_float,
            ldb: *const c_int,
            beta: *const c_float,
            c: *mut c_float,
            ldc: *const c_int,
        );
    }
}

/// Row-major `y[m,n] = alpha * (x[m,k] @ W_col[k,n]) + beta * y`.
///
/// `w_col` is column-major k×n (`w[ki + col*k]`).
pub fn gemm_row_x_col(
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    lhs: &[f32],
    w_col: &[f32],
    beta: f32,
    dst: &mut [f32],
) {
    assert_eq!(lhs.len(), m * k);
    assert_eq!(w_col.len(), k * n);
    assert_eq!(dst.len(), m * n);

    #[cfg(target_os = "macos")]
    {
        // SAFETY: debug_assert checked lens; row-major (m,k)×col-major (k,n)
        // maps to this sgemm layout; dst is uniquely borrowed.
        unsafe {
            sgemm(
                b'T', b'N', n as i32, m as i32, k as i32, alpha, w_col, k as i32, lhs, k as i32,
                beta, dst, n as i32,
            );
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        use gemm::{Parallelism, gemm};
        let read_dst = beta != 0.0;
        // SAFETY: lengths checked; `dst` is uniquely borrowed. `w_col` is
        // [n,k] row-major = k×n column-major (lda = k).
        unsafe {
            gemm(
                m,
                n,
                k,
                dst.as_mut_ptr(),
                1,
                n as isize,
                read_dst,
                lhs.as_ptr(),
                1,
                k as isize,
                w_col.as_ptr(),
                k as isize,
                1,
                beta,
                alpha,
                false,
                false,
                false,
                Parallelism::Rayon(0),
            );
        }
    }
}

/// Row-major `c[m,n] = a[m,k] @ b[k,n]`.
#[cfg(test)]
pub fn gemm_nn(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    #[cfg(target_os = "macos")]
    {
        // SAFETY: lengths checked; `c` is uniquely borrowed.
        unsafe {
            sgemm(
                b'N', b'N', n as i32, m as i32, k as i32, 1.0, b, n as i32, a, k as i32, 0.0, c,
                n as i32,
            );
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        use gemm::{Parallelism, gemm};
        // SAFETY: lengths checked; `c` is uniquely borrowed.
        unsafe {
            gemm(
                m,
                n,
                k,
                c.as_mut_ptr(),
                1,
                n as isize,
                false,
                a.as_ptr(),
                1,
                k as isize,
                b.as_ptr(),
                1,
                n as isize,
                0.0f32,
                1.0f32,
                false,
                false,
                false,
                Parallelism::Rayon(0),
            );
        }
    }
}

/// Row-major `c[m,n] = a[m,k] @ b_t[n,k]^T` (weight rows).
pub fn gemm_nt_weight(m: usize, k: usize, n: usize, a: &[f32], b_t: &[f32], c: &mut [f32]) {
    gemm_row_x_col(m, k, n, 1.0, a, b_t, 0.0, c);
}

/// `scores[t,t] = q[t,d] @ k[t,d]^T` with Q/K in `[t, ld]` and the head at `off`.
pub fn gemm_scores_strided(
    t: usize,
    d: usize,
    q: &[f32],
    k: &[f32],
    ld: usize,
    off: usize,
    scores: &mut [f32],
) {
    assert_eq!(scores.len(), t * t);
    assert!(ld >= d && off + d <= ld);
    assert!(
        q.len()
            >= t.saturating_sub(1)
                .saturating_mul(ld)
                .saturating_add(off + d)
    );
    assert!(
        k.len()
            >= t.saturating_sub(1)
                .saturating_mul(ld)
                .saturating_add(off + d)
    );
    let qh = &q[off..];
    let kh = &k[off..];

    #[cfg(target_os = "macos")]
    {
        // SAFETY: lengths checked; `scores` is uniquely borrowed; lda = `ld`.
        unsafe {
            sgemm(
                b'T', b'N', t as i32, t as i32, d as i32, 1.0, kh, ld as i32, qh, ld as i32, 0.0,
                scores, t as i32,
            );
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        use gemm::{Parallelism, gemm};
        // SAFETY: lengths checked; `scores` is uniquely borrowed.
        unsafe {
            gemm(
                t,
                t,
                d,
                scores.as_mut_ptr(),
                1,
                t as isize,
                false,
                qh.as_ptr(),
                1,
                ld as isize,
                kh.as_ptr(),
                ld as isize,
                1,
                0.0f32,
                1.0f32,
                false,
                false,
                false,
                Parallelism::None,
            );
        }
    }
}

/// `out[t,d] = scores[t,t] @ v[t,d]` with V/out in `[t, ld]` and the head at `off`.
pub fn gemm_attn_v_strided(
    t: usize,
    d: usize,
    scores: &[f32],
    v: &[f32],
    out: &mut [f32],
    ld: usize,
    off: usize,
) {
    assert_eq!(scores.len(), t * t);
    assert!(ld >= d && off + d <= ld);
    assert!(
        v.len()
            >= t.saturating_sub(1)
                .saturating_mul(ld)
                .saturating_add(off + d)
    );
    assert!(
        out.len()
            >= t.saturating_sub(1)
                .saturating_mul(ld)
                .saturating_add(off + d)
    );
    let vh = &v[off..];
    let oh = &mut out[off..];

    #[cfg(target_os = "macos")]
    {
        // SAFETY: lengths checked; `oh` is uniquely borrowed; lda = `ld`.
        unsafe {
            sgemm(
                b'N', b'N', d as i32, t as i32, t as i32, 1.0, vh, ld as i32, scores, t as i32,
                0.0, oh, ld as i32,
            );
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        use gemm::{Parallelism, gemm};
        // SAFETY: lengths checked; `oh` is uniquely borrowed.
        unsafe {
            gemm(
                t,
                d,
                t,
                oh.as_mut_ptr(),
                1,
                ld as isize,
                false,
                scores.as_ptr(),
                1,
                t as isize,
                vh.as_ptr(),
                1,
                ld as isize,
                0.0f32,
                1.0f32,
                false,
                false,
                false,
                Parallelism::None,
            );
        }
    }
}
