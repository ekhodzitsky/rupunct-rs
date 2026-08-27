//! Thin F32 GEMM backends.
//!
//! - **macOS:** Apple Accelerate `sgemm_` (column-major BLAS, same as vendor candle).
//! - **elsewhere:** `gemm` crate (NEON/AVX), used for product Linux targets.

#![allow(clippy::too_many_arguments)]

/// Ensure Accelerate can use multiple cores for large product GEMMs.
///
/// Safe to call repeatedly; only sets env when unset. Prefer calling once at
/// engine boot via `apply_candle_thread_env`; this is a belt-and-suspenders
/// path for pure `gigastt-tensor` benches.
#[cfg(target_os = "macos")]
pub fn ensure_accelerate_threads(n: usize) {
    let n = n.max(1);
    let s = n.to_string();
    if std::env::var_os("VECLIB_MAXIMUM_THREADS").is_none() {
        // SAFETY: set before first sgemm; only when operator has not overridden.
        unsafe { std::env::set_var("VECLIB_MAXIMUM_THREADS", &s) };
    }
    if std::env::var_os("OMP_NUM_THREADS").is_none() {
        unsafe { std::env::set_var("OMP_NUM_THREADS", &s) };
    }
}

/// `C = alpha * op(A) * op(B) + beta * C` with **column-major** storage (BLAS).
///
/// `transa`/`transb`: `b'N'` or `b'T'`. Leading dims `lda`/`ldb`/`ldc` follow BLAS.
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
/// `w_col` is column-major k×n (`w[ki + col*k]`), matching Q8 dequant pack.
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
    debug_assert_eq!(lhs.len(), m * k);
    debug_assert_eq!(w_col.len(), k * n);
    debug_assert_eq!(dst.len(), m * n);

    #[cfg(target_os = "macos")]
    {
        // BLAS: C(n,m) col-major = A^T(n,k) * B(k,m)
        // with A = w_col (k×n col-major), B = lhs^T as (k×m) from row-major m×k lhs.
        // Result C col-major n×m is row-major m×n dst layout.
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
        use rayon::prelude::*;
        let n_threads = rayon::current_num_threads().max(1);
        let rows_per = m.div_ceil(n_threads).max(1);
        let w_ptr = w_col.as_ptr() as usize;
        let w_len = w_col.len();
        let lhs_base = lhs.as_ptr() as usize;
        let dst_base = dst.as_mut_ptr() as usize;
        // When beta != 0 we need read_dst; product path uses beta=0 overwrite.
        let read_dst = beta != 0.0;
        (0..n_threads).into_par_iter().for_each(|tid| {
            let r0 = tid * rows_per;
            if r0 >= m {
                return;
            }
            let r1 = m.min(r0 + rows_per);
            let rows = r1 - r0;
            let lhs_slab = unsafe {
                std::slice::from_raw_parts((lhs_base as *const f32).add(r0 * k), rows * k)
            };
            let dst_slab = unsafe {
                std::slice::from_raw_parts_mut((dst_base as *mut f32).add(r0 * n), rows * n)
            };
            let w_slab = unsafe { std::slice::from_raw_parts(w_ptr as *const f32, w_len) };
            unsafe {
                gemm(
                    rows,
                    n,
                    k,
                    dst_slab.as_mut_ptr(),
                    1,
                    n as isize,
                    read_dst,
                    lhs_slab.as_ptr(),
                    1,
                    k as isize,
                    w_slab.as_ptr(),
                    k as isize,
                    1,
                    beta,
                    alpha,
                    false,
                    false,
                    false,
                    Parallelism::None,
                );
            }
        });
    }
}

/// Row-major `c[m,n] = a[m,k] @ b[k,n]`.
pub fn gemm_nn(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    #[cfg(target_os = "macos")]
    {
        // C(n,m) = B^T(n,k) * A^T(k,m) with row-major A,B → col-major of transposes.
        // Easier: C_rm = A_rm @ B_rm ⇔ C_cm^T = ... use sgemm on transposed view:
        // treat row-major m×n as col-major n×m.
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
    // b_t row-major n×k is col-major pack of W (k×n) with col j = row j of b_t.
    gemm_row_x_col(m, k, n, 1.0, a, b_t, 0.0, c);
}

/// `scores[t,t] = q[t,d] @ k[t,d]^T`.
pub fn gemm_scores(t: usize, d: usize, q: &[f32], k: &[f32], scores: &mut [f32]) {
    debug_assert_eq!(q.len(), t * d);
    debug_assert_eq!(k.len(), t * d);
    debug_assert_eq!(scores.len(), t * t);

    #[cfg(target_os = "macos")]
    {
        // scores = Q @ K^T. BLAS: C(t,t) = K_cm_view * Q...
        // Vendor: sgemm('T','N', t,t,d, 1, k, d, q, d, 0, scores, t)
        // with row-major Q,K of shape t×d treated as col-major d×t of their transpose.
        unsafe {
            sgemm(
                b'T', b'N', t as i32, t as i32, d as i32, 1.0, k, d as i32, q, d as i32, 0.0,
                scores, t as i32,
            );
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        use gemm::{Parallelism, gemm};
        unsafe {
            gemm(
                t,
                t,
                d,
                scores.as_mut_ptr(),
                1,
                t as isize,
                false,
                q.as_ptr(),
                1,
                d as isize,
                k.as_ptr(),
                d as isize,
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

/// `out[t,d] = scores[t,t] @ v[t,d]`.
pub fn gemm_attn_v(t: usize, d: usize, scores: &[f32], v: &[f32], out: &mut [f32]) {
    debug_assert_eq!(scores.len(), t * t);
    debug_assert_eq!(v.len(), t * d);
    debug_assert_eq!(out.len(), t * d);

    #[cfg(target_os = "macos")]
    {
        // Vendor: sgemm('N','N', d, t, t, 1, v, d, scores, t, 0, out, d)
        unsafe {
            sgemm(
                b'N', b'N', d as i32, t as i32, t as i32, 1.0, v, d as i32, scores, t as i32, 0.0,
                out, d as i32,
            );
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        use gemm::{Parallelism, gemm};
        unsafe {
            gemm(
                t,
                d,
                t,
                out.as_mut_ptr(),
                1,
                d as isize,
                false,
                scores.as_ptr(),
                1,
                t as isize,
                v.as_ptr(),
                1,
                d as isize,
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
