//! Contiguous F32 GEMM (Accelerate on macOS, `gemm` crate elsewhere).

use crate::blas::{gemm_attn_v, gemm_nn, gemm_nt_weight, gemm_scores};
use crate::error::{Result, TensorError};

/// `c = a @ b` with shapes `(m,k) x (k,n) -> (m,n)` (row-major).
pub fn matmul_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) -> Result<()> {
    if a.len()
        != m.checked_mul(k)
            .ok_or_else(|| TensorError::shape("m*k overflow"))?
    {
        return Err(TensorError::shape(format!(
            "a len {} != m*k {}",
            a.len(),
            m * k
        )));
    }
    if b.len()
        != k.checked_mul(n)
            .ok_or_else(|| TensorError::shape("k*n overflow"))?
    {
        return Err(TensorError::shape(format!(
            "b len {} != k*n {}",
            b.len(),
            k * n
        )));
    }
    if c.len()
        != m.checked_mul(n)
            .ok_or_else(|| TensorError::shape("m*n overflow"))?
    {
        return Err(TensorError::shape(format!(
            "c len {} != m*n {}",
            c.len(),
            m * n
        )));
    }
    gemm_nn(m, k, n, a, b, c);
    Ok(())
}

/// `c = a @ b^T` with `a: (m,k)`, `b: (n,k)` → `c: (m,n)`.
pub fn matmul_f32_bt(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b_t: &[f32],
    c: &mut [f32],
) -> Result<()> {
    if b_t.len()
        != n.checked_mul(k)
            .ok_or_else(|| TensorError::shape("n*k overflow"))?
    {
        return Err(TensorError::shape(format!(
            "b_t len {} != n*k {}",
            b_t.len(),
            n * k
        )));
    }
    if a.len() != m * k {
        return Err(TensorError::shape(format!(
            "a len {} != m*k {}",
            a.len(),
            m * k
        )));
    }
    if c.len() != m * n {
        return Err(TensorError::shape(format!(
            "c len {} != m*n {}",
            c.len(),
            m * n
        )));
    }
    gemm_nt_weight(m, k, n, a, b_t, c);
    Ok(())
}

/// `scores[t,t] = q[t,d] @ k[t,d]^T`.
pub fn matmul_f32_nt(t: usize, d: usize, q: &[f32], k: &[f32], scores: &mut [f32]) -> Result<()> {
    if q.len() != t * d || k.len() != t * d || scores.len() != t * t {
        return Err(TensorError::shape("matmul_f32_nt size mismatch"));
    }
    gemm_scores(t, d, q, k, scores);
    Ok(())
}

/// `out[t,d] = scores[t,t] @ v[t,d]`.
pub fn matmul_f32_nn_attn(
    t: usize,
    d: usize,
    scores: &[f32],
    v: &[f32],
    out: &mut [f32],
) -> Result<()> {
    if scores.len() != t * t || v.len() != t * d || out.len() != t * d {
        return Err(TensorError::shape("matmul_f32_nn_attn size mismatch"));
    }
    gemm_attn_v(t, d, scores, v, out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_identity() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let i = [1.0f32, 0.0, 0.0, 1.0];
        let mut c = [0.0f32; 4];
        matmul_f32(2, 2, 2, &a, &i, &mut c).unwrap();
        assert_eq!(c, a);
    }

    #[test]
    fn matmul_bt_matches_explicit() {
        let a = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let w = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let mut c = [0.0f32; 8];
        matmul_f32_bt(2, 3, 4, &a, &w, &mut c).unwrap();
        assert!((c[0] - 1.0).abs() < 1e-5);
        assert!((c[1] - 0.0).abs() < 1e-5);
        assert!((c[2] - 0.0).abs() < 1e-5);
        assert!((c[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn matmul_nt_dot_rows() {
        let q = [1.0f32, 0.0, 0.0, 1.0];
        let k = [1.0f32, 0.0, 0.0, 1.0];
        let mut s = [0.0f32; 4];
        matmul_f32_nt(2, 2, &q, &k, &mut s).unwrap();
        assert!((s[0] - 1.0).abs() < 1e-5);
        assert!((s[3] - 1.0).abs() < 1e-5);
    }
}
