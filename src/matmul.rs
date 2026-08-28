//! Contiguous F32 GEMM (Accelerate on macOS, `gemm` crate elsewhere).

#[cfg(test)]
use crate::blas::gemm_nn;
use crate::blas::gemm_nt_weight;
use crate::error::{TensorError, TensorResult};

/// `c = a @ b` with shapes `(m,k) x (k,n) -> (m,n)` (row-major).
#[cfg(test)]
pub(crate) fn matmul_f32(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> TensorResult<()> {
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
pub(crate) fn matmul_f32_bt(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b_t: &[f32],
    c: &mut [f32],
) -> TensorResult<()> {
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
    let mk = m
        .checked_mul(k)
        .ok_or_else(|| TensorError::shape("m*k overflow"))?;
    let mn = m
        .checked_mul(n)
        .ok_or_else(|| TensorError::shape("m*n overflow"))?;
    if a.len() != mk {
        return Err(TensorError::shape(format!("a len {} != m*k {mk}", a.len())));
    }
    if c.len() != mn {
        return Err(TensorError::shape(format!("c len {} != m*n {mn}", c.len())));
    }
    gemm_nt_weight(m, k, n, a, b_t, c);
    Ok(())
}

/// `scores[t,t] = q[t,d] @ k[t,d]^T`.
#[cfg(test)]
pub(crate) fn matmul_f32_nt(
    t: usize,
    d: usize,
    q: &[f32],
    k: &[f32],
    scores: &mut [f32],
) -> TensorResult<()> {
    matmul_f32_nt_strided(t, d, q, k, d, 0, scores)
}

/// Q/K are `[t, ld]` with the `d`-wide head starting at `off`.
pub(crate) fn matmul_f32_nt_strided(
    t: usize,
    d: usize,
    q: &[f32],
    k: &[f32],
    ld: usize,
    off: usize,
    scores: &mut [f32],
) -> TensorResult<()> {
    let need = t
        .saturating_sub(1)
        .saturating_mul(ld)
        .saturating_add(off + d);
    let tt = t
        .checked_mul(t)
        .ok_or_else(|| TensorError::shape("t*t overflow"))?;
    if ld < d || off + d > ld || q.len() < need || k.len() < need || scores.len() != tt {
        return Err(TensorError::shape("matmul_f32_nt_strided size mismatch"));
    }
    crate::blas::gemm_scores_strided(t, d, q, k, ld, off, scores);
    Ok(())
}

/// `out[t,d] = scores[t,t] @ v[t,d]`.
#[cfg(test)]
pub(crate) fn matmul_f32_nn_attn(
    t: usize,
    d: usize,
    scores: &[f32],
    v: &[f32],
    out: &mut [f32],
) -> TensorResult<()> {
    matmul_f32_nn_attn_strided(t, d, scores, v, out, d, 0)
}

/// V/out are `[t, ld]` with the head at `off`.
pub(crate) fn matmul_f32_nn_attn_strided(
    t: usize,
    d: usize,
    scores: &[f32],
    v: &[f32],
    out: &mut [f32],
    ld: usize,
    off: usize,
) -> TensorResult<()> {
    let need = t
        .saturating_sub(1)
        .saturating_mul(ld)
        .saturating_add(off + d);
    let tt = t
        .checked_mul(t)
        .ok_or_else(|| TensorError::shape("t*t overflow"))?;
    if ld < d || off + d > ld || v.len() < need || out.len() < need || scores.len() != tt {
        return Err(TensorError::shape(
            "matmul_f32_nn_attn_strided size mismatch",
        ));
    }
    crate::blas::gemm_attn_v_strided(t, d, scores, v, out, ld, off);
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
    fn matmul_rejects_len() {
        let a = [1.0f32];
        let mut c = [0.0f32; 4];
        assert!(matmul_f32(2, 2, 2, &a, &a, &mut c).is_err());
    }

    #[test]
    fn matmul_bt_matches_explicit() {
        let a = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let w = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let mut c = [0.0f32; 8];
        matmul_f32_bt(2, 3, 4, &a, &w, &mut c).unwrap();
        assert!((c[0] - 1.0).abs() < 1e-5);
        assert!((c[3] - 1.0).abs() < 1e-5);
        assert!(matmul_f32_bt(2, 3, 4, &[1.0], &w, &mut c).is_err());
    }

    #[test]
    fn matmul_nt_dot_rows() {
        let q = [1.0f32, 0.0, 0.0, 1.0];
        let k = [1.0f32, 0.0, 0.0, 1.0];
        let mut s = [0.0f32; 4];
        matmul_f32_nt(2, 2, &q, &k, &mut s).unwrap();
        assert!((s[0] - 1.0).abs() < 1e-5);
        assert!((s[3] - 1.0).abs() < 1e-5);
        assert!(matmul_f32_nt(2, 2, &[1.0], &k, &mut s).is_err());
    }

    #[test]
    fn matmul_attn_v_identity_scores() {
        let scores = [1.0f32, 0.0, 0.0, 1.0];
        let v = [3.0f32, 4.0, 5.0, 6.0];
        let mut out = [0.0f32; 4];
        matmul_f32_nn_attn(2, 2, &scores, &v, &mut out).unwrap();
        assert!((out[0] - 3.0).abs() < 1e-5);
        assert!((out[3] - 6.0).abs() < 1e-5);
        assert!(matmul_f32_nn_attn(2, 2, &[1.0], &v, &mut out).is_err());
    }

    #[test]
    fn strided_head_matches_packed() {
        // two tokens, hidden=4, head_dim=2, head 1 at offset 2
        let q = [1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let k = q;
        let packed_q = [0.0f32, 1.0, 1.0, 0.0];
        let mut packed = [0.0f32; 4];
        let mut strided = [0.0f32; 4];
        matmul_f32_nt(2, 2, &packed_q, &packed_q, &mut packed).unwrap();
        matmul_f32_nt_strided(2, 2, &q, &k, 4, 2, &mut strided).unwrap();
        for i in 0..4 {
            assert!((packed[i] - strided[i]).abs() < 1e-5);
        }
        let mut s = [0.0f32; 4];
        assert!(matmul_f32_nt_strided(2, 2, &[1.0], &[1.0], 2, 0, &mut s).is_err());
        let v = [1.0f32; 4];
        let mut o = [0.0f32; 4];
        assert!(matmul_f32_nn_attn_strided(2, 2, &[1.0], &v, &mut o, 2, 0).is_err());
    }
}
