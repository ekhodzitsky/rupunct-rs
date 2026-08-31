use crate::error::{TensorError, TensorResult};
use crate::q8::Q8Weight;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
enum Weight {
    F32(Tensor),
    Q8(Q8Weight),
}

/// Linear: `y = x @ W^T + b`, weight layout `[out_dim, in_dim]`.
/// Q8_0 stays packed; GEMM dequants an output tile then F32 BLAS.
#[derive(Debug, Clone)]
pub(crate) struct Linear {
    weight: Weight,
    bias: Option<Tensor>,
}

impl Linear {
    pub(crate) fn new(weight: Tensor, bias: Option<Tensor>) -> TensorResult<Self> {
        if weight.rank() != 2 {
            return Err(TensorError::shape("Linear weight must be rank-2 [out,in]"));
        }
        let out = weight.dim(0)?;
        if let Some(ref b) = bias
            && (b.rank() != 1 || b.dim(0)? != out)
        {
            return Err(TensorError::shape(format!(
                "bias shape {:?} != [out={out}]",
                b.dims()
            )));
        }
        Ok(Self {
            weight: Weight::F32(weight),
            bias,
        })
    }

    pub(crate) fn from_q8(weight: Q8Weight, bias: Option<Tensor>) -> TensorResult<Self> {
        let shape = weight.shape();
        if shape.len() != 2 {
            return Err(TensorError::shape(
                "Linear Q8 weight must be rank-2 [out,in]",
            ));
        }
        let out = shape[0];
        if let Some(ref b) = bias
            && (b.rank() != 1 || b.dim(0)? != out)
        {
            return Err(TensorError::shape(format!(
                "bias shape {:?} != [out={out}]",
                b.dims()
            )));
        }
        Ok(Self {
            weight: Weight::Q8(weight),
            bias,
        })
    }

    #[cfg(test)]
    pub(crate) fn from_slices(
        out_dim: usize,
        in_dim: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
    ) -> TensorResult<Self> {
        let w = Tensor::from_vec(weight.to_vec(), vec![out_dim, in_dim])?;
        let b = match bias {
            Some(b) => Some(Tensor::from_vec(b.to_vec(), vec![out_dim])?),
            None => None,
        };
        Self::new(w, b)
    }

    pub(crate) fn out_dim(&self) -> usize {
        match &self.weight {
            Weight::F32(t) => t.dims()[0],
            Weight::Q8(q) => q.shape()[0],
        }
    }

    pub(crate) fn in_dim(&self) -> usize {
        match &self.weight {
            Weight::F32(t) => t.dims()[1],
            Weight::Q8(q) => q.shape()[1],
        }
    }

    /// `x` rank-2 `[batch, in]` → `[batch, out]`.
    pub(crate) fn forward(&self, x: &Tensor) -> TensorResult<Tensor> {
        if x.rank() != 2 {
            return Err(TensorError::shape(
                "Linear::forward supports rank-2 [batch, in] only",
            ));
        }
        let in_dim = self.in_dim();
        if x.dim(1)? != in_dim {
            return Err(TensorError::shape(format!(
                "x last dim {} != in_dim {in_dim}",
                x.dim(1)?
            )));
        }
        let mut y = match &self.weight {
            Weight::F32(t) => x.matmul_weight_t(t)?,
            Weight::Q8(q) => {
                let m = x.dim(0)?;
                let n = q.shape()[0];
                let n_out = m
                    .checked_mul(n)
                    .ok_or_else(|| TensorError::shape("m*n overflow"))?;
                let mut out = vec![0.0f32; n_out];
                q.matmul_nt(m, x.as_slice(), &mut out)?;
                Tensor::from_vec(out, vec![m, n])?
            }
        };
        if let Some(ref b) = self.bias {
            y.add_bias_2d_inplace(b)?;
        }
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_identity_bias() {
        let layer = Linear::from_slices(2, 2, &[1.0, 0.0, 0.0, 1.0], Some(&[1.0, -1.0])).unwrap();
        assert_eq!(layer.out_dim(), 2);
        assert_eq!(layer.in_dim(), 2);
        let x = Tensor::from_vec(vec![3.0, 5.0], vec![1, 2]).unwrap();
        let y = layer.forward(&x).unwrap();
        assert!((y.as_slice()[0] - 4.0).abs() < 1e-5);
        assert!((y.as_slice()[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn linear_no_bias() {
        let layer = Linear::from_slices(1, 2, &[1.0, 1.0], None).unwrap();
        let x = Tensor::from_vec(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let y = layer.forward(&x).unwrap();
        assert!((y.as_slice()[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn linear_rejects_bad_shapes() {
        let w = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        assert!(Linear::new(w, None).is_err());
        let w = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0], vec![1]).unwrap();
        assert!(Linear::new(w, Some(b)).is_err());
    }

    #[test]
    fn forward_rejects_rank_and_in_dim() {
        let layer = Linear::from_slices(2, 2, &[1.0, 0.0, 0.0, 1.0], None).unwrap();
        let x = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        assert!(layer.forward(&x).is_err());
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        assert!(layer.forward(&x).is_err());
    }
}
