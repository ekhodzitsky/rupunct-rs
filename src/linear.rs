//! F32 linear layer: `y = x @ W^T + b`, weight layout `[out_dim, in_dim]`.
//!
//! Matches `candle_nn::Linear` contract used by decoder/joiner and aux models.
//! Product encoder uses Q8 [`QLinear`] in this crate.

use crate::device::Device;
use crate::error::{Result, TensorError};
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct Linear {
    weight: Tensor, // [out, in]
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
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
        Ok(Self { weight, bias })
    }

    pub fn from_slices(
        out_dim: usize,
        in_dim: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
        device: Device,
    ) -> Result<Self> {
        let w = Tensor::from_vec(weight.to_vec(), vec![out_dim, in_dim], device)?;
        let b = match bias {
            Some(b) => Some(Tensor::from_vec(b.to_vec(), vec![out_dim], device)?),
            None => None,
        };
        Self::new(w, b)
    }

    pub fn out_dim(&self) -> usize {
        self.weight.dim(0).unwrap_or(0)
    }

    pub fn in_dim(&self) -> usize {
        self.weight.dim(1).unwrap_or(0)
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// `x` rank-2 `[batch, in]` → `[batch, out]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.rank() != 2 {
            return Err(TensorError::shape(
                "Linear::forward spike supports rank-2 [batch, in] only",
            ));
        }
        let in_dim = self.in_dim();
        if x.dim(1)? != in_dim {
            return Err(TensorError::shape(format!(
                "x last dim {} != in_dim {in_dim}",
                x.dim(1)?
            )));
        }
        let mut y = x.matmul_weight_t(&self.weight)?;
        if let Some(ref b) = self.bias {
            y = y.add_bias_2d(b)?;
        }
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_identity_bias() {
        // W = I_2, b = [1, -1]
        let layer =
            Linear::from_slices(2, 2, &[1.0, 0.0, 0.0, 1.0], Some(&[1.0, -1.0]), Device::Cpu)
                .unwrap();
        let x = Tensor::from_vec(vec![3.0, 5.0], vec![1, 2], Device::Cpu).unwrap();
        let y = layer.forward(&x).unwrap();
        assert!((y.as_slice()[0] - 4.0).abs() < 1e-5);
        assert!((y.as_slice()[1] - 4.0).abs() < 1e-5);
    }
}
