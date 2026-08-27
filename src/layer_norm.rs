//! LayerNorm over the last dimension (product Conformer / rupunct style).

use rayon::prelude::*;

use crate::device::Device;
use crate::error::{Result, TensorError};
use crate::tensor::Tensor;

/// Affine LayerNorm: `y = (x - mean) / sqrt(var + eps) * weight + bias`.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    weight: Tensor, // [dim]
    bias: Tensor,   // [dim]
    eps: f32,
    dim: usize,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f32) -> Result<Self> {
        if weight.rank() != 1 || bias.rank() != 1 || weight.dim(0)? != bias.dim(0)? {
            return Err(TensorError::shape(
                "LayerNorm weight/bias must be rank-1 equal",
            ));
        }
        let dim = weight.dim(0)?;
        Ok(Self {
            weight,
            bias,
            eps,
            dim,
        })
    }

    pub fn from_slices(weight: &[f32], bias: &[f32], eps: f32, device: Device) -> Result<Self> {
        if weight.len() != bias.len() {
            return Err(TensorError::shape("LayerNorm weight/bias len mismatch"));
        }
        let dim = weight.len();
        Self::new(
            Tensor::from_vec(weight.to_vec(), vec![dim], device)?,
            Tensor::from_vec(bias.to_vec(), vec![dim], device)?,
            eps,
        )
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// `x` last dim must equal `dim`. Supports rank-2 `[n, dim]` and rank-3 `[b,t,dim]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let rank = x.rank();
        if rank < 1 || x.dim(rank - 1)? != self.dim {
            return Err(TensorError::shape(format!(
                "LayerNorm: last dim {:?} != {}",
                x.dims(),
                self.dim
            )));
        }
        let dim = self.dim;
        let outer = x.elem_count() / dim;
        let mut out = vec![0.0f32; x.elem_count()];
        let xs = x.as_slice();
        let w = self.weight.as_slice();
        let b = self.bias.as_slice();
        let eps = self.eps;

        // Parallel over token rows when the map is large (product: T≈100+, dim=768).
        if outer >= 8 {
            let out_ptr = out.as_mut_ptr() as usize;
            let xs_ptr = xs.as_ptr() as usize;
            let w_ptr = w.as_ptr() as usize;
            let b_ptr = b.as_ptr() as usize;
            (0..outer).into_par_iter().for_each(|o| {
                let xs = unsafe { std::slice::from_raw_parts(xs_ptr as *const f32, outer * dim) };
                let w = unsafe { std::slice::from_raw_parts(w_ptr as *const f32, dim) };
                let b = unsafe { std::slice::from_raw_parts(b_ptr as *const f32, dim) };
                let out =
                    unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, outer * dim) };
                ln_row(
                    &xs[o * dim..(o + 1) * dim],
                    w,
                    b,
                    eps,
                    &mut out[o * dim..(o + 1) * dim],
                );
            });
        } else {
            for o in 0..outer {
                let base = o * dim;
                ln_row(&xs[base..base + dim], w, b, eps, &mut out[base..base + dim]);
            }
        }
        Tensor::from_vec(out, x.dims().to_vec(), x.device())
    }
}

#[inline]
fn ln_row(row: &[f32], w: &[f32], b: &[f32], eps: f32, out: &mut [f32]) {
    let dim = row.len();
    let mut mean = 0.0f32;
    for &v in row {
        mean += v;
    }
    mean /= dim as f32;
    let mut var = 0.0f32;
    for &v in row {
        let d = v - mean;
        var += d * d;
    }
    var /= dim as f32;
    let inv = 1.0 / (var + eps).sqrt();
    for j in 0..dim {
        let n = (row[j] - mean) * inv;
        out[j] = n * w[j] + b[j];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_norm_zero_mean_unit_var_approx() {
        let device = Device::Cpu;
        let w = [1.0f32; 4];
        let b = [0.0f32; 4];
        let ln = LayerNorm::from_slices(&w, &b, 1e-5, device).unwrap();
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4], device).unwrap();
        let y = ln.forward(&x).unwrap();
        let mean: f32 = y.as_slice().iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean {mean}");
    }
}
