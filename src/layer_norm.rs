//! LayerNorm over the last dimension.

use rayon::prelude::*;

#[cfg(test)]
use crate::device::Device;
use crate::error::{TensorError, TensorResult};
use crate::tensor::Tensor;

/// Affine LayerNorm: `y = (x - mean) / sqrt(var + eps) * weight + bias`.
#[derive(Debug, Clone)]
pub(crate) struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
    dim: usize,
}

impl LayerNorm {
    pub(crate) fn new(weight: Tensor, bias: Tensor, eps: f32) -> TensorResult<Self> {
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

    #[cfg(test)]
    pub(crate) fn from_slices(
        weight: &[f32],
        bias: &[f32],
        eps: f32,
        device: Device,
    ) -> TensorResult<Self> {
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

    #[cfg(test)]
    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    /// `x` last dim must equal `dim`. Any rank whose last axis is `dim`.
    pub(crate) fn forward(&self, x: &Tensor) -> TensorResult<Tensor> {
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

        if outer >= 8 {
            let out_ptr = out.as_mut_ptr() as usize;
            let xs_ptr = xs.as_ptr() as usize;
            let w_ptr = w.as_ptr() as usize;
            let b_ptr = b.as_ptr() as usize;
            (0..outer).into_par_iter().for_each(|o| {
                // SAFETY: each `o` writes a disjoint `dim`-wide slab of `out`;
                // `xs`/`w`/`b` are read-only for the duration of the parallel map.
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
        let w = [1.0f32; 4];
        let b = [0.0f32; 4];
        let ln = LayerNorm::from_slices(&w, &b, 1e-5, Device::Cpu).unwrap();
        assert_eq!(ln.dim(), 4);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4], Device::Cpu).unwrap();
        let y = ln.forward(&x).unwrap();
        let mean: f32 = y.as_slice().iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean {mean}");
    }

    #[test]
    fn layer_norm_parallel_path() {
        let w = [1.0f32; 2];
        let b = [0.0f32; 2];
        let ln = LayerNorm::from_slices(&w, &b, 1e-5, Device::Cpu).unwrap();
        let mut data = Vec::new();
        for i in 0..8 {
            data.push(i as f32);
            data.push((i + 1) as f32);
        }
        let x = Tensor::from_vec(data, vec![8, 2], Device::Cpu).unwrap();
        let y = ln.forward(&x).unwrap();
        assert_eq!(y.dims(), &[8, 2]);
    }

    #[test]
    fn layer_norm_rejects_mismatch() {
        assert!(LayerNorm::from_slices(&[1.0], &[0.0, 0.0], 1e-5, Device::Cpu).is_err());
        let ln = LayerNorm::from_slices(&[1.0, 1.0], &[0.0, 0.0], 1e-5, Device::Cpu).unwrap();
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3], Device::Cpu).unwrap();
        assert!(ln.forward(&x).is_err());
    }
}
