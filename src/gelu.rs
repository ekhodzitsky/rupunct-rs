//! BERT `gelu` (erf form): `0.5 * x * (1 + erf(x / sqrt(2)))`.

use crate::error::Result;
use crate::tensor::Tensor;

#[inline]
pub(crate) fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

pub(crate) fn gelu_tensor(x: &Tensor) -> Result<Tensor> {
    let data: Vec<f32> = x.as_slice().iter().map(|&v| gelu(v)).collect();
    Ok(Tensor::from_vec(data, x.dims().to_vec())?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_zero_is_zero() {
        assert!(gelu(0.0).abs() < 1e-7);
    }

    #[test]
    fn gelu_known_values() {
        assert!((gelu(1.0) - 0.8413).abs() < 1e-3);
        assert!((gelu(-1.0) + 0.1587).abs() < 1e-3);
    }

    #[test]
    fn gelu_tensor_preserves_shape() {
        let x = Tensor::from_vec(vec![0.0, 1.0, -1.0, 2.0], vec![2, 2]).unwrap();
        let y = gelu_tensor(&x).unwrap();
        assert_eq!(y.dims(), &[2, 2]);
        assert!(y.as_slice()[0].abs() < 1e-7);
    }
}
