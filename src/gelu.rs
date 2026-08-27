//! BERT `gelu` (erf form): `0.5 * x * (1 + erf(x / sqrt(2)))`.

use crate::error::Result;
use crate::tensor::Tensor;

#[inline]
pub(crate) fn gelu(x: f32) -> f32 {
    let x64 = x as f64;
    let cdf = 0.5 * (1.0 + erf_approx(x64 * std::f64::consts::FRAC_1_SQRT_2));
    (x64 * cdf) as f32
}

/// A&S 7.1.26 approximation; max |err| ~1.5e-7 — enough for label argmax parity.
#[inline]
fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

pub(crate) fn gelu_tensor(x: &Tensor) -> Result<Tensor> {
    let data: Vec<f32> = x.as_slice().iter().map(|&v| gelu(v)).collect();
    Ok(Tensor::from_vec(data, x.dims().to_vec(), x.device())?)
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
        use crate::device::Device;
        let x = Tensor::from_vec(vec![0.0, 1.0, -1.0, 2.0], vec![2, 2], Device::Cpu).unwrap();
        let y = gelu_tensor(&x).unwrap();
        assert_eq!(y.dims(), &[2, 2]);
        assert!(y.as_slice()[0].abs() < 1e-7);
    }
}
