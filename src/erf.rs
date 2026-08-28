//! In-tree `erf` for BERT GELU (no `libm` crate).

/// `erf(x)` via Abramowitz & Stegun 7.1.26 in f64, rounded to f32.
/// Max |err| ~1.5e-7 — one ulp of f32, enough for label argmax.
#[inline]
pub(crate) fn erff(x: f32) -> f32 {
    let z = x as f64;
    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let a = z.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-a * a).exp();
    (sign * y) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_known() {
        assert!(erff(0.0).abs() < 1e-7);
        assert!((erff(1.0) - 0.8427).abs() < 2e-4);
        assert!((erff(-1.0) + 0.8427).abs() < 2e-4);
    }
}
