use crate::error::{TensorError, TensorResult};

/// Tensor shape (row-major logical dims).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Shape(pub Vec<usize>);

impl Shape {
    pub(crate) fn dims(&self) -> &[usize] {
        &self.0
    }

    pub(crate) fn rank(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn elem_count(&self) -> TensorResult<usize> {
        if self.0.is_empty() {
            return Ok(1);
        }
        self.0.iter().try_fold(1usize, |acc, &d| {
            acc.checked_mul(d)
                .ok_or_else(|| TensorError::shape("shape overflow"))
        })
    }

    pub(crate) fn dim(&self, i: usize) -> TensorResult<usize> {
        self.0
            .get(i)
            .copied()
            .ok_or_else(|| TensorError::shape(format!("dim {i} out of rank {}", self.rank())))
    }
}

impl From<Vec<usize>> for Shape {
    fn from(v: Vec<usize>) -> Self {
        Self(v)
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_shape_is_scalar() {
        let s = Shape::from(vec![]);
        assert_eq!(s.rank(), 0);
        assert_eq!(s.elem_count().unwrap(), 1);
        assert_eq!(s.to_string(), "[]");
    }

    #[test]
    fn overflow_and_oob_dim() {
        let s = Shape::from(vec![usize::MAX, 2]);
        assert!(s.elem_count().is_err());
        let s = Shape::from(vec![2, 3]);
        assert_eq!(s.dim(1).unwrap(), 3);
        assert!(s.dim(2).is_err());
        assert_eq!(s.to_string(), "[2, 3]");
    }
}
