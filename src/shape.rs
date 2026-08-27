use crate::error::{Result, TensorError};

/// Tensor shape (row-major logical dims).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self(dims.into())
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn elem_count(&self) -> Result<usize> {
        if self.0.is_empty() {
            return Ok(1);
        }
        self.0.iter().try_fold(1usize, |acc, &d| {
            acc.checked_mul(d)
                .ok_or_else(|| TensorError::shape("shape overflow"))
        })
    }

    pub fn dim(&self, i: usize) -> Result<usize> {
        self.0
            .get(i)
            .copied()
            .ok_or_else(|| TensorError::shape(format!("dim {i} out of rank {}", self.rank())))
    }

    /// Index from the end (`0` = last dim).
    pub fn dim_back(&self, from_end: usize) -> Result<usize> {
        let r = self.rank();
        if from_end >= r {
            return Err(TensorError::shape(format!(
                "dim_back {from_end} out of rank {r}"
            )));
        }
        Ok(self.0[r - 1 - from_end])
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
