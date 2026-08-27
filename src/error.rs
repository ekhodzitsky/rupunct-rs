use thiserror::Error;

/// Errors from tensor construction and ops.
#[derive(Debug, Error)]
pub enum TensorError {
    #[error("shape error: {0}")]
    Shape(String),
    #[error("dtype error: {0}")]
    DType(String),
    #[error("device error: {0}")]
    Device(String),
    #[error("{0}")]
    Msg(String),
}

impl TensorError {
    pub fn shape(msg: impl Into<String>) -> Self {
        Self::Shape(msg.into())
    }
    pub fn msg(msg: impl Into<String>) -> Self {
        Self::Msg(msg.into())
    }
}

pub type Result<T> = std::result::Result<T, TensorError>;
