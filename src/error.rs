use std::io;
use std::path::Path;

use thiserror::Error;

/// Recoverable failure from load or restore.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum Error {
    #[error("failed to read {path}: {kind}")]
    Io { path: String, kind: io::ErrorKind },
    #[error("{0}")]
    Config(String),
    #[error("{0}")]
    Tokenizer(String),
    #[error("{0}")]
    Model(String),
}

impl Error {
    pub(crate) fn io(path: impl AsRef<Path>, err: io::Error) -> Self {
        Self::Io {
            path: path.as_ref().display().to_string(),
            kind: err.kind(),
        }
    }

    pub(crate) fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    pub(crate) fn tokenizer(msg: impl Into<String>) -> Self {
        Self::Tokenizer(msg.into())
    }

    pub(crate) fn model(msg: impl Into<String>) -> Self {
        Self::Model(msg.into())
    }
}

/// Tensor construction and op failures (internal).
#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum TensorError {
    #[error("shape error: {0}")]
    Shape(String),
    #[error("{0}")]
    Msg(String),
}

impl TensorError {
    pub(crate) fn shape(msg: impl Into<String>) -> Self {
        Self::Shape(msg.into())
    }

    pub(crate) fn msg(msg: impl Into<String>) -> Self {
        Self::Msg(msg.into())
    }
}

impl From<TensorError> for Error {
    fn from(err: TensorError) -> Self {
        Error::Model(err.to_string())
    }
}

pub(crate) type TensorResult<T> = std::result::Result<T, TensorError>;
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
#[path = "error_tests.rs"]
mod tests;
