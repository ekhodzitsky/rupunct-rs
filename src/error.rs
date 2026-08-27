use std::fmt;
use std::io;
use std::path::Path;

/// Recoverable failure from load or restore.
#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Io { path: String, kind: io::ErrorKind },
    Config(String),
    Tokenizer(String),
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

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, kind } => write!(f, "failed to read {path}: {kind}"),
            Self::Config(msg) | Self::Tokenizer(msg) | Self::Model(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for Error {}

/// Tensor construction and op failures (internal).
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum TensorError {
    Shape(String),
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

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(msg) => write!(f, "shape error: {msg}"),
            Self::Msg(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for TensorError {}

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
