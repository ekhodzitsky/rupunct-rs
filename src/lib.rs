//! On-device Russian punctuation and casing restoration.
//!
//! BERT token-classification compatible with
//! [RUPunct/RUPunct_small](https://huggingface.co/RUPunct/RUPunct_small)
//! (`aggregation_strategy="first"`). One checkpoint resident. No network.

mod bert;
mod blas;
mod cli;
mod config;
mod encoder;
mod error;
mod gelu;
mod labels;
mod layer_norm;
mod linear;
mod matmul;
mod model;
mod punct;
mod restore;
mod shape;
mod tensor;
mod tokenizer_json;
mod unicode;
mod weights;
mod wordpiece;
mod ws;

#[cfg(test)]
mod fixture;

pub use error::{Error, Result};
pub use labels::process_token;
pub use model::{CONFIG_FILE, MODEL_FILE, TOKENIZER_FILE, default_model_dir};
pub use punct::Punctuator;
pub use restore::RestoreOpts;

/// Entry point used by the `rupunct` binary. Not part of the library contract.
#[doc(hidden)]
pub fn cli_run() -> i32 {
    cli::run()
}
