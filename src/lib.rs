//! On-device Russian punctuation and casing restoration.
//!
//! One checkpoint resident. No network. Default weights:
//! [RUPunct/RUPunct_big](https://huggingface.co/RUPunct/RUPunct_big)
//! (`aggregation_strategy="first"` on each word, not HuggingFace entity grouping).
//! Also loads [RUPunct_small](https://huggingface.co/RUPunct/RUPunct_small).

mod bert;
mod blas;
mod cli;
mod config;
mod encoder;
mod erf;
mod error;
mod gelu;
mod labels;
mod layer_norm;
mod limits;
mod linear;
mod matmul;
mod model;
mod par;
mod punct;
mod restore;
mod shape;
mod st;
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
pub use model::{
    CONFIG_FILE, DEFAULT_CHECKPOINT_DIR, MODEL_FILE, TOKENIZER_FILE, default_model_dir,
};
pub use punct::Punctuator;
pub use restore::{RestoreOpts, RestoredWord};

/// Entry point used by the `rupunct` binary. Not part of the library contract.
#[doc(hidden)]
pub fn cli_run() -> i32 {
    cli::run()
}
