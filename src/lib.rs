//! On-device Russian punctuation and casing restoration.
//!
//! BERT token-classification (RUPunct_small). Extracted from gigastt-pro.
//! Weights are not in git: `model.safetensors` + `tokenizer.json` + `config.json`.

#![allow(dead_code)] // vendored F32 runtime is a slice of gigastt-tensor

mod bert;
mod blas;
mod device;
mod dtype;
mod error;
mod labels;
mod layer_norm;
mod linear;
mod matmul;
mod shape;
mod tensor;
mod weights;
mod wordpiece;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub use labels::process_token;

/// Punctuation + capitalization restorer.
pub struct Punctuator {
    inner: bert::TensorPunctuator,
}

impl Punctuator {
    /// Load from a directory with `model.safetensors`, `tokenizer.json`, `config.json`.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let p = bert::TensorPunctuator::load(model_dir).context("punct load")?;
        Ok(Self { inner: p })
    }

    /// Restore punctuation and casing. Never panics: on failure returns `text` unchanged.
    pub fn restore(&self, text: &str) -> String {
        self.inner.restore(text)
    }
}

/// `RUPUNCT_MODEL`, else `~/.rupunct/punct` if present, else `~/.gigastt/models/punct`.
pub fn default_model_dir() -> PathBuf {
    if let Ok(p) = std::env::var("RUPUNCT_MODEL") {
        let t = p.trim();
        if !t.is_empty() {
            return PathBuf::from(t);
        }
    }
    let home = std::env::var_os("HOME").map(PathBuf::from);
    if let Some(home) = home {
        let native = home.join(".rupunct").join("punct");
        if native.join("model.safetensors").is_file() {
            return native;
        }
        let gigastt = home.join(".gigastt").join("models").join("punct");
        if gigastt.join("model.safetensors").is_file() {
            return gigastt;
        }
        return native;
    }
    PathBuf::from("punct")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::labels::{PUNCT_CONFIG_FILE, PUNCT_TOKENIZER_FILE};

    const MINIMAL_TOKENIZER_JSON: &str = r###"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordPiece",
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "vocab": {"[UNK]": 0, "a": 1, "b": 2}
        }
    }"###;

    const MINIMAL_CONFIG_JSON: &str = r#"{"id2label": {"0": "LOWER_O"}, "hidden_size": 2, "num_attention_heads": 1, "num_hidden_layers": 1, "intermediate_size": 2, "vocab_size": 3, "max_position_embeddings": 8, "type_vocab_size": 2, "layer_norm_eps": 1e-12, "num_labels": 1}"#;

    #[test]
    fn test_load_missing_dir_errors() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let missing = tmp.path().join("does-not-exist");
        assert!(Punctuator::load(&missing).is_err());
    }

    #[test]
    fn test_load_valid_config_invalid_tokenizer_errors() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(tmp.path().join(PUNCT_CONFIG_FILE), MINIMAL_CONFIG_JSON).unwrap();
        std::fs::write(tmp.path().join(PUNCT_TOKENIZER_FILE), "{ not valid json").unwrap();
        match Punctuator::load(tmp.path()) {
            Ok(_) => panic!("malformed tokenizer must error"),
            Err(e) => {
                let full = format!("{e:#}");
                assert!(
                    full.contains("tokenizer"),
                    "expected tokenizer in error chain, got {full}"
                );
            }
        }
    }

    #[test]
    fn test_load_valid_config_and_tokenizer_missing_model_errors() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(tmp.path().join(PUNCT_CONFIG_FILE), MINIMAL_CONFIG_JSON).unwrap();
        std::fs::write(
            tmp.path().join(PUNCT_TOKENIZER_FILE),
            MINIMAL_TOKENIZER_JSON,
        )
        .unwrap();
        assert!(Punctuator::load(tmp.path()).is_err());
    }

    #[test]
    #[ignore = "requires punct model"]
    fn test_restore_reference_string() {
        let dir = default_model_dir();
        let punct = Punctuator::load(&dir).expect("load punct model");
        let out =
            punct.restore("привет меня зовут анна сколько будет стоить шестьдесят тысяч тенге");
        assert_eq!(
            out,
            "Привет меня зовут Анна, Сколько будет стоить шестьдесят тысяч тенге."
        );
    }
}
