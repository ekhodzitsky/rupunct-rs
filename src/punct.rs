use std::path::Path;

use crate::bert::BertForTokenClassification;
use crate::error::{Error, Result};
use crate::limits::MAX_RESTORE_BYTES;
use crate::model::TOKENIZER_FILE;
use crate::restore::{self, RestoreOpts, RestoredWord, join_words};
use crate::wordpiece::Tokenizer;

/// Punctuation and capitalization restorer.
pub struct Punctuator {
    model: BertForTokenClassification,
    tokenizer: Tokenizer,
}

impl Punctuator {
    /// Load from a directory with `model.safetensors`, `tokenizer.json`, `config.json`.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(&model_dir.join(TOKENIZER_FILE))?;
        let model = BertForTokenClassification::load(model_dir)?;
        Ok(Self { model, tokenizer })
    }

    /// Restore punctuation and casing. Empty / whitespace-only input is returned unchanged.
    pub fn restore(&self, text: &str) -> Result<String> {
        self.restore_with(text, RestoreOpts::default())
    }

    /// [`restore`](Self::restore) with explicit window / strip / lowercase knobs.
    pub fn restore_with(&self, text: &str, opts: RestoreOpts) -> Result<String> {
        let words = self.restore_words_with(text, opts)?;
        if words.is_empty() {
            return Ok(text.to_string());
        }
        Ok(join_words(&words))
    }

    /// Per-word restore: surface, label, softmax score of the first subtoken.
    pub fn restore_words(&self, text: &str) -> Result<Vec<RestoredWord>> {
        self.restore_words_with(text, RestoreOpts::default())
    }

    /// [`restore_words`](Self::restore_words) with explicit knobs.
    pub fn restore_words_with(&self, text: &str, opts: RestoreOpts) -> Result<Vec<RestoredWord>> {
        if text.len() > MAX_RESTORE_BYTES {
            return Err(Error::model(format!(
                "input larger than {MAX_RESTORE_BYTES} bytes"
            )));
        }
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Ok(Vec::new());
        }
        restore::restore_words(&self.model, &self.tokenizer, trimmed, opts)
    }

    /// Like [`restore`](Self::restore), but on inference failure returns `text` unchanged.
    pub fn restore_lossy(&self, text: &str) -> String {
        self.restore(text).unwrap_or_else(|_| text.to_string())
    }

    /// Restore each string independently (no padded batch GEMM). First failure aborts.
    pub fn restore_batch(&self, texts: &[&str]) -> Result<Vec<String>> {
        self.restore_batch_with(texts, RestoreOpts::default())
    }

    /// [`restore_batch`](Self::restore_batch) with an explicit token window / overlap.
    pub fn restore_batch_with(&self, texts: &[&str], opts: RestoreOpts) -> Result<Vec<String>> {
        texts.iter().map(|t| self.restore_with(t, opts)).collect()
    }

    #[cfg(test)]
    pub(crate) fn debug_logits(&self, text: &str) -> Result<crate::tensor::Tensor> {
        let words = self.tokenizer.split_words(text);
        let (ids, mask, _) = self.tokenizer.encode_words(&words, true).into_parts();
        let tt = vec![0u32; ids.len()];
        self.model.forward(&ids, &tt, &mask)
    }
}

#[cfg(test)]
#[path = "punct_tests.rs"]
mod tests;
