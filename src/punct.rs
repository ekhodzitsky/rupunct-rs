use std::path::Path;

use crate::bert::BertForTokenClassification;
use crate::error::Result;
use crate::model::TOKENIZER_FILE;
use crate::restore;
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
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Ok(text.to_string());
        }
        restore::restore(&self.model, &self.tokenizer, trimmed)
    }

    /// Like [`restore`](Self::restore), but on inference failure returns `text` unchanged.
    pub fn restore_lossy(&self, text: &str) -> String {
        self.restore(text).unwrap_or_else(|_| text.to_string())
    }

    /// Restore each string independently. First failure aborts the batch.
    pub fn restore_batch(&self, texts: &[&str]) -> Result<Vec<String>> {
        texts.iter().map(|t| self.restore(t)).collect()
    }
}

#[cfg(test)]
#[path = "punct_tests.rs"]
mod tests;
