use std::path::Path;

use crate::bert::BertForTokenClassification;
use crate::error::{Error, Result};
use crate::limits::MAX_RESTORE_BYTES;
use crate::model::TOKENIZER_FILE;
use crate::restore::{self, RestoreOpts, RestoredWord};
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
        Ok(self.restore_bundle(text, opts)?.0)
    }

    /// Per-word restore: surface, label, softmax score of the first subtoken.
    pub fn restore_words(&self, text: &str) -> Result<Vec<RestoredWord>> {
        self.restore_words_with(text, RestoreOpts::default())
    }

    /// [`restore_words`](Self::restore_words) with explicit knobs.
    pub fn restore_words_with(&self, text: &str, opts: RestoreOpts) -> Result<Vec<RestoredWord>> {
        Ok(self.restore_bundle(text, opts)?.1)
    }

    /// Restored string (original whitespace kept) plus per-word labels.
    pub(crate) fn restore_bundle(
        &self,
        text: &str,
        opts: RestoreOpts,
    ) -> Result<(String, Vec<RestoredWord>)> {
        if text.len() > MAX_RESTORE_BYTES {
            return Err(Error::model(format!(
                "input larger than {MAX_RESTORE_BYTES} bytes"
            )));
        }
        if text.trim().is_empty() {
            return Ok((text.to_string(), Vec::new()));
        }
        let (words, dropped) = restore::restore_parts(&self.model, &self.tokenizer, text, opts)?;
        if words.is_empty() && dropped.is_empty() {
            return Ok((text.to_string(), words));
        }
        let out = restore::render(text, &words, &dropped)?;
        Ok((out, words))
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

    pub(crate) fn max_position_embeddings(&self) -> usize {
        self.model.max_position_embeddings()
    }

    #[cfg(test)]
    pub(crate) fn debug_logits(&self, text: &str) -> Result<crate::tensor::Tensor> {
        let spans = self.tokenizer.split_word_spans(text);
        let (ids, mask, _) = self.tokenizer.encode_words(&spans, true).into_parts();
        let tt = vec![0u32; ids.len()];
        self.model.forward(&ids, &tt, &mask)
    }
}

#[cfg(test)]
#[path = "punct_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "live_parity_tests.rs"]
mod live_parity_tests;
