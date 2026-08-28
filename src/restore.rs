//! Word-aligned restore with BERT windowing.

mod window;

use crate::bert::BertForTokenClassification;
use crate::error::{Error, Result};
use crate::labels::{argmax_softmax, first_subword_preds, process_token};
use crate::limits::{MAX_RESTORE_BYTES, MIN_POSITIONS};
use crate::unicode::is_bert_punc;
use crate::wordpiece::Tokenizer;
use window::{fit_seq, pack_windows};

#[cfg(test)]
pub(crate) use window::Window;

/// On-device window cap in tokens (including specials). The checkpoint may
/// allow more (`max_position_embeddings`); 512 keeps attention cheap.
pub(crate) const DEFAULT_WINDOW_TOKENS: usize = 512;

/// Left-context overlap in tokens carried into the next window.
pub(crate) const WINDOW_OVERLAP_TOKENS: usize = 64;

/// Windowing and input knobs for [`crate::Punctuator::restore_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestoreOpts {
    /// BERT window in tokens, including `[CLS]` / `[SEP]`.
    pub max_window_tokens: usize,
    /// Tokens of left context carried into the next window.
    pub overlap_tokens: usize,
    /// Drop isolated punctuation words before the encoder.
    pub strip_punct: bool,
    /// Lowercase words before the encoder (ASR-style).
    pub lowercase: bool,
}

impl Default for RestoreOpts {
    fn default() -> Self {
        Self {
            max_window_tokens: DEFAULT_WINDOW_TOKENS,
            overlap_tokens: WINDOW_OVERLAP_TOKENS,
            strip_punct: false,
            lowercase: false,
        }
    }
}

impl RestoreOpts {
    pub(crate) fn resolve(self, max_position_embeddings: usize) -> Result<(usize, usize)> {
        if max_position_embeddings < MIN_POSITIONS {
            return Err(Error::config(format!(
                "max_position_embeddings {max_position_embeddings} must be >= {MIN_POSITIONS}"
            )));
        }
        let window = self
            .max_window_tokens
            .clamp(MIN_POSITIONS, max_position_embeddings);
        let overlap = self.overlap_tokens.min(window.saturating_sub(1));
        Ok((window, overlap))
    }
}

/// One restored word: classifier input, rendered form, label, softmax score.
#[derive(Debug, Clone, PartialEq)]
pub struct RestoredWord {
    /// Word after strip/lowercase — the classifier input.
    pub word: String,
    /// [`process_token`] rendering.
    pub text: String,
    /// RUPunct `id2label` string.
    pub label: String,
    /// Softmax of the argmax logit on the first subtoken. Not calibrated.
    pub score: f32,
}

pub(crate) fn join_words(words: &[RestoredWord]) -> String {
    let mut out = String::new();
    for w in words {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&w.text);
    }
    out.trim().to_string()
}

pub(crate) fn restore_words(
    model: &BertForTokenClassification,
    tokenizer: &Tokenizer,
    text: &str,
    opts: RestoreOpts,
) -> Result<Vec<RestoredWord>> {
    if text.len() > MAX_RESTORE_BYTES {
        return Err(Error::model(format!(
            "input larger than {MAX_RESTORE_BYTES} bytes"
        )));
    }
    let mut words = tokenizer.split_words(text);
    if opts.strip_punct {
        words.retain(|w| !is_punct_word(w));
    }
    if opts.lowercase {
        for w in &mut words {
            *w = w.to_lowercase();
        }
    }
    if words.is_empty() {
        return Ok(Vec::new());
    }
    let (window, overlap) = opts.resolve(model.max_position_embeddings())?;
    let windows = pack_windows(tokenizer, &words, window, overlap);
    let default_lid = model.default_label();
    let mut preds = Vec::with_capacity(words.len());
    for w in &windows {
        let slice = &words[w.start..w.end];
        let (mut ids, mut mask, mut word_ids) = tokenizer.encode_words(slice, true).into_parts();
        fit_seq(&mut ids, &mut mask, &mut word_ids, window);
        let token_type_ids = vec![0u32; ids.len()];
        let logits = model.forward(&ids, &token_type_ids, &mask)?;
        let seq = logits.dim(0)?;
        let nlab = logits.dim(1)?;
        let flat = logits.as_slice();
        let per_token: Vec<_> = (0..seq)
            .map(|t| argmax_softmax(&flat[t * nlab..(t + 1) * nlab]))
            .collect();
        let window_preds = first_subword_preds(&word_ids, &per_token, slice.len(), default_lid);
        preds.extend(window_preds.into_iter().skip(w.skip));
    }

    if preds.len() != words.len() {
        return Err(Error::model(format!(
            "label count {} != word count {}",
            preds.len(),
            words.len()
        )));
    }

    let id2 = model.id2label();
    let mut out = Vec::with_capacity(words.len());
    for (word, pred) in words.into_iter().zip(preds.iter()) {
        let label = id2.get(pred.label).map(String::as_str).unwrap_or("LOWER_O");
        let text = process_token(&word, label);
        out.push(RestoredWord {
            word,
            text,
            label: label.to_string(),
            score: pred.score,
        });
    }
    Ok(out)
}

fn is_punct_word(w: &str) -> bool {
    !w.is_empty() && w.chars().all(is_bert_punc)
}

#[cfg(test)]
#[path = "restore_tests.rs"]
mod tests;
