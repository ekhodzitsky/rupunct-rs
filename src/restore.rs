//! Word-aligned restore with BERT windowing.

use crate::bert::BertForTokenClassification;
use crate::error::{Error, Result};
use crate::labels::{argmax, first_subword_labels, process_token};
use crate::wordpiece::Tokenizer;

/// On-device window cap in tokens (including specials). The checkpoint may
/// allow more (`max_position_embeddings`); 512 keeps attention cheap.
pub(crate) const DEFAULT_WINDOW_TOKENS: usize = 512;

/// Left-context overlap in tokens carried into the next window.
pub(crate) const WINDOW_OVERLAP_TOKENS: usize = 64;

/// Windowing knobs for [`crate::Punctuator::restore_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestoreOpts {
    /// BERT window in tokens, including `[CLS]` / `[SEP]`.
    pub max_window_tokens: usize,
    /// Tokens of left context carried into the next window.
    pub overlap_tokens: usize,
}

impl Default for RestoreOpts {
    fn default() -> Self {
        Self {
            max_window_tokens: DEFAULT_WINDOW_TOKENS,
            overlap_tokens: WINDOW_OVERLAP_TOKENS,
        }
    }
}

impl RestoreOpts {
    pub(crate) fn resolve(self, max_position_embeddings: usize) -> (usize, usize) {
        let max_pos = max_position_embeddings.max(2);
        let window = self.max_window_tokens.clamp(2, max_pos);
        let overlap = self.overlap_tokens.min(window.saturating_sub(1));
        (window, overlap)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Window {
    pub start: usize,
    pub end: usize,
    pub skip: usize,
}

pub(crate) fn restore(
    model: &BertForTokenClassification,
    tokenizer: &Tokenizer,
    text: &str,
    opts: RestoreOpts,
) -> Result<String> {
    let words = tokenizer.split_words(text);
    if words.is_empty() {
        return Ok(text.to_string());
    }
    let (window, overlap) = opts.resolve(model.max_position_embeddings());
    let windows = pack_windows(tokenizer, &words, window, overlap);
    let default_lid = model.default_label();
    let mut labels = Vec::with_capacity(words.len());
    for w in &windows {
        let slice = &words[w.start..w.end];
        let encoding = tokenizer.encode_words(slice, true);
        let mut ids = encoding.get_ids().to_vec();
        let mut mask = encoding.get_attention_mask().to_vec();
        let mut word_ids = encoding.get_word_ids().to_vec();
        fit_seq(&mut ids, &mut mask, &mut word_ids, window);
        let token_type_ids = vec![0u32; ids.len()];
        let logits = model.forward(&ids, &token_type_ids, &mask)?;
        let seq = logits.dim(0)?;
        let nlab = logits.dim(1)?;
        let flat = logits.as_slice();
        let argmax_per_token: Vec<usize> = (0..seq)
            .map(|t| argmax(&flat[t * nlab..(t + 1) * nlab]))
            .collect();
        let window_labels =
            first_subword_labels(&word_ids, &argmax_per_token, slice.len(), default_lid);
        labels.extend(window_labels.into_iter().skip(w.skip));
    }

    if labels.len() != words.len() {
        return Err(Error::model(format!(
            "label count {} != word count {}",
            labels.len(),
            words.len()
        )));
    }

    let mut out = String::new();
    for (word, &lid) in words.iter().zip(labels.iter()) {
        let label = model
            .id2label()
            .get(lid)
            .map(String::as_str)
            .unwrap_or("LOWER_O");
        let processed = process_token(word, label);
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&processed);
    }
    Ok(out.trim().to_string())
}

pub(crate) fn pack_windows(
    tokenizer: &Tokenizer,
    words: &[String],
    max_pos: usize,
    overlap_tokens: usize,
) -> Vec<Window> {
    if words.is_empty() {
        return Vec::new();
    }
    let specials = tokenizer.special_token_count(true);
    let counts: Vec<usize> = words
        .iter()
        .map(|w| tokenizer.word_token_count(w))
        .collect();
    let mut windows = Vec::new();
    let mut committed = 0usize;
    while committed < words.len() {
        let mut start = committed;
        let mut overlap_used = 0usize;
        while start > 0 && overlap_used < overlap_tokens {
            start -= 1;
            overlap_used += counts[start];
        }
        let mut toks = specials;
        for &c in &counts[start..committed] {
            toks = toks.saturating_add(c);
        }
        while start < committed && toks > max_pos {
            toks = toks.saturating_sub(counts[start]);
            start += 1;
        }
        let mut end = committed;
        while end < words.len() {
            let next = counts[end];
            if toks > max_pos || toks + next > max_pos {
                break;
            }
            toks += next;
            end += 1;
        }
        if end <= committed {
            end = (committed + 1).min(words.len());
        }
        let skip = committed.saturating_sub(start);
        windows.push(Window { start, end, skip });
        committed = end;
    }
    windows
}

fn fit_seq(
    ids: &mut Vec<u32>,
    mask: &mut Vec<u32>,
    word_ids: &mut Vec<Option<u32>>,
    max_pos: usize,
) {
    if ids.len() <= max_pos {
        return;
    }
    if ids.len() >= 2 && max_pos >= 2 {
        let sep = ids[ids.len() - 1];
        ids.truncate(max_pos - 1);
        ids.push(sep);
        mask.truncate(max_pos);
        word_ids.truncate(max_pos - 1);
        word_ids.push(None);
    } else {
        ids.truncate(max_pos);
        mask.truncate(max_pos);
        word_ids.truncate(max_pos);
    }
}

#[cfg(test)]
#[path = "restore_tests.rs"]
mod tests;
