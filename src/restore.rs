//! Word-aligned restore with BERT windowing.

use crate::bert::BertForTokenClassification;
use crate::error::Result;
use crate::labels::{argmax, first_subword_labels, process_token};
use crate::wordpiece::Tokenizer;

/// Overlap words carried into the next window for left context.
pub(crate) const WINDOW_OVERLAP_WORDS: usize = 8;

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
) -> Result<String> {
    let words = tokenizer.split_words(text);
    if words.is_empty() {
        return Ok(text.to_string());
    }
    let max_pos = model.max_position_embeddings().max(2);
    let windows = pack_windows(tokenizer, &words, max_pos, WINDOW_OVERLAP_WORDS);
    let single = windows.len() == 1;
    let mut labels = Vec::with_capacity(words.len());
    for w in &windows {
        let slice = &words[w.start..w.end];
        let encoding = if single {
            tokenizer.encode(text, true)
        } else {
            tokenizer.encode_words(slice, true)
        };
        let mut ids = encoding.get_ids().to_vec();
        let mut mask = encoding.get_attention_mask().to_vec();
        let mut word_ids = encoding.get_word_ids().to_vec();
        fit_seq(&mut ids, &mut mask, &mut word_ids, max_pos);
        let token_type_ids = vec![0u32; ids.len()];
        let logits = model.forward(&ids, &token_type_ids, &mask)?;
        let seq = logits.dim(0)?;
        let nlab = logits.dim(1)?;
        let flat = logits.as_slice();
        let argmax_per_token: Vec<usize> = (0..seq)
            .map(|t| argmax(&flat[t * nlab..(t + 1) * nlab]))
            .collect();
        let window_labels = first_subword_labels(&word_ids, &argmax_per_token, slice.len());
        labels.extend(window_labels.into_iter().skip(w.skip));
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
    overlap: usize,
) -> Vec<Window> {
    let mut windows = Vec::new();
    let mut committed = 0usize;
    while committed < words.len() {
        let mut start = committed.saturating_sub(overlap);
        while start < committed
            && tokenizer
                .encode_words(&words[start..=committed], true)
                .get_ids()
                .len()
                > max_pos
        {
            start += 1;
        }
        let mut end = (committed + 1).min(words.len());
        while end < words.len()
            && tokenizer
                .encode_words(&words[start..=end], true)
                .get_ids()
                .len()
                <= max_pos
        {
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
