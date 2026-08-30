//! Word-aligned restore with BERT windowing.

mod splice;
mod window;

use crate::bert::BertForTokenClassification;
use crate::error::{Error, Result};
use crate::labels::{WordPred, argmax_softmax, first_subword_preds, process_token};
use crate::limits::{MAX_RESTORE_BYTES, MIN_POSITIONS};
use crate::unicode::is_bert_punc;
use crate::wordpiece::{Tokenizer, WordSpan};
use splice::{Edit, splice};
pub(crate) use window::Window;
use window::{fit_seq, pack_windows};

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
    /// Parallel BERT windows. `0` uses [`std::thread::available_parallelism`].
    pub threads: usize,
}

impl Default for RestoreOpts {
    fn default() -> Self {
        Self {
            max_window_tokens: DEFAULT_WINDOW_TOKENS,
            overlap_tokens: WINDOW_OVERLAP_TOKENS,
            strip_punct: false,
            lowercase: false,
            threads: 0,
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
    /// Byte offset of this word in the original restore input (inclusive).
    pub start: usize,
    /// Byte offset of this word in the original restore input (exclusive).
    pub end: usize,
}

pub(crate) fn render(
    original: &str,
    words: &[RestoredWord],
    dropped: &[(usize, usize)],
) -> Result<String> {
    let mut edits: Vec<Edit> = Vec::with_capacity(words.len() + dropped.len());
    for w in words {
        edits.push(Edit {
            start: w.start,
            end: w.end,
            text: w.text.clone(),
        });
    }
    for &(start, end) in dropped {
        edits.push(Edit {
            start,
            end,
            text: String::new(),
        });
    }
    splice(original, &mut edits)
}

pub(crate) type DroppedSpans = Vec<(usize, usize)>;

pub(crate) fn restore_parts(
    model: &BertForTokenClassification,
    tokenizer: &Tokenizer,
    text: &str,
    opts: RestoreOpts,
) -> Result<(Vec<RestoredWord>, DroppedSpans)> {
    if text.len() > MAX_RESTORE_BYTES {
        return Err(Error::model(format!(
            "input larger than {MAX_RESTORE_BYTES} bytes"
        )));
    }
    let mut spans = tokenizer.split_word_spans(text);
    let dropped = if opts.strip_punct {
        let (kept, drop) = strip_isolated_punct(spans);
        spans = kept;
        drop
    } else {
        Vec::new()
    };
    if opts.lowercase {
        for w in &mut spans {
            w.text = w.text.to_lowercase();
        }
    }
    if spans.is_empty() {
        return Ok((Vec::new(), dropped));
    }
    let (window, overlap) = opts.resolve(model.max_position_embeddings())?;
    let windows = pack_windows(tokenizer, &spans, window, overlap);
    let default_lid = model.default_label();
    let missing = WordPred {
        label: default_lid,
        score: 0.0,
    };
    let jobs = crate::par::worker_count(opts.threads, windows.len());
    let window_preds = run_windows(
        model,
        tokenizer,
        &spans,
        &windows,
        window,
        default_lid,
        jobs,
    )?;
    let mut best = vec![None; spans.len()];
    for (w, preds) in windows.iter().zip(window_preds) {
        for (i, pred) in preds.into_iter().enumerate() {
            let gi = w.start + i;
            if gi >= best.len() {
                continue;
            }
            merge_pred(&mut best[gi], pred);
        }
    }

    let id2 = model.id2label();
    let mut out = Vec::with_capacity(spans.len());
    for (span, slot) in spans.into_iter().zip(best.into_iter()) {
        let pred = slot.unwrap_or(missing);
        let label = id2.get(pred.label).map(String::as_str).unwrap_or("LOWER_O");
        let rendered = process_token(&span.text, label);
        out.push(RestoredWord {
            word: span.text,
            text: rendered,
            label: label.to_string(),
            score: pred.score,
            start: span.start,
            end: span.end,
        });
    }
    Ok((out, dropped))
}

fn infer_window(
    model: &BertForTokenClassification,
    tokenizer: &Tokenizer,
    spans: &[WordSpan],
    w: &Window,
    max_pos: usize,
    default_lid: usize,
) -> Result<Vec<WordPred>> {
    let slice = &spans[w.start..w.end];
    let (mut ids, mut mask, mut word_ids) = tokenizer.encode_words(slice, true).into_parts();
    fit_seq(&mut ids, &mut mask, &mut word_ids, max_pos);
    let token_type_ids = vec![0u32; ids.len()];
    let logits = model.forward(&ids, &token_type_ids, &mask)?;
    let seq = logits.dim(0)?;
    let nlab = logits.dim(1)?;
    let flat = logits.as_slice();
    let per_token: Vec<_> = (0..seq)
        .map(|t| argmax_softmax(&flat[t * nlab..(t + 1) * nlab]))
        .collect();
    Ok(first_subword_preds(
        &word_ids,
        &per_token,
        slice.len(),
        default_lid,
    ))
}

fn run_windows(
    model: &BertForTokenClassification,
    tokenizer: &Tokenizer,
    spans: &[WordSpan],
    windows: &[Window],
    max_pos: usize,
    default_lid: usize,
    jobs: usize,
) -> Result<Vec<Vec<WordPred>>> {
    crate::par::parallel_map(windows, jobs, |w| {
        infer_window(model, tokenizer, spans, w, max_pos, default_lid)
    })
}

fn merge_pred(slot: &mut Option<WordPred>, pred: WordPred) {
    match slot {
        None => *slot = Some(pred),
        Some(old) if pred.score > old.score => *slot = Some(pred),
        _ => {}
    }
}

fn is_punct_word(w: &str) -> bool {
    !w.is_empty() && w.chars().all(is_bert_punc)
}

fn is_ascii_digits(w: &str) -> bool {
    !w.is_empty() && w.bytes().all(|b| b.is_ascii_digit())
}

fn strip_isolated_punct(spans: Vec<WordSpan>) -> (Vec<WordSpan>, Vec<(usize, usize)>) {
    let drop: Vec<bool> = (0..spans.len()).map(|i| drop_punct_at(&spans, i)).collect();
    let mut kept = Vec::new();
    let mut dropped = Vec::new();
    for (s, d) in spans.into_iter().zip(drop) {
        if d {
            dropped.push((s.start, s.end));
        } else {
            kept.push(s);
        }
    }
    (kept, dropped)
}

fn drop_punct_at(spans: &[WordSpan], i: usize) -> bool {
    let w = &spans[i].text;
    if !is_punct_word(w) {
        return false;
    }
    let left = i.checked_sub(1).and_then(|j| spans.get(j));
    let right = spans.get(i + 1);
    let between = matches!(
        (left, right),
        (Some(l), Some(r)) if !is_punct_word(&l.text) && !is_punct_word(&r.text)
    );
    if !between {
        return true;
    }
    if w == "-" {
        return false;
    }
    if w == "."
        && left.is_some_and(|l| is_ascii_digits(&l.text))
        && right.is_some_and(|r| is_ascii_digits(&r.text))
    {
        return false;
    }
    true
}

#[cfg(test)]
#[path = "restore_tests.rs"]
mod tests;
