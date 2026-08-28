//! Word-aligned token windows for long restore.

use crate::wordpiece::Tokenizer;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Window {
    pub start: usize,
    pub end: usize,
    pub skip: usize,
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
        // Drop left context until specials + overlap + the next new word fit.
        while start < committed {
            let next = counts[committed];
            if toks.saturating_add(next) <= max_pos {
                break;
            }
            toks = toks.saturating_sub(counts[start]);
            start += 1;
        }
        let mut end = committed;
        while end < words.len() {
            let next = counts[end];
            if toks > max_pos || toks.saturating_add(next) > max_pos {
                break;
            }
            toks = toks.saturating_add(next);
            end += 1;
        }
        if end <= committed {
            // One word still does not fit: encode it alone so fit_seq keeps
            // its first subtokens instead of truncating them behind overlap.
            start = committed;
            end = (committed + 1).min(words.len());
        }
        let skip = committed.saturating_sub(start);
        windows.push(Window { start, end, skip });
        committed = end;
    }
    windows
}

pub(crate) fn fit_seq(
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
