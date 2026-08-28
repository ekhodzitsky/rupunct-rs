//! Normalizer, greedy WordPiece, and added-token scan for [`super::Tokenizer`].

use crate::tokenizer_json::Normalizer;
use crate::unicode::{is_chinese_char, is_control};
use crate::wordpiece::Tokenizer;

pub(super) struct AddedToken {
    pub id: u32,
    pub content: String,
    pub lstrip: bool,
    pub rstrip: bool,
}

pub(super) enum Span<'a> {
    Text {
        text: &'a str,
        start: usize,
    },
    Added {
        idx: usize,
        start: usize,
        end: usize,
    },
}

impl Tokenizer {
    pub(super) fn push_word(&self, word: &str, out: &mut Vec<u32>) {
        if let Some(id) = self.added_id(word) {
            out.push(id);
            return;
        }
        self.tokenize_word(word, out);
    }

    fn added_id(&self, word: &str) -> Option<u32> {
        self.added_tokens
            .iter()
            .find(|t| t.content == word)
            .map(|t| t.id)
    }

    /// Normalize `text` (a slice of the original input starting at `orig_start`).
    /// The map has one `(orig_start, orig_end)` per byte of the returned string.
    pub(super) fn normalize_mapped(
        &self,
        text: &str,
        orig_start: usize,
    ) -> (String, Vec<(usize, usize)>) {
        let (clean_text, handle_chinese_chars, lowercase) = match &self.normalizer {
            Normalizer::None => {
                return (text.to_string(), identity_byte_map(text, orig_start));
            }
            Normalizer::Bert {
                clean_text,
                handle_chinese_chars,
                lowercase,
            } => (*clean_text, *handle_chinese_chars, *lowercase),
        };

        let mut out = String::with_capacity(text.len());
        let mut map = Vec::with_capacity(text.len());
        let mut i = orig_start;
        for c in text.chars() {
            let e = i + c.len_utf8();
            let span = (i, e);
            i = e;
            if clean_text {
                if c == '\0' || c == '\u{FFFD}' || is_control(c) {
                    continue;
                }
                let o = if c.is_whitespace() { ' ' } else { c };
                push_mapped(&mut out, &mut map, o, span);
            } else {
                push_mapped(&mut out, &mut map, c, span);
            }
        }
        if handle_chinese_chars {
            let mut spaced = String::with_capacity(out.len());
            let mut smap = Vec::with_capacity(map.len());
            let mut b = 0usize;
            for c in out.chars() {
                let n = c.len_utf8();
                let span = map[b];
                if is_chinese_char(c) {
                    push_mapped(&mut spaced, &mut smap, ' ', span);
                    push_mapped(&mut spaced, &mut smap, c, span);
                    push_mapped(&mut spaced, &mut smap, ' ', span);
                } else {
                    spaced.push(c);
                    smap.extend_from_slice(&map[b..b + n]);
                }
                b += n;
            }
            out = spaced;
            map = smap;
        }
        if lowercase {
            let mut low = String::with_capacity(out.len());
            let mut lmap = Vec::with_capacity(map.len());
            let mut b = 0usize;
            for c in out.chars() {
                let n = c.len_utf8();
                let span = map[b];
                for lc in c.to_lowercase() {
                    push_mapped(&mut low, &mut lmap, lc, span);
                }
                b += n;
            }
            out = low;
            map = lmap;
        }
        (out, map)
    }

    pub(super) fn tokenize_word(&self, word: &str, out: &mut Vec<u32>) {
        if word.chars().count() > self.max_input_chars_per_word {
            out.push(self.unk_id);
            return;
        }

        let mut start = 0;
        let mut is_bad = false;
        let checkpoint = out.len();
        let mut candidate = String::new();
        while start < word.len() {
            let mut end = word.len();
            let mut found = None;
            while start < end {
                let substr = &word[start..end];
                let id = if start > 0 {
                    candidate.clear();
                    candidate.push_str(&self.continuing_subword_prefix);
                    candidate.push_str(substr);
                    self.vocab.get(candidate.as_str()).copied()
                } else {
                    self.vocab.get(substr).copied()
                };
                if let Some(id) = id {
                    found = Some(id);
                    break;
                }
                end -= substr.chars().last().map_or(1, |c| c.len_utf8());
            }
            let Some(id) = found else {
                is_bad = true;
                break;
            };
            out.push(id);
            start = end;
        }

        if is_bad {
            out.truncate(checkpoint);
            out.push(self.unk_id);
        }
    }

    pub(super) fn extract_added<'a>(&self, text: &'a str) -> Vec<Span<'a>> {
        let mut spans = Vec::new();
        if self.added_tokens.is_empty() {
            if !text.is_empty() {
                spans.push(Span::Text { text, start: 0 });
            }
            return spans;
        }

        let mut span_start = 0;
        let mut pos = 0;
        while pos < text.len() {
            let mut best: Option<usize> = None;
            for (i, token) in self.added_tokens.iter().enumerate() {
                if text[pos..].starts_with(token.content.as_str()) {
                    match best {
                        Some(b) if self.added_tokens[b].content.len() >= token.content.len() => {}
                        _ => best = Some(i),
                    }
                }
            }
            let Some(idx) = best else {
                pos += text[pos..].chars().next().map_or(1, |c| c.len_utf8());
                continue;
            };

            let token = &self.added_tokens[idx];
            let mut start = pos;
            let mut stop = pos + token.content.len();
            if token.lstrip {
                while start > span_start {
                    let prev = text[..start].chars().next_back();
                    match prev {
                        Some(c) if c.is_whitespace() => start -= c.len_utf8(),
                        _ => break,
                    }
                }
            }
            if token.rstrip {
                while let Some(c) = text[stop..].chars().next() {
                    if !c.is_whitespace() {
                        break;
                    }
                    stop += c.len_utf8();
                }
            }

            if start > span_start {
                spans.push(Span::Text {
                    text: &text[span_start..start],
                    start: span_start,
                });
            }
            spans.push(Span::Added {
                idx,
                start,
                end: stop,
            });
            pos = stop;
            span_start = stop;
        }
        if span_start < text.len() {
            spans.push(Span::Text {
                text: &text[span_start..],
                start: span_start,
            });
        }
        spans
    }
}

fn identity_byte_map(text: &str, orig_start: usize) -> Vec<(usize, usize)> {
    let mut map = Vec::with_capacity(text.len());
    let mut i = orig_start;
    for c in text.chars() {
        let e = i + c.len_utf8();
        for _ in 0..c.len_utf8() {
            map.push((i, e));
        }
        i = e;
    }
    map
}

fn push_mapped(out: &mut String, map: &mut Vec<(usize, usize)>, c: char, span: (usize, usize)) {
    out.push(c);
    for _ in 0..c.len_utf8() {
        map.push(span);
    }
}
