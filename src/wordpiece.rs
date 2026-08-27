//! In-tree WordPiece tokenizer for RUPunct BERT checkpoints.
//!
//! Supported scheme: `BertNormalizer { clean_text, handle_chinese_chars,
//! strip_accents: false, lowercase: false }` + `BertPreTokenizer` + greedy
//! WordPiece (`##` continuation, `[UNK]` fallback, 100-char word cap) + the
//! `[CLS] A [SEP]` template. Anything outside this scheme fails at load.

use std::collections::HashMap;
use std::path::Path;

use crate::error::{Error, Result};
use crate::tokenizer_json::{
    Normalizer, TokenizerFile, parse_normalizer, parse_post_processor, parse_pre_tokenizer,
};
use crate::unicode::{bert_pre_tokenize, is_chinese_char, is_control};

pub(crate) struct Tokenizer {
    vocab: HashMap<String, u32>,
    unk_id: u32,
    continuing_subword_prefix: String,
    max_input_chars_per_word: usize,
    normalizer: Normalizer,
    bert_pre_tokenizer: bool,
    added_tokens: Vec<AddedToken>,
    cls: Option<u32>,
    sep: Option<u32>,
}

pub(crate) struct Encoding {
    ids: Vec<u32>,
    attention_mask: Vec<u32>,
    word_ids: Vec<Option<u32>>,
}

impl Encoding {
    pub(crate) fn get_ids(&self) -> &[u32] {
        &self.ids
    }

    pub(crate) fn get_attention_mask(&self) -> &[u32] {
        &self.attention_mask
    }

    pub(crate) fn get_word_ids(&self) -> &[Option<u32>] {
        &self.word_ids
    }
}

struct AddedToken {
    id: u32,
    content: String,
    lstrip: bool,
    rstrip: bool,
}

impl Tokenizer {
    pub(crate) fn from_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
        let file: TokenizerFile = serde_json::from_str(&raw).map_err(|e| {
            Error::tokenizer(format!(
                "failed to parse tokenizer file {}: {e}",
                path.display()
            ))
        })?;
        Self::from_parsed(file).map_err(|e| {
            Error::tokenizer(format!(
                "unsupported tokenizer file {}: {e}",
                path.display()
            ))
        })
    }

    fn from_parsed(file: TokenizerFile) -> Result<Self> {
        if let Some(version) = &file.version
            && version != "1.0"
        {
            return Err(Error::tokenizer(format!(
                "unknown tokenizer version '{version}'"
            )));
        }
        if file.truncation.is_some() {
            return Err(Error::tokenizer(
                "truncation is not supported (expected null)",
            ));
        }
        if file.padding.is_some() {
            return Err(Error::tokenizer("padding is not supported (expected null)"));
        }
        if file.model.kind != "WordPiece" {
            return Err(Error::tokenizer(format!(
                "unsupported model type '{}' (expected 'WordPiece')",
                file.model.kind
            )));
        }
        if file.model.vocab.is_empty() {
            return Err(Error::tokenizer("WordPiece vocabulary is empty"));
        }

        let unk_token = file.model.unk_token.unwrap_or_else(|| "[UNK]".to_string());
        let unk_id = *file.model.vocab.get(&unk_token).ok_or_else(|| {
            Error::tokenizer(format!(
                "WordPiece vocabulary is missing unk_token '{unk_token}'"
            ))
        })?;

        let normalizer = parse_normalizer(file.normalizer)?;
        let bert_pre_tokenizer = parse_pre_tokenizer(file.pre_tokenizer)?;
        let (cls, sep) = parse_post_processor(file.post_processor)?;

        let mut added_tokens = Vec::with_capacity(file.added_tokens.len());
        for token in &file.added_tokens {
            if token.content.is_empty() {
                continue;
            }
            if token.normalized {
                return Err(Error::tokenizer(format!(
                    "added token '{}' has normalized=true, which is not supported",
                    token.content
                )));
            }
            if token.single_word {
                return Err(Error::tokenizer(format!(
                    "added token '{}' has single_word=true, which is not supported",
                    token.content
                )));
            }
            let id = file
                .model
                .vocab
                .get(&token.content)
                .copied()
                .unwrap_or(token.id);
            added_tokens.push(AddedToken {
                id,
                content: token.content.clone(),
                lstrip: token.lstrip,
                rstrip: token.rstrip,
            });
        }

        Ok(Self {
            vocab: file.model.vocab,
            unk_id,
            continuing_subword_prefix: file
                .model
                .continuing_subword_prefix
                .unwrap_or_else(|| "##".to_string()),
            max_input_chars_per_word: file.model.max_input_chars_per_word.unwrap_or(100),
            normalizer,
            bert_pre_tokenizer,
            added_tokens,
            cls,
            sep,
        })
    }

    pub(crate) fn encode(&self, text: &str, add_special_tokens: bool) -> Encoding {
        let mut ids: Vec<u32> = Vec::new();
        let mut word_ids: Vec<Option<u32>> = Vec::new();

        if add_special_tokens && let Some(cls) = self.cls {
            ids.push(cls);
            word_ids.push(None);
        }

        let mut next_word: u32 = 0;
        for span in self.extract_added(text) {
            match span {
                Span::Added(idx) => {
                    ids.push(self.added_tokens[idx].id);
                    word_ids.push(Some(next_word));
                    next_word += 1;
                }
                Span::Text(slice) => {
                    let normalized = self.normalize(slice);
                    let words = if self.bert_pre_tokenizer {
                        bert_pre_tokenize(&normalized)
                    } else if normalized.is_empty() {
                        Vec::new()
                    } else {
                        vec![normalized.as_str()]
                    };
                    for word in words {
                        self.tokenize_word(word, &mut ids);
                        word_ids.resize(ids.len(), Some(next_word));
                        next_word += 1;
                    }
                }
            }
        }

        if add_special_tokens && let Some(sep) = self.sep {
            ids.push(sep);
            word_ids.push(None);
        }

        Encoding {
            attention_mask: vec![1; ids.len()],
            ids,
            word_ids,
        }
    }

    /// Surface words after added-token extraction, normalization, and pre-tokenization.
    pub(crate) fn split_words(&self, text: &str) -> Vec<String> {
        let mut words = Vec::new();
        for span in self.extract_added(text) {
            match span {
                Span::Added(idx) => words.push(self.added_tokens[idx].content.clone()),
                Span::Text(slice) => {
                    let normalized = self.normalize(slice);
                    if self.bert_pre_tokenizer {
                        for w in bert_pre_tokenize(&normalized) {
                            words.push(w.to_string());
                        }
                    } else if !normalized.is_empty() {
                        words.push(normalized);
                    }
                }
            }
        }
        words
    }

    pub(crate) fn encode_words(&self, words: &[String], add_special_tokens: bool) -> Encoding {
        let mut ids: Vec<u32> = Vec::new();
        let mut word_ids: Vec<Option<u32>> = Vec::new();

        if add_special_tokens && let Some(cls) = self.cls {
            ids.push(cls);
            word_ids.push(None);
        }

        for (i, word) in words.iter().enumerate() {
            self.tokenize_word(word, &mut ids);
            word_ids.resize(ids.len(), Some(i as u32));
        }

        if add_special_tokens && let Some(sep) = self.sep {
            ids.push(sep);
            word_ids.push(None);
        }

        let attention_mask = vec![1; ids.len()];
        Encoding {
            ids,
            attention_mask,
            word_ids,
        }
    }

    fn normalize(&self, text: &str) -> String {
        let (clean_text, handle_chinese_chars) = match &self.normalizer {
            Normalizer::None => return text.to_string(),
            Normalizer::Bert {
                clean_text,
                handle_chinese_chars,
            } => (*clean_text, *handle_chinese_chars),
        };

        let mut out = String::with_capacity(text.len());
        for c in text.chars() {
            if clean_text {
                if c == '\0' || c == '\u{FFFD}' || is_control(c) {
                    continue;
                }
                out.push(if c.is_whitespace() { ' ' } else { c });
            } else {
                out.push(c);
            }
        }
        if handle_chinese_chars {
            let mut spaced = String::with_capacity(out.len());
            for c in out.chars() {
                if is_chinese_char(c) {
                    spaced.push(' ');
                    spaced.push(c);
                    spaced.push(' ');
                } else {
                    spaced.push(c);
                }
            }
            out = spaced;
        }
        out
    }

    fn tokenize_word(&self, word: &str, out: &mut Vec<u32>) {
        if word.chars().count() > self.max_input_chars_per_word {
            out.push(self.unk_id);
            return;
        }

        let mut start = 0;
        let mut is_bad = false;
        let checkpoint = out.len();
        while start < word.len() {
            let mut end = word.len();
            let mut found = None;
            while start < end {
                let substr = &word[start..end];
                let id = if start > 0 {
                    let mut candidate =
                        String::with_capacity(self.continuing_subword_prefix.len() + substr.len());
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

    fn extract_added<'a>(&self, text: &'a str) -> Vec<Span<'a>> {
        let mut spans = Vec::new();
        if self.added_tokens.is_empty() {
            if !text.is_empty() {
                spans.push(Span::Text(text));
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
                spans.push(Span::Text(&text[span_start..start]));
            }
            spans.push(Span::Added(idx));
            pos = stop;
            span_start = stop;
        }
        if span_start < text.len() {
            spans.push(Span::Text(&text[span_start..]));
        }
        spans
    }
}

enum Span<'a> {
    Text(&'a str),
    Added(usize),
}

#[cfg(test)]
#[path = "wordpiece_tests.rs"]
mod tests;
