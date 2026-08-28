//! In-tree WordPiece tokenizer for RUPunct BERT checkpoints.
//!
//! Supported scheme: `BertNormalizer { clean_text, handle_chinese_chars,
//! strip_accents: false, lowercase }` + `BertPreTokenizer` + greedy
//! WordPiece (`##` continuation, `[UNK]` fallback, 100-char word cap) + the
//! `[CLS] A [SEP]` template. Anything outside this scheme fails at load.

use std::collections::HashMap;
use std::path::Path;

use crate::error::{Error, Result};
use crate::limits::{MAX_ADDED_TOKENS, MAX_TOKENIZER_BYTES, MAX_WORD_CHARS, read_capped_text};
use crate::tokenizer_json::{
    Normalizer, TokenizerFile, parse_normalizer, parse_post_processor, parse_pre_tokenizer,
};
use crate::unicode::bert_pre_tokenize_spans;

mod scan;
use scan::{AddedToken, Span};

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WordSpan {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

pub(crate) struct Encoding {
    ids: Vec<u32>,
    attention_mask: Vec<u32>,
    word_ids: Vec<Option<u32>>,
}

impl Encoding {
    #[cfg(test)]
    pub(crate) fn get_ids(&self) -> &[u32] {
        &self.ids
    }

    #[cfg(test)]
    pub(crate) fn get_attention_mask(&self) -> &[u32] {
        &self.attention_mask
    }

    #[cfg(test)]
    pub(crate) fn get_word_ids(&self) -> &[Option<u32>] {
        &self.word_ids
    }

    pub(crate) fn into_parts(self) -> (Vec<u32>, Vec<u32>, Vec<Option<u32>>) {
        (self.ids, self.attention_mask, self.word_ids)
    }
}

impl Tokenizer {
    pub(crate) fn from_file(path: &Path) -> Result<Self> {
        let raw = match read_capped_text(path, MAX_TOKENIZER_BYTES) {
            Err(Error::Config(m)) => return Err(Error::tokenizer(m)),
            other => other?,
        };
        let value: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
            Error::tokenizer(format!(
                "failed to parse tokenizer file {}: {e}",
                path.display()
            ))
        })?;
        let file = TokenizerFile::from_value(value)
            .map_err(|e| Error::tokenizer(format!("{}: {e}", path.display())))?;
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

        if file.added_tokens.len() > MAX_ADDED_TOKENS {
            return Err(Error::tokenizer(format!(
                "too many added_tokens ({}, max {MAX_ADDED_TOKENS})",
                file.added_tokens.len()
            )));
        }
        let max_chars = file
            .model
            .max_input_chars_per_word
            .unwrap_or(100)
            .min(MAX_WORD_CHARS);
        if max_chars == 0 {
            return Err(Error::tokenizer("max_input_chars_per_word must be > 0"));
        }
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
            max_input_chars_per_word: max_chars,
            normalizer,
            bert_pre_tokenizer,
            added_tokens,
            cls,
            sep,
        })
    }

    #[cfg(test)]
    pub(crate) fn encode(&self, text: &str, add_special_tokens: bool) -> Encoding {
        self.encode_words(&self.split_words(text), add_special_tokens)
    }

    /// Surface words after added-token extraction, normalization, and pre-tokenization.
    #[cfg(test)]
    pub(crate) fn split_words(&self, text: &str) -> Vec<String> {
        self.split_word_spans(text)
            .into_iter()
            .map(|w| w.text)
            .collect()
    }

    /// Like [`split_words`](Self::split_words) with byte offsets into `text`.
    pub(crate) fn split_word_spans(&self, text: &str) -> Vec<WordSpan> {
        let mut words = Vec::new();
        for span in self.extract_added(text) {
            match span {
                Span::Added { idx, start, end } => words.push(WordSpan {
                    text: self.added_tokens[idx].content.clone(),
                    start,
                    end,
                }),
                Span::Text {
                    text: slice,
                    start: orig,
                } => {
                    let (normalized, byte_map) = self.normalize_mapped(slice, orig);
                    if self.bert_pre_tokenizer {
                        for (a, b) in bert_pre_tokenize_spans(&normalized) {
                            if b == 0 || a >= byte_map.len() {
                                continue;
                            }
                            words.push(WordSpan {
                                text: normalized[a..b].to_string(),
                                start: byte_map[a].0,
                                end: byte_map[b - 1].1,
                            });
                        }
                    } else if !normalized.is_empty() && !byte_map.is_empty() {
                        words.push(WordSpan {
                            text: normalized,
                            start: byte_map[0].0,
                            end: byte_map[byte_map.len() - 1].1,
                        });
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
            self.push_word(word, &mut ids);
            word_ids.resize(ids.len(), Some(i as u32));
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

    pub(crate) fn word_token_count(&self, word: &str) -> usize {
        let mut ids = Vec::new();
        self.push_word(word, &mut ids);
        ids.len().max(1)
    }

    pub(crate) fn special_token_count(&self, add_special_tokens: bool) -> usize {
        if !add_special_tokens {
            return 0;
        }
        usize::from(self.cls.is_some()) + usize::from(self.sep.is_some())
    }
}

#[cfg(test)]
#[path = "wordpiece_tests.rs"]
mod tests;
