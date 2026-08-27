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
use crate::unicode::bert_pre_tokenize;

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

    #[cfg(test)]
    pub(crate) fn encode(&self, text: &str, add_special_tokens: bool) -> Encoding {
        self.encode_words(&self.split_words(text), add_special_tokens)
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
