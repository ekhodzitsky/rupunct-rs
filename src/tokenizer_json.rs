//! HuggingFace `tokenizer.json` parsing for the supported BERT WordPiece scheme.

use std::collections::HashMap;

use serde::Deserialize;

use crate::error::{Error, Result};

#[derive(Deserialize)]
pub(crate) struct TokenizerFile {
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub truncation: Option<serde_json::Value>,
    #[serde(default)]
    pub padding: Option<serde_json::Value>,
    #[serde(default)]
    pub added_tokens: Vec<AddedTokenFile>,
    #[serde(default)]
    pub normalizer: Option<serde_json::Value>,
    #[serde(default)]
    pub pre_tokenizer: Option<serde_json::Value>,
    #[serde(default)]
    pub post_processor: Option<serde_json::Value>,
    pub model: ModelFile,
}

#[derive(Deserialize)]
pub(crate) struct AddedTokenFile {
    pub id: u32,
    pub content: String,
    #[serde(default)]
    pub single_word: bool,
    #[serde(default)]
    pub lstrip: bool,
    #[serde(default)]
    pub rstrip: bool,
    #[serde(default = "default_true")]
    pub normalized: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Deserialize)]
pub(crate) struct ModelFile {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub unk_token: Option<String>,
    #[serde(default)]
    pub continuing_subword_prefix: Option<String>,
    #[serde(default)]
    pub max_input_chars_per_word: Option<usize>,
    pub vocab: HashMap<String, u32>,
}

pub(crate) enum Normalizer {
    None,
    Bert {
        clean_text: bool,
        handle_chinese_chars: bool,
    },
}

pub(crate) fn parse_normalizer(value: Option<serde_json::Value>) -> Result<Normalizer> {
    let Some(value) = value else {
        return Ok(Normalizer::None);
    };
    #[derive(Deserialize)]
    struct BertNormalizerFile {
        #[serde(rename = "type")]
        kind: String,
        #[serde(default)]
        clean_text: Option<bool>,
        #[serde(default)]
        handle_chinese_chars: Option<bool>,
        #[serde(default)]
        strip_accents: Option<bool>,
        #[serde(default)]
        lowercase: Option<bool>,
    }
    let parsed: BertNormalizerFile = serde_json::from_value(value)
        .map_err(|e| Error::tokenizer(format!("failed to parse normalizer: {e}")))?;
    if parsed.kind != "BertNormalizer" {
        return Err(Error::tokenizer(format!(
            "unsupported normalizer type '{}' (expected 'BertNormalizer' or null)",
            parsed.kind
        )));
    }
    let lowercase = parsed.lowercase.unwrap_or(true);
    let strip_accents = parsed.strip_accents.unwrap_or(lowercase);
    if lowercase {
        return Err(Error::tokenizer(
            "BertNormalizer with lowercase=true is not supported",
        ));
    }
    if strip_accents {
        return Err(Error::tokenizer(
            "BertNormalizer with strip_accents=true is not supported",
        ));
    }
    Ok(Normalizer::Bert {
        clean_text: parsed.clean_text.unwrap_or(true),
        handle_chinese_chars: parsed.handle_chinese_chars.unwrap_or(true),
    })
}

pub(crate) fn parse_pre_tokenizer(value: Option<serde_json::Value>) -> Result<bool> {
    let Some(value) = value else {
        return Ok(false);
    };
    let kind = value
        .get("type")
        .and_then(|t| t.as_str())
        .ok_or_else(|| Error::tokenizer("pre_tokenizer is missing a 'type' field"))?;
    if kind != "BertPreTokenizer" {
        return Err(Error::tokenizer(format!(
            "unsupported pre_tokenizer type '{kind}' (expected 'BertPreTokenizer' or null)"
        )));
    }
    Ok(true)
}

pub(crate) fn parse_post_processor(
    value: Option<serde_json::Value>,
) -> Result<(Option<u32>, Option<u32>)> {
    let Some(value) = value else {
        return Ok((None, None));
    };

    #[derive(Deserialize)]
    struct TemplateProcessingFile {
        #[serde(rename = "type")]
        kind: String,
        single: Vec<TemplatePiece>,
        special_tokens: HashMap<String, SpecialTokenEntry>,
    }
    #[derive(Deserialize)]
    enum TemplatePiece {
        SpecialToken { id: String, type_id: u32 },
        Sequence { id: String, type_id: u32 },
    }
    #[derive(Deserialize)]
    struct SpecialTokenEntry {
        ids: Vec<u32>,
    }

    let parsed: TemplateProcessingFile = serde_json::from_value(value)
        .map_err(|e| Error::tokenizer(format!("failed to parse post_processor: {e}")))?;
    if parsed.kind != "TemplateProcessing" {
        return Err(Error::tokenizer(format!(
            "unsupported post_processor type '{}' (expected 'TemplateProcessing' or null)",
            parsed.kind
        )));
    }
    let [first, TemplatePiece::Sequence { id, type_id }, last] = parsed.single.as_slice() else {
        return Err(Error::tokenizer(
            "unsupported post_processor template (expected '[Special] A [Special]')",
        ));
    };
    if id != "A" || *type_id != 0 {
        return Err(Error::tokenizer(
            "unsupported post_processor template (expected sequence 'A' with type_id 0)",
        ));
    }
    let special_id = |piece: &TemplatePiece| -> Result<u32> {
        let TemplatePiece::SpecialToken { id, type_id } = piece else {
            return Err(Error::tokenizer(
                "unsupported post_processor template (expected '[Special] A [Special]')",
            ));
        };
        if *type_id != 0 {
            return Err(Error::tokenizer(format!(
                "unsupported special token type_id {type_id} (expected 0)"
            )));
        }
        let entry = parsed.special_tokens.get(id).ok_or_else(|| {
            Error::tokenizer(format!(
                "post_processor references unknown special token '{id}'"
            ))
        })?;
        let [token_id] = entry.ids.as_slice() else {
            return Err(Error::tokenizer(format!(
                "special token '{id}' expands to more than one id, which is not supported"
            )));
        };
        Ok(*token_id)
    };
    Ok((Some(special_id(first)?), Some(special_id(last)?)))
}

#[cfg(test)]
#[path = "tokenizer_json_tests.rs"]
mod tests;
