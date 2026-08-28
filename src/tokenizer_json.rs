//! HuggingFace `tokenizer.json` parsing for the supported BERT WordPiece scheme.
//! Field walking uses `serde_json::Value` only — no `Deserialize`.

use std::collections::HashMap;

use crate::error::{Error, Result};

pub(crate) struct TokenizerFile {
    pub version: Option<String>,
    pub truncation: Option<serde_json::Value>,
    pub padding: Option<serde_json::Value>,
    pub added_tokens: Vec<AddedTokenFile>,
    pub normalizer: Option<serde_json::Value>,
    pub pre_tokenizer: Option<serde_json::Value>,
    pub post_processor: Option<serde_json::Value>,
    pub model: ModelFile,
}

pub(crate) struct AddedTokenFile {
    pub id: u32,
    pub content: String,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
    pub normalized: bool,
}

pub(crate) struct ModelFile {
    pub kind: String,
    pub unk_token: Option<String>,
    pub continuing_subword_prefix: Option<String>,
    pub max_input_chars_per_word: Option<usize>,
    pub vocab: HashMap<String, u32>,
}

pub(crate) enum Normalizer {
    None,
    Bert {
        clean_text: bool,
        handle_chinese_chars: bool,
        lowercase: bool,
    },
}

impl TokenizerFile {
    pub(crate) fn from_value(value: serde_json::Value) -> Result<Self> {
        let obj = as_object(&value, "tokenizer")?;
        let model = parse_model(
            obj.get("model")
                .ok_or_else(|| Error::tokenizer("tokenizer file is missing a 'model' object"))?,
        )?;
        Ok(Self {
            version: opt_string(obj, "version")?,
            truncation: opt_non_null(obj, "truncation"),
            padding: opt_non_null(obj, "padding"),
            added_tokens: parse_added_tokens(obj.get("added_tokens"))?,
            normalizer: opt_non_null(obj, "normalizer"),
            pre_tokenizer: opt_non_null(obj, "pre_tokenizer"),
            post_processor: opt_non_null(obj, "post_processor"),
            model,
        })
    }
}

fn parse_model(value: &serde_json::Value) -> Result<ModelFile> {
    let obj = as_object(value, "model")?;
    let kind = req_string(obj, "type")?;
    let vocab_v = obj
        .get("vocab")
        .ok_or_else(|| Error::tokenizer("WordPiece model is missing 'vocab'"))?;
    let vocab_obj = as_object(vocab_v, "vocab")?;
    let mut vocab = HashMap::with_capacity(vocab_obj.len());
    for (tok, id) in vocab_obj {
        let id = as_u32(id, &format!("vocab['{tok}']"))?;
        vocab.insert(tok.clone(), id);
    }
    Ok(ModelFile {
        kind,
        unk_token: opt_string(obj, "unk_token")?,
        continuing_subword_prefix: opt_string(obj, "continuing_subword_prefix")?,
        max_input_chars_per_word: opt_usize(obj, "max_input_chars_per_word")?,
        vocab,
    })
}

fn parse_added_tokens(value: Option<&serde_json::Value>) -> Result<Vec<AddedTokenFile>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    if value.is_null() {
        return Ok(Vec::new());
    }
    let arr = value
        .as_array()
        .ok_or_else(|| Error::tokenizer("added_tokens must be an array"))?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, item) in arr.iter().enumerate() {
        let obj = as_object(item, &format!("added_tokens[{i}]"))?;
        out.push(AddedTokenFile {
            id: req_u32(obj, "id")?,
            content: req_string(obj, "content")?,
            single_word: opt_bool(obj, "single_word", false)?,
            lstrip: opt_bool(obj, "lstrip", false)?,
            rstrip: opt_bool(obj, "rstrip", false)?,
            normalized: opt_bool(obj, "normalized", true)?,
        });
    }
    Ok(out)
}

pub(crate) fn parse_normalizer(value: Option<serde_json::Value>) -> Result<Normalizer> {
    let Some(value) = value else {
        return Ok(Normalizer::None);
    };
    let obj = as_object(&value, "normalizer")?;
    let kind = req_string(obj, "type")?;
    if kind != "BertNormalizer" {
        return Err(Error::tokenizer(format!(
            "unsupported normalizer type '{kind}' (expected 'BertNormalizer' or null)"
        )));
    }
    let lowercase = opt_bool(obj, "lowercase", true)?;
    let strip_accents = opt_bool(obj, "strip_accents", lowercase)?;
    if strip_accents {
        return Err(Error::tokenizer(
            "BertNormalizer with strip_accents=true is not supported",
        ));
    }
    Ok(Normalizer::Bert {
        clean_text: opt_bool(obj, "clean_text", true)?,
        handle_chinese_chars: opt_bool(obj, "handle_chinese_chars", true)?,
        lowercase,
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
    let obj = as_object(&value, "post_processor")?;
    let kind = req_string(obj, "type")?;
    if kind != "TemplateProcessing" {
        return Err(Error::tokenizer(format!(
            "unsupported post_processor type '{kind}' (expected 'TemplateProcessing' or null)"
        )));
    }
    let single = obj
        .get("single")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error::tokenizer("post_processor is missing a 'single' template"))?;
    let [first, mid, last] = single.as_slice() else {
        return Err(Error::tokenizer(
            "unsupported post_processor template (expected '[Special] A [Special]')",
        ));
    };
    let (mid_kind, mid_id, mid_ty) = parse_piece(mid)?;
    if mid_kind != "Sequence" || mid_id != "A" || mid_ty != 0 {
        return Err(Error::tokenizer(
            "unsupported post_processor template (expected sequence 'A' with type_id 0)",
        ));
    }
    let specials = obj
        .get("special_tokens")
        .and_then(|v| v.as_object())
        .ok_or_else(|| Error::tokenizer("post_processor is missing 'special_tokens'"))?;
    let special_id = |piece: &serde_json::Value| -> Result<u32> {
        let (kind, id, type_id) = parse_piece(piece)?;
        if kind != "SpecialToken" {
            return Err(Error::tokenizer(
                "unsupported post_processor template (expected '[Special] A [Special]')",
            ));
        }
        if type_id != 0 {
            return Err(Error::tokenizer(format!(
                "unsupported special token type_id {type_id} (expected 0)"
            )));
        }
        let entry = specials.get(&id).ok_or_else(|| {
            Error::tokenizer(format!(
                "post_processor references unknown special token '{id}'"
            ))
        })?;
        let ids = entry
            .get("ids")
            .and_then(|v| v.as_array())
            .ok_or_else(|| Error::tokenizer(format!("special token '{id}' is missing 'ids'")))?;
        let [token_id] = ids.as_slice() else {
            return Err(Error::tokenizer(format!(
                "special token '{id}' expands to more than one id, which is not supported"
            )));
        };
        as_u32(token_id, &format!("special token '{id}' id"))
    };
    Ok((Some(special_id(first)?), Some(special_id(last)?)))
}

fn parse_piece(value: &serde_json::Value) -> Result<(String, String, u32)> {
    let obj = as_object(value, "template piece")?;
    if obj.len() != 1 {
        return Err(Error::tokenizer(
            "unsupported post_processor template (expected '[Special] A [Special]')",
        ));
    }
    let Some((kind, inner)) = obj.iter().next() else {
        return Err(Error::tokenizer(
            "unsupported post_processor template (expected '[Special] A [Special]')",
        ));
    };
    let inner = as_object(inner, "template piece")?;
    let id = req_string(inner, "id")?;
    let type_id = req_u32(inner, "type_id")?;
    Ok((kind.clone(), id, type_id))
}

fn as_object<'a>(
    v: &'a serde_json::Value,
    ctx: &str,
) -> Result<&'a serde_json::Map<String, serde_json::Value>> {
    v.as_object()
        .ok_or_else(|| Error::tokenizer(format!("{ctx} must be a JSON object")))
}

fn opt_non_null(
    obj: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<serde_json::Value> {
    match obj.get(key) {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => Some(v.clone()),
    }
}

fn opt_string(
    obj: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<Option<String>> {
    match obj.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(v) => v
            .as_str()
            .map(|s| Some(s.to_string()))
            .ok_or_else(|| Error::tokenizer(format!("'{key}' must be a string"))),
    }
}

fn req_string(obj: &serde_json::Map<String, serde_json::Value>, key: &str) -> Result<String> {
    obj.get(key)
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| Error::tokenizer(format!("missing string field '{key}'")))
}

fn opt_bool(
    obj: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    default: bool,
) -> Result<bool> {
    match obj.get(key) {
        None | Some(serde_json::Value::Null) => Ok(default),
        Some(v) => v
            .as_bool()
            .ok_or_else(|| Error::tokenizer(format!("'{key}' must be a boolean"))),
    }
}

fn opt_usize(obj: &serde_json::Map<String, serde_json::Value>, key: &str) -> Result<Option<usize>> {
    match obj.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(v) => Ok(Some(as_usize(v, key)?)),
    }
}

fn req_u32(obj: &serde_json::Map<String, serde_json::Value>, key: &str) -> Result<u32> {
    let v = obj
        .get(key)
        .ok_or_else(|| Error::tokenizer(format!("missing field '{key}'")))?;
    as_u32(v, key)
}

fn as_u32(v: &serde_json::Value, ctx: &str) -> Result<u32> {
    v.as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| Error::tokenizer(format!("{ctx} must be a u32")))
}

fn as_usize(v: &serde_json::Value, ctx: &str) -> Result<usize> {
    v.as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| Error::tokenizer(format!("{ctx} must be an integer")))
}

#[cfg(test)]
#[path = "tokenizer_json_tests.rs"]
mod tests;
