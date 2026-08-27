//! RUPunct label decode: casing + punctuation suffix per word.

use std::path::Path;

use crate::error::{Error, Result};
use crate::model::CONFIG_FILE;

fn capitalize(token: &str) -> String {
    let mut chars = token.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => {
            let mut out: String = first.to_uppercase().collect();
            for c in chars {
                out.extend(c.to_lowercase());
            }
            out
        }
    }
}

/// Cased + punctuated rendering of one word given its RUPunct label.
///
/// Matches `process_token` on the RUPunct_small model card.
pub fn process_token(token: &str, label: &str) -> String {
    let (cased, punct_class) = if let Some(rest) = label.strip_prefix("UPPER_TOTAL_") {
        (token.to_uppercase(), rest)
    } else if let Some(rest) = label.strip_prefix("UPPER_") {
        (capitalize(token), rest)
    } else if let Some(rest) = label.strip_prefix("LOWER_") {
        (token.to_string(), rest)
    } else {
        return token.to_string();
    };

    let is_upper = !label.starts_with("LOWER_");
    let suffix: &str = match punct_class {
        "O" => "",
        "PERIOD" => ".",
        "COMMA" => ",",
        "QUESTION" => "?",
        "VOSKL" => "!",
        "DVOETOCHIE" => ":",
        "PERIODCOMMA" => ";",
        "DEFIS" => "-",
        "MNOGOTOCHIE" => "...",
        "QUESTIONVOSKL" => "?!",
        "TIRE" => {
            if is_upper {
                " —"
            } else {
                "—"
            }
        }
        _ => "",
    };

    let mut out = cased;
    out.push_str(suffix);
    out
}

pub(crate) fn first_subword_labels(
    word_ids: &[Option<u32>],
    argmax_per_token: &[usize],
    num_words: usize,
) -> Vec<usize> {
    let mut labels = vec![0usize; num_words];
    let mut seen = vec![false; num_words];
    for (tok_idx, wid) in word_ids.iter().enumerate() {
        let Some(w) = wid else { continue };
        let w = *w as usize;
        if w < num_words && !seen[w] {
            seen[w] = true;
            labels[w] = argmax_per_token.get(tok_idx).copied().unwrap_or(0);
        }
    }
    labels
}

pub(crate) fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

pub(crate) fn load_id2label(config_path: &Path) -> Result<Vec<String>> {
    let raw = std::fs::read_to_string(config_path).map_err(|e| Error::io(config_path, e))?;
    let config: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| Error::config(format!("{CONFIG_FILE} is not valid JSON: {e}")))?;
    let map = config
        .get("id2label")
        .and_then(|v| v.as_object())
        .ok_or_else(|| Error::config(format!("{CONFIG_FILE} missing id2label object")))?;

    let mut labels = vec![String::new(); map.len()];
    for (k, v) in map {
        let idx: usize = k
            .parse()
            .map_err(|_| Error::config(format!("id2label key '{k}' is not an integer")))?;
        let label = v
            .as_str()
            .ok_or_else(|| Error::config(format!("id2label['{k}'] is not a string")))?;
        if idx >= labels.len() {
            return Err(Error::config(format!(
                "id2label index {idx} out of range ({} labels)",
                map.len()
            )));
        }
        labels[idx] = label.to_string();
    }
    if labels.iter().any(|l| l.is_empty()) {
        return Err(Error::config("id2label has a gap (non-contiguous indices)"));
    }
    Ok(labels)
}

#[cfg(test)]
#[path = "labels_tests.rs"]
mod tests;
