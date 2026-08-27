//! RUPunct label decode: casing + punctuation suffix per word.

use std::path::Path;

use anyhow::{Context, Result};

pub const PUNCT_TOKENIZER_FILE: &str = "tokenizer.json";
pub const PUNCT_CONFIG_FILE: &str = "config.json";

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
/// Verbatim port of `process_token` from the `RUPunct/RUPunct_small` model card.
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
    let raw = std::fs::read_to_string(config_path)
        .with_context(|| format!("Failed to read {}", config_path.display()))?;
    let config: serde_json::Value =
        serde_json::from_str(&raw).context("config.json is not valid JSON")?;
    let map = config
        .get("id2label")
        .and_then(|v| v.as_object())
        .context("config.json missing id2label object")?;

    let mut labels = vec![String::new(); map.len()];
    for (k, v) in map {
        let idx: usize = k
            .parse()
            .with_context(|| format!("id2label key '{k}' is not an integer"))?;
        let label = v
            .as_str()
            .with_context(|| format!("id2label['{k}'] is not a string"))?;
        if idx >= labels.len() {
            anyhow::bail!("id2label index {idx} out of range ({} labels)", map.len());
        }
        labels[idx] = label.to_string();
    }
    if labels.iter().any(|l| l.is_empty()) {
        anyhow::bail!("id2label has a gap (non-contiguous indices)");
    }
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capitalize_python_semantics() {
        assert_eq!(capitalize("привет"), "Привет");
        assert_eq!(capitalize("ПРИВЕТ"), "Привет");
        assert_eq!(capitalize("пРиВеТ"), "Привет");
        assert_eq!(capitalize(""), "");
        assert_eq!(capitalize("a"), "A");
    }

    #[test]
    fn test_process_token_lower_modes() {
        assert_eq!(process_token("слово", "LOWER_O"), "слово");
        assert_eq!(process_token("слово", "LOWER_PERIOD"), "слово.");
        assert_eq!(process_token("слово", "LOWER_COMMA"), "слово,");
        assert_eq!(process_token("слово", "LOWER_QUESTION"), "слово?");
        assert_eq!(process_token("слово", "LOWER_VOSKL"), "слово!");
        assert_eq!(process_token("слово", "LOWER_DVOETOCHIE"), "слово:");
        assert_eq!(process_token("слово", "LOWER_PERIODCOMMA"), "слово;");
        assert_eq!(process_token("слово", "LOWER_DEFIS"), "слово-");
        assert_eq!(process_token("слово", "LOWER_MNOGOTOCHIE"), "слово...");
        assert_eq!(process_token("слово", "LOWER_QUESTIONVOSKL"), "слово?!");
    }

    #[test]
    fn test_process_token_upper_capitalizes_first_lowercases_rest() {
        assert_eq!(process_token("анна", "UPPER_O"), "Анна");
        assert_eq!(process_token("анна", "UPPER_COMMA"), "Анна,");
        assert_eq!(process_token("ПРИВЕТ", "UPPER_PERIOD"), "Привет.");
    }

    #[test]
    fn test_process_token_upper_total_uppercases_all() {
        assert_eq!(process_token("ооо", "UPPER_TOTAL_O"), "ООО");
        assert_eq!(process_token("ссср", "UPPER_TOTAL_PERIOD"), "СССР.");
        assert_eq!(process_token("ооо", "UPPER_TOTAL_COMMA"), "ООО,");
    }

    #[test]
    fn test_process_token_tire_spacing_quirk() {
        assert_eq!(process_token("это", "LOWER_TIRE"), "это—");
        assert_eq!(process_token("это", "UPPER_TIRE"), "Это —");
        assert_eq!(process_token("это", "UPPER_TOTAL_TIRE"), "ЭТО —");
    }

    #[test]
    fn test_process_token_unknown_label_is_identity() {
        assert_eq!(process_token("слово", "GARBAGE"), "слово");
        assert_eq!(process_token("слово", "LOWER_BOGUS"), "слово");
    }

    #[test]
    fn test_first_subword_labels_picks_first_subtoken() {
        let word_ids = vec![None, Some(0), Some(0), Some(1), None];
        let argmax = vec![0, 3, 9, 7, 0];
        let labels = first_subword_labels(&word_ids, &argmax, 2);
        assert_eq!(labels, vec![3, 7]);
    }

    #[test]
    fn test_first_subword_labels_missing_word_defaults_zero() {
        let word_ids = vec![None, Some(0), None];
        let argmax = vec![0, 5, 0];
        let labels = first_subword_labels(&word_ids, &argmax, 2);
        assert_eq!(labels, vec![5, 0]);
    }

    #[test]
    fn test_argmax_returns_index_of_max() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 2.0]), 0);
        assert_eq!(argmax(&[1.0, 1.0, 3.0]), 2);
    }

    #[test]
    fn test_load_id2label_parses_contiguous_map() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cfg = tmp.path().join("config.json");
        std::fs::write(
            &cfg,
            r#"{"id2label": {"0": "UPPER_PERIOD", "1": "LOWER_PERIOD", "2": "UPPER_TOTAL_PERIOD"}}"#,
        )
        .unwrap();
        let labels = load_id2label(&cfg).expect("parse");
        assert_eq!(
            labels,
            vec!["UPPER_PERIOD", "LOWER_PERIOD", "UPPER_TOTAL_PERIOD"]
        );
    }

    #[test]
    fn test_load_id2label_rejects_gap() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cfg = tmp.path().join("config.json");
        std::fs::write(&cfg, r#"{"id2label": {"0": "A", "2": "C"}}"#).unwrap();
        assert!(load_id2label(&cfg).is_err());
    }
}
