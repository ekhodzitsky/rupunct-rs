use std::path::Path;

use crate::error::{Error, Result};
use crate::labels::id2label_from_map;
use crate::limits::{
    MAX_CONFIG_BYTES, MAX_HEADS, MAX_HIDDEN, MAX_INTERMEDIATE, MAX_LABELS, MAX_LAYERS,
    MAX_POSITIONS, MAX_TYPE_VOCAB, MAX_VOCAB, MIN_POSITIONS, check_cap, read_capped_text,
};
use crate::model::CONFIG_FILE;

#[derive(Clone, Debug)]
pub(crate) struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f32,
    pub num_labels: usize,
    pub id2label: Vec<String>,
}

impl Config {
    pub(crate) fn from_json(config_path: &Path) -> Result<Self> {
        let raw = read_capped_text(config_path, MAX_CONFIG_BYTES)?;
        let v: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| Error::config(format!("{CONFIG_FILE} is not valid JSON: {e}")))?;
        if let Some(mt) = v.get("model_type").and_then(|x| x.as_str())
            && mt != "bert"
        {
            return Err(Error::config(format!(
                "unsupported model_type '{mt}' (expected bert)"
            )));
        }
        if let Some(pe) = v.get("position_embedding_type").and_then(|x| x.as_str())
            && pe != "absolute"
        {
            return Err(Error::config(format!(
                "unsupported position_embedding_type '{pe}' (expected absolute)"
            )));
        }
        let hidden = json_usize(&v, "hidden_size")?;
        let heads = json_usize(&v, "num_attention_heads")?;
        if hidden == 0 {
            return Err(Error::config("hidden_size must be > 0"));
        }
        if heads == 0 {
            return Err(Error::config("num_attention_heads must be > 0"));
        }
        if hidden % heads != 0 {
            return Err(Error::config(format!(
                "hidden_size {hidden} not divisible by num_attention_heads {heads}"
            )));
        }
        if let Some(act) = v.get("hidden_act").and_then(|x| x.as_str())
            && act != "gelu"
        {
            return Err(Error::config(format!(
                "unsupported hidden_act '{act}' (expected gelu)"
            )));
        }
        let id2label = match v.get("id2label").and_then(|m| m.as_object()) {
            Some(map) => id2label_from_map(map)?,
            None => Vec::new(),
        };
        let num_labels = match json_usize(&v, "num_labels") {
            Ok(n) => n,
            Err(_) if !id2label.is_empty() => id2label.len(),
            Err(e) => return Err(e),
        };
        if !id2label.is_empty() && id2label.len() != num_labels {
            return Err(Error::config(format!(
                "id2label len {} != num_labels {num_labels}",
                id2label.len()
            )));
        }
        let vocab_size = json_usize(&v, "vocab_size")?;
        let num_hidden_layers = json_usize(&v, "num_hidden_layers")?;
        let intermediate_size = json_usize(&v, "intermediate_size")?;
        let max_position_embeddings = json_usize(&v, "max_position_embeddings")?;
        let type_vocab_size = json_usize(&v, "type_vocab_size")?;
        if num_hidden_layers == 0 {
            return Err(Error::config("num_hidden_layers must be > 0"));
        }
        if intermediate_size == 0 {
            return Err(Error::config("intermediate_size must be > 0"));
        }
        if vocab_size == 0 {
            return Err(Error::config("vocab_size must be > 0"));
        }
        if type_vocab_size == 0 {
            return Err(Error::config("type_vocab_size must be > 0"));
        }
        if max_position_embeddings < MIN_POSITIONS {
            return Err(Error::config(format!(
                "max_position_embeddings must be >= {MIN_POSITIONS}"
            )));
        }
        if num_labels == 0 {
            return Err(Error::config("num_labels must be > 0"));
        }
        check_cap("hidden_size", hidden, MAX_HIDDEN)?;
        check_cap("num_attention_heads", heads, MAX_HEADS)?;
        check_cap("num_hidden_layers", num_hidden_layers, MAX_LAYERS)?;
        check_cap("intermediate_size", intermediate_size, MAX_INTERMEDIATE)?;
        check_cap(
            "max_position_embeddings",
            max_position_embeddings,
            MAX_POSITIONS,
        )?;
        check_cap("vocab_size", vocab_size, MAX_VOCAB)?;
        check_cap("type_vocab_size", type_vocab_size, MAX_TYPE_VOCAB)?;
        check_cap("num_labels", num_labels, MAX_LABELS)?;
        let eps = json_f64(&v, "layer_norm_eps")? as f32;
        if !eps.is_finite() || eps <= 0.0 {
            return Err(Error::config("layer_norm_eps must be finite and > 0"));
        }
        Ok(Self {
            vocab_size,
            hidden_size: hidden,
            num_hidden_layers,
            num_attention_heads: heads,
            intermediate_size,
            max_position_embeddings,
            type_vocab_size,
            layer_norm_eps: eps,
            num_labels,
            id2label,
        })
    }

    pub(crate) fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

fn json_usize(v: &serde_json::Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(|x| x.as_u64())
        .and_then(|x| usize::try_from(x).ok())
        .ok_or_else(|| Error::config(format!("missing or non-integer field '{key}'")))
}

fn json_f64(v: &serde_json::Value, key: &str) -> Result<f64> {
    v.get(key)
        .and_then(|x| x.as_f64())
        .ok_or_else(|| Error::config(format!("missing or non-float field '{key}'")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_rejects_indivisible_heads() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tmp.path().join(CONFIG_FILE);
        std::fs::write(
            &cfg,
            r#"{"hidden_size": 5, "num_attention_heads": 2, "num_hidden_layers": 1,
               "intermediate_size": 8, "vocab_size": 4, "max_position_embeddings": 8,
               "type_vocab_size": 2, "layer_norm_eps": 1e-12}"#,
        )
        .unwrap();
        assert!(Config::from_json(&cfg).is_err());
    }

    #[test]
    fn config_rejects_zero_heads_and_hidden() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tmp.path().join(CONFIG_FILE);
        let body = |hidden: u32, heads: u32| {
            format!(
                r#"{{"hidden_size": {hidden}, "num_attention_heads": {heads}, "num_hidden_layers": 1,
               "intermediate_size": 8, "vocab_size": 4, "max_position_embeddings": 8,
               "type_vocab_size": 2, "layer_norm_eps": 1e-12, "num_labels": 1,
               "id2label": {{"0": "LOWER_O"}}}}"#
            )
        };
        std::fs::write(&cfg, body(4, 0)).unwrap();
        let err = Config::from_json(&cfg).unwrap_err().to_string();
        assert!(err.contains("num_attention_heads"), "{err}");
        std::fs::write(&cfg, body(0, 1)).unwrap();
        let err = Config::from_json(&cfg).unwrap_err().to_string();
        assert!(err.contains("hidden_size"), "{err}");
    }

    #[test]
    fn config_reads_num_labels_from_id2label() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tmp.path().join(CONFIG_FILE);
        std::fs::write(
            &cfg,
            r#"{"hidden_size": 4, "num_attention_heads": 2, "num_hidden_layers": 1,
               "intermediate_size": 8, "vocab_size": 4, "max_position_embeddings": 8,
               "type_vocab_size": 2, "layer_norm_eps": 1e-12,
               "id2label": {"0": "LOWER_O", "1": "UPPER_O"}}"#,
        )
        .unwrap();
        let c = Config::from_json(&cfg).unwrap();
        assert_eq!(c.num_labels, 2);
        assert_eq!(c.id2label, ["LOWER_O", "UPPER_O"]);
        assert_eq!(c.head_dim(), 2);
    }

    #[test]
    fn config_rejects_missing_labels_and_bad_act() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tmp.path().join(CONFIG_FILE);
        std::fs::write(
            &cfg,
            r#"{"hidden_size": 4, "num_attention_heads": 2, "num_hidden_layers": 1,
               "intermediate_size": 8, "vocab_size": 4, "max_position_embeddings": 8,
               "type_vocab_size": 2, "layer_norm_eps": 1e-12}"#,
        )
        .unwrap();
        assert!(Config::from_json(&cfg).is_err());
        std::fs::write(
            &cfg,
            r#"{"hidden_size": 4, "num_attention_heads": 2, "num_hidden_layers": 1,
               "intermediate_size": 8, "vocab_size": 4, "max_position_embeddings": 8,
               "type_vocab_size": 2, "layer_norm_eps": 1e-12, "num_labels": 2,
               "hidden_act": "relu"}"#,
        )
        .unwrap();
        assert!(Config::from_json(&cfg).is_err());
    }

    #[test]
    fn config_rejects_bad_json() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tmp.path().join(CONFIG_FILE);
        std::fs::write(&cfg, "{").unwrap();
        assert!(Config::from_json(&cfg).is_err());
        assert!(Config::from_json(&tmp.path().join("missing.json")).is_err());
    }

    #[test]
    fn config_rejects_relative_positions_and_tiny_max_pos() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tmp.path().join(CONFIG_FILE);
        std::fs::write(
            &cfg,
            r#"{"hidden_size": 4, "num_attention_heads": 2, "num_hidden_layers": 1,
               "intermediate_size": 8, "vocab_size": 4, "max_position_embeddings": 8,
               "type_vocab_size": 2, "layer_norm_eps": 1e-12, "num_labels": 1,
               "id2label": {"0": "LOWER_O"}, "position_embedding_type": "relative"}"#,
        )
        .unwrap();
        let err = Config::from_json(&cfg).unwrap_err().to_string();
        assert!(err.contains("position_embedding_type"), "{err}");
        std::fs::write(
            &cfg,
            r#"{"hidden_size": 4, "num_attention_heads": 2, "num_hidden_layers": 1,
               "intermediate_size": 8, "vocab_size": 4, "max_position_embeddings": 2,
               "type_vocab_size": 2, "layer_norm_eps": 1e-12, "num_labels": 1,
               "id2label": {"0": "LOWER_O"}}"#,
        )
        .unwrap();
        let err = Config::from_json(&cfg).unwrap_err().to_string();
        assert!(err.contains("max_position_embeddings"), "{err}");
        std::fs::write(
            &cfg,
            r#"{"hidden_size": 5000, "num_attention_heads": 2, "num_hidden_layers": 1,
               "intermediate_size": 8, "vocab_size": 4, "max_position_embeddings": 8,
               "type_vocab_size": 2, "layer_norm_eps": 1e-12, "num_labels": 1,
               "id2label": {"0": "LOWER_O"}}"#,
        )
        .unwrap();
        let err = Config::from_json(&cfg).unwrap_err().to_string();
        assert!(err.contains("hidden_size"), "{err}");
        std::fs::write(
            &cfg,
            r#"{"hidden_size": 4, "num_attention_heads": 2, "num_hidden_layers": 1,
               "intermediate_size": 8, "vocab_size": 4, "max_position_embeddings": 8,
               "type_vocab_size": 2, "layer_norm_eps": 0, "num_labels": 1,
               "id2label": {"0": "LOWER_O"}}"#,
        )
        .unwrap();
        let err = Config::from_json(&cfg).unwrap_err().to_string();
        assert!(err.contains("layer_norm_eps"), "{err}");
        std::fs::write(
            &cfg,
            r#"{"hidden_size": 4, "num_attention_heads": 2, "num_hidden_layers": 1,
               "intermediate_size": 8, "vocab_size": 4, "max_position_embeddings": 8,
               "type_vocab_size": 2, "layer_norm_eps": 1e-12, "num_labels": 1,
               "id2label": {"0": "LOWER_O"}, "model_type": "roberta"}"#,
        )
        .unwrap();
        let err = Config::from_json(&cfg).unwrap_err().to_string();
        assert!(err.contains("model_type"), "{err}");
        std::fs::write(
            &cfg,
            r#"{"hidden_size": 4, "num_attention_heads": 2, "num_hidden_layers": 0,
               "intermediate_size": 8, "vocab_size": 4, "max_position_embeddings": 8,
               "type_vocab_size": 2, "layer_norm_eps": 1e-12, "num_labels": 1,
               "id2label": {"0": "LOWER_O"}}"#,
        )
        .unwrap();
        let err = Config::from_json(&cfg).unwrap_err().to_string();
        assert!(err.contains("num_hidden_layers"), "{err}");
    }
}
