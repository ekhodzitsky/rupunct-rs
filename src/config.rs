use std::path::Path;

use crate::error::{Error, Result};
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
}

impl Config {
    pub(crate) fn from_json(config_path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(config_path).map_err(|e| Error::io(config_path, e))?;
        let v: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| Error::config(format!("{CONFIG_FILE} is not valid JSON: {e}")))?;
        let hidden = json_usize(&v, "hidden_size")?;
        let heads = json_usize(&v, "num_attention_heads")?;
        if hidden % heads != 0 {
            return Err(Error::config(format!(
                "hidden_size {hidden} not divisible by num_attention_heads {heads}"
            )));
        }
        let num_labels = json_usize(&v, "num_labels").unwrap_or_else(|_| {
            v.get("id2label")
                .and_then(|m| m.as_object())
                .map(|m| m.len())
                .unwrap_or(33)
        });
        Ok(Self {
            vocab_size: json_usize(&v, "vocab_size")?,
            hidden_size: hidden,
            num_hidden_layers: json_usize(&v, "num_hidden_layers")?,
            num_attention_heads: heads,
            intermediate_size: json_usize(&v, "intermediate_size")?,
            max_position_embeddings: json_usize(&v, "max_position_embeddings")?,
            type_vocab_size: json_usize(&v, "type_vocab_size")?,
            layer_norm_eps: json_f64(&v, "layer_norm_eps")? as f32,
            num_labels,
        })
    }

    pub(crate) fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

fn json_usize(v: &serde_json::Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
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
        assert_eq!(c.head_dim(), 2);
    }

    #[test]
    fn config_rejects_bad_json() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tmp.path().join(CONFIG_FILE);
        std::fs::write(&cfg, "{").unwrap();
        assert!(Config::from_json(&cfg).is_err());
        assert!(Config::from_json(&tmp.path().join("missing.json")).is_err());
    }
}
