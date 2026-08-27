//! RUPunct BERT token-classification.

use std::path::Path;

use crate::config::Config;
use crate::encoder::BertLayer;
use crate::error::{Error, Result};
use crate::labels::load_id2label;
use crate::layer_norm::LayerNorm;
use crate::linear::Linear;
use crate::model::{CONFIG_FILE, MODEL_FILE};
use crate::tensor::Tensor;
use crate::weights::load_safetensors_file;
use crate::ws::{
    add_inplace, check_emb_shape, gather_rows, layer_norm_from_ws, linear_from_ws, require_f32,
};

struct Embeddings {
    word: Tensor,
    position: Tensor,
    token_type: Tensor,
    layer_norm: LayerNorm,
}

impl Embeddings {
    fn load(ws: &crate::weights::WeightSet, config: &Config) -> Result<Self> {
        let word = require_f32(ws, "bert.embeddings.word_embeddings.weight")?;
        let position = require_f32(ws, "bert.embeddings.position_embeddings.weight")?;
        let token_type = require_f32(ws, "bert.embeddings.token_type_embeddings.weight")?;
        check_emb_shape(&word, config.vocab_size, config.hidden_size, "word")?;
        check_emb_shape(
            &position,
            config.max_position_embeddings,
            config.hidden_size,
            "position",
        )?;
        check_emb_shape(
            &token_type,
            config.type_vocab_size,
            config.hidden_size,
            "token_type",
        )?;
        let layer_norm = layer_norm_from_ws(
            ws,
            "bert.embeddings.LayerNorm.weight",
            "bert.embeddings.LayerNorm.bias",
            config.layer_norm_eps,
        )?;
        Ok(Self {
            word,
            position,
            token_type,
            layer_norm,
        })
    }

    fn forward(&self, input_ids: &[u32], token_type_ids: &[u32]) -> Result<Tensor> {
        let seq = input_ids.len();
        if token_type_ids.len() != seq {
            return Err(Error::model(format!(
                "token_type_ids len {} != seq {seq}",
                token_type_ids.len()
            )));
        }
        let mut x = gather_rows(&self.word, input_ids)?;
        let tt = gather_rows(&self.token_type, token_type_ids)?;
        add_inplace(x.as_mut_slice(), tt.as_slice());
        let pos_ids: Vec<u32> = (0..seq as u32).collect();
        let pos = gather_rows(&self.position, &pos_ids)?;
        add_inplace(x.as_mut_slice(), pos.as_slice());
        Ok(self.layer_norm.forward(&x)?)
    }
}

pub(crate) struct BertForTokenClassification {
    embeddings: Embeddings,
    layers: Vec<BertLayer>,
    classifier: Linear,
    id2label: Vec<String>,
    max_position_embeddings: usize,
}

impl BertForTokenClassification {
    pub(crate) fn load(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join(CONFIG_FILE);
        let config = Config::from_json(&config_path)?;
        let weights_path = model_dir.join(MODEL_FILE);
        if !weights_path.is_file() {
            return Err(Error::model(format!("missing {}", weights_path.display())));
        }
        let ws = load_safetensors_file(&weights_path)
            .map_err(|e| Error::model(format!("load weights {}: {e}", weights_path.display())))?;

        let embeddings = Embeddings::load(&ws, &config)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(BertLayer::load(&ws, i, &config)?);
        }
        let classifier = linear_from_ws(&ws, "classifier.weight", "classifier.bias")?;
        if classifier.out_dim() != config.num_labels {
            return Err(Error::config(format!(
                "classifier out {} != num_labels {}",
                classifier.out_dim(),
                config.num_labels
            )));
        }
        let id2label = load_id2label(&config_path)?;
        Ok(Self {
            embeddings,
            layers,
            classifier,
            id2label,
            max_position_embeddings: config.max_position_embeddings,
        })
    }

    pub(crate) fn forward(
        &self,
        input_ids: &[u32],
        token_type_ids: &[u32],
        attention_mask: &[u32],
    ) -> Result<Tensor> {
        let mut x = self.embeddings.forward(input_ids, token_type_ids)?;
        for layer in &self.layers {
            x = layer.forward(&x, attention_mask)?;
        }
        Ok(self.classifier.forward(&x)?)
    }

    pub(crate) fn id2label(&self) -> &[String] {
        &self.id2label
    }

    pub(crate) fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }
}
