//! RUPunct BERT token-classification.

use std::path::Path;

use crate::config::Config;
use crate::encoder::BertLayer;
use crate::error::{Error, Result};
use crate::labels::lower_o_index;
use crate::layer_norm::LayerNorm;
use crate::linear::Linear;
use crate::model::{CONFIG_FILE, MODEL_FILE};
use crate::tensor::Tensor;
use crate::weights::load_safetensors_file;
use crate::ws::{
    EmbTable, add_inplace, emb_from_ws, gather_emb, layer_norm_from_ws, linear_from_ws,
};

struct Embeddings {
    word: EmbTable,
    position: EmbTable,
    token_type: EmbTable,
    layer_norm: LayerNorm,
}

impl Embeddings {
    fn load(ws: &crate::weights::WeightSet, config: &Config) -> Result<Self> {
        let word = emb_from_ws(
            ws,
            "bert.embeddings.word_embeddings.weight",
            config.vocab_size,
            config.hidden_size,
            "word",
        )?;
        let position = emb_from_ws(
            ws,
            "bert.embeddings.position_embeddings.weight",
            config.max_position_embeddings,
            config.hidden_size,
            "position",
        )?;
        let token_type = emb_from_ws(
            ws,
            "bert.embeddings.token_type_embeddings.weight",
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
        let mut x = gather_emb(&self.word, input_ids)?;
        let tt = gather_emb(&self.token_type, token_type_ids)?;
        add_inplace(x.as_mut_slice(), tt.as_slice());
        let pos_ids: Vec<u32> = (0..seq as u32).collect();
        let pos = gather_emb(&self.position, &pos_ids)?;
        add_inplace(x.as_mut_slice(), pos.as_slice());
        Ok(self.layer_norm.forward(&x)?)
    }
}

pub(crate) struct BertForTokenClassification {
    embeddings: Embeddings,
    layers: Vec<BertLayer>,
    classifier: Linear,
    id2label: Vec<String>,
    default_label: usize,
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
        if config.id2label.is_empty() {
            return Err(Error::config(format!(
                "{CONFIG_FILE} missing id2label object"
            )));
        }
        if config.id2label.len() != config.num_labels {
            return Err(Error::config(format!(
                "id2label len {} != num_labels {}",
                config.id2label.len(),
                config.num_labels
            )));
        }
        let default_label = lower_o_index(&config.id2label)?;
        Ok(Self {
            embeddings,
            layers,
            classifier,
            id2label: config.id2label,
            default_label,
            max_position_embeddings: config.max_position_embeddings,
        })
    }

    pub(crate) fn default_label(&self) -> usize {
        self.default_label
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
