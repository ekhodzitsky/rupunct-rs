//! RUPunct BERT token-classification (ported from gigastt-pro).

use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::device::Device;
use crate::labels::{
    PUNCT_CONFIG_FILE, argmax, first_subword_labels, load_id2label, process_token,
};
use crate::layer_norm::LayerNorm;
use crate::linear::Linear;
use crate::tensor::Tensor;
use crate::weights::{WeightSet, load_safetensors_file};
use crate::wordpiece::Tokenizer;

/// HuggingFace-style BERT config subset needed for inference.
#[derive(Clone, Debug)]
pub struct Config {
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
    pub fn from_json(config_path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&raw).context("config.json is not valid JSON")?;
        let hidden = json_usize(&v, "hidden_size")?;
        let heads = json_usize(&v, "num_attention_heads")?;
        if hidden % heads != 0 {
            bail!("hidden_size {hidden} not divisible by num_attention_heads {heads}");
        }
        // classifier.weight is [num_labels, hidden]; prefer config, else 33 (RUPunct small).
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

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

fn json_usize(v: &serde_json::Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .with_context(|| format!("missing or non-integer field '{key}'"))
}

fn json_f64(v: &serde_json::Value, key: &str) -> Result<f64> {
    v.get(key)
        .and_then(|x| x.as_f64())
        .with_context(|| format!("missing or non-float field '{key}'"))
}

/// Word / position / token-type embeddings + LayerNorm.
struct Embeddings {
    word: Tensor,       // [vocab, H]
    position: Tensor,   // [max_pos, H]
    token_type: Tensor, // [type_vocab, H]
    layer_norm: LayerNorm,
}

impl Embeddings {
    fn load(ws: &WeightSet, config: &Config) -> Result<Self> {
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

    /// Returns `[seq, H]`.
    fn forward(&self, input_ids: &[u32], token_type_ids: &[u32]) -> Result<Tensor> {
        let seq = input_ids.len();
        if token_type_ids.len() != seq {
            bail!("token_type_ids len {} != seq {seq}", token_type_ids.len());
        }
        let mut x = gather_rows(&self.word, input_ids)?;
        let tt = gather_rows(&self.token_type, token_type_ids)?;
        add_inplace(x.as_mut_slice(), tt.as_slice());
        // Absolute positions 0..seq-1
        let pos_ids: Vec<u32> = (0..seq as u32).collect();
        let pos = gather_rows(&self.position, &pos_ids)?;
        add_inplace(x.as_mut_slice(), pos.as_slice());
        self.layer_norm.forward(&x).map_err(te)
    }
}

struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    dense: Linear,
    layer_norm: LayerNorm,
    n_heads: usize,
    head_dim: usize,
    hidden: usize,
}

impl Attention {
    fn load(ws: &WeightSet, prefix: &str, config: &Config) -> Result<Self> {
        Ok(Self {
            query: linear_from_ws(
                ws,
                &format!("{prefix}.attention.self.query.weight"),
                &format!("{prefix}.attention.self.query.bias"),
            )?,
            key: linear_from_ws(
                ws,
                &format!("{prefix}.attention.self.key.weight"),
                &format!("{prefix}.attention.self.key.bias"),
            )?,
            value: linear_from_ws(
                ws,
                &format!("{prefix}.attention.self.value.weight"),
                &format!("{prefix}.attention.self.value.bias"),
            )?,
            dense: linear_from_ws(
                ws,
                &format!("{prefix}.attention.output.dense.weight"),
                &format!("{prefix}.attention.output.dense.bias"),
            )?,
            layer_norm: layer_norm_from_ws(
                ws,
                &format!("{prefix}.attention.output.LayerNorm.weight"),
                &format!("{prefix}.attention.output.LayerNorm.bias"),
                config.layer_norm_eps,
            )?,
            n_heads: config.num_attention_heads,
            head_dim: config.head_dim(),
            hidden: config.hidden_size,
        })
    }

    /// `hidden_states`: `[seq, H]`, `attention_mask`: 1 = keep, 0 = pad.
    fn forward(&self, hidden_states: &Tensor, attention_mask: &[u32]) -> Result<Tensor> {
        let seq = hidden_states.dim(0).map_err(te)?;
        let q = self.query.forward(hidden_states).map_err(te)?;
        let k = self.key.forward(hidden_states).map_err(te)?;
        let v = self.value.forward(hidden_states).map_err(te)?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut context = vec![0.0f32; seq * self.hidden];
        let qs = q.as_slice();
        let ks = k.as_slice();
        let vs = v.as_slice();

        // Per-head SDPA (seq is small for punct; keeps deps to Linear only).
        let mut scores = vec![0.0f32; seq * seq];
        let mut probs = vec![0.0f32; seq * seq];
        let mut head_out = vec![0.0f32; seq * self.head_dim];

        for h in 0..self.n_heads {
            // Extract head slices into contiguous buffers.
            let mut qh = vec![0.0f32; seq * self.head_dim];
            let mut kh = vec![0.0f32; seq * self.head_dim];
            let mut vh = vec![0.0f32; seq * self.head_dim];
            for t in 0..seq {
                let src = t * self.hidden + h * self.head_dim;
                let dst = t * self.head_dim;
                qh[dst..dst + self.head_dim].copy_from_slice(&qs[src..src + self.head_dim]);
                kh[dst..dst + self.head_dim].copy_from_slice(&ks[src..src + self.head_dim]);
                vh[dst..dst + self.head_dim].copy_from_slice(&vs[src..src + self.head_dim]);
            }

            // scores[t,s] = (qh[t] · kh[s]) * scale
            for t in 0..seq {
                for s in 0..seq {
                    let mut acc = 0.0f32;
                    let qb = t * self.head_dim;
                    let kb = s * self.head_dim;
                    for d in 0..self.head_dim {
                        acc += qh[qb + d] * kh[kb + d];
                    }
                    scores[t * seq + s] = acc * scale;
                }
            }

            // Mask: pad positions (mask==0) get large negative on keys.
            for s in 0..seq {
                if attention_mask.get(s).copied().unwrap_or(0) == 0 {
                    for t in 0..seq {
                        scores[t * seq + s] += -1.0e4;
                    }
                }
            }

            // Softmax over keys (last dim).
            for t in 0..seq {
                let row = &scores[t * seq..(t + 1) * seq];
                let mut max_v = f32::NEG_INFINITY;
                for &v in row {
                    max_v = max_v.max(v);
                }
                let mut sum = 0.0f32;
                for s in 0..seq {
                    let e = (row[s] - max_v).exp();
                    probs[t * seq + s] = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                for s in 0..seq {
                    probs[t * seq + s] *= inv;
                }
            }

            // head_out = probs @ vh
            for t in 0..seq {
                for d in 0..self.head_dim {
                    let mut acc = 0.0f32;
                    for s in 0..seq {
                        acc += probs[t * seq + s] * vh[s * self.head_dim + d];
                    }
                    head_out[t * self.head_dim + d] = acc;
                }
            }

            for t in 0..seq {
                let dst = t * self.hidden + h * self.head_dim;
                let src = t * self.head_dim;
                context[dst..dst + self.head_dim]
                    .copy_from_slice(&head_out[src..src + self.head_dim]);
            }
        }

        let attn = Tensor::from_vec(context, vec![seq, self.hidden], Device::Cpu).map_err(te)?;
        let attn = self.dense.forward(&attn).map_err(te)?;
        let residual = hidden_states.add(&attn).map_err(te)?;
        self.layer_norm.forward(&residual).map_err(te)
    }
}

struct FeedForward {
    intermediate: Linear,
    output: Linear,
    layer_norm: LayerNorm,
}

impl FeedForward {
    fn load(ws: &WeightSet, prefix: &str, config: &Config) -> Result<Self> {
        Ok(Self {
            intermediate: linear_from_ws(
                ws,
                &format!("{prefix}.intermediate.dense.weight"),
                &format!("{prefix}.intermediate.dense.bias"),
            )?,
            output: linear_from_ws(
                ws,
                &format!("{prefix}.output.dense.weight"),
                &format!("{prefix}.output.dense.bias"),
            )?,
            layer_norm: layer_norm_from_ws(
                ws,
                &format!("{prefix}.output.LayerNorm.weight"),
                &format!("{prefix}.output.LayerNorm.bias"),
                config.layer_norm_eps,
            )?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let x = self.intermediate.forward(hidden_states).map_err(te)?;
        let x = gelu_tensor(&x)?;
        let x = self.output.forward(&x).map_err(te)?;
        let residual = hidden_states.add(&x).map_err(te)?;
        self.layer_norm.forward(&residual).map_err(te)
    }
}

struct BertLayer {
    attention: Attention,
    feed_forward: FeedForward,
}

impl BertLayer {
    fn load(ws: &WeightSet, layer_idx: usize, config: &Config) -> Result<Self> {
        let prefix = format!("bert.encoder.layer.{layer_idx}");
        Ok(Self {
            attention: Attention::load(ws, &prefix, config)?,
            feed_forward: FeedForward::load(ws, &prefix, config)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &[u32]) -> Result<Tensor> {
        let x = self.attention.forward(hidden_states, attention_mask)?;
        self.feed_forward.forward(&x)
    }
}

struct BertForTokenClassification {
    embeddings: Embeddings,
    layers: Vec<BertLayer>,
    classifier: Linear,
    id2label: Vec<String>,
}

impl BertForTokenClassification {
    fn load(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join(PUNCT_CONFIG_FILE);
        let config = Config::from_json(&config_path)?;
        let weights_path = model_dir.join("model.safetensors");
        if !weights_path.is_file() {
            bail!("missing {}", weights_path.display());
        }
        let ws = load_safetensors_file(&weights_path)
            .map_err(|e| anyhow::anyhow!("load punct weights: {e}"))?;

        let embeddings = Embeddings::load(&ws, &config)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(BertLayer::load(&ws, i, &config)?);
        }
        let classifier = linear_from_ws(&ws, "classifier.weight", "classifier.bias")?;
        if classifier.out_dim() != config.num_labels {
            bail!(
                "classifier out {} != num_labels {}",
                classifier.out_dim(),
                config.num_labels
            );
        }
        let id2label = load_id2label(&config_path)?;
        Ok(Self {
            embeddings,
            layers,
            classifier,
            id2label,
        })
    }

    /// Logits `[seq, num_labels]`.
    fn forward(
        &self,
        input_ids: &[u32],
        token_type_ids: &[u32],
        attention_mask: &[u32],
    ) -> Result<Tensor> {
        let mut x = self.embeddings.forward(input_ids, token_type_ids)?;
        for layer in &self.layers {
            x = layer.forward(&x, attention_mask)?;
        }
        self.classifier.forward(&x).map_err(te)
    }
}

/// Tensor-native punctuation restorer (product path for roadmap 37.4).
pub struct TensorPunctuator {
    model: BertForTokenClassification,
    tokenizer: Tokenizer,
}

impl TensorPunctuator {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
        let model = BertForTokenClassification::load(model_dir)?;
        Ok(Self { model, tokenizer })
    }

    pub fn restore(&self, text: &str) -> String {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return text.to_string();
        }
        match self.restore_inner(trimmed) {
            Ok(out) => out,
            Err(_) => text.to_string(),
        }
    }

    fn restore_inner(&self, text: &str) -> Result<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Ok(text.to_string());
        }

        let encoding = self.tokenizer.encode(text, true);
        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        let token_type_ids = vec![0u32; ids.len()];

        let logits = self.model.forward(ids, &token_type_ids, mask)?;
        // logits [seq, num_labels]
        let seq = logits.dim(0).map_err(te)?;
        let nlab = logits.dim(1).map_err(te)?;
        let flat = logits.as_slice();
        let argmax_per_token: Vec<usize> = (0..seq)
            .map(|t| argmax(&flat[t * nlab..(t + 1) * nlab]))
            .collect();
        let labels = first_subword_labels(encoding.get_word_ids(), &argmax_per_token, words.len());

        let mut out = String::new();
        for (word, &lid) in words.iter().zip(labels.iter()) {
            let label = self
                .model
                .id2label
                .get(lid)
                .map(String::as_str)
                .unwrap_or("LOWER_O");
            let processed = process_token(word, label);
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(&processed);
        }
        Ok(out.trim().to_string())
    }
}

// ── helpers ──────────────────────────────────────────────────────────────

fn te(e: crate::error::TensorError) -> anyhow::Error {
    anyhow::anyhow!(e)
}

fn require_f32(ws: &WeightSet, name: &str) -> Result<Tensor> {
    ws.get_f32(name)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing weight {name}"))
}

fn check_emb_shape(t: &Tensor, rows: usize, cols: usize, what: &str) -> Result<()> {
    if t.rank() != 2 || t.dim(0).map_err(te)? != rows || t.dim(1).map_err(te)? != cols {
        bail!(
            "{what} embeddings expected [{rows}, {cols}], got {:?}",
            t.dims()
        );
    }
    Ok(())
}

fn linear_from_ws(ws: &WeightSet, w_name: &str, b_name: &str) -> Result<Linear> {
    let w = require_f32(ws, w_name)?;
    let b = require_f32(ws, b_name)?;
    if w.rank() != 2 {
        bail!("{w_name}: expected rank-2, got {:?}", w.dims());
    }
    Linear::new(w, Some(b)).map_err(te)
}

fn layer_norm_from_ws(ws: &WeightSet, w_name: &str, b_name: &str, eps: f32) -> Result<LayerNorm> {
    let w = require_f32(ws, w_name)?;
    let b = require_f32(ws, b_name)?;
    LayerNorm::new(w, b, eps).map_err(te)
}

fn gather_rows(table: &Tensor, ids: &[u32]) -> Result<Tensor> {
    let rows = table.dim(0).map_err(te)?;
    let h = table.dim(1).map_err(te)?;
    let src = table.as_slice();
    let mut out = vec![0.0f32; ids.len() * h];
    for (i, &id) in ids.iter().enumerate() {
        let id = id as usize;
        if id >= rows {
            bail!("embedding index {id} >= rows {rows}");
        }
        out[i * h..(i + 1) * h].copy_from_slice(&src[id * h..(id + 1) * h]);
    }
    Tensor::from_vec(out, vec![ids.len(), h], Device::Cpu).map_err(te)
}

fn add_inplace(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
    }
}

/// PyTorch / candle `gelu` (erf form): `0.5 * x * (1 + erf(x / sqrt(2)))`.
#[inline]
fn gelu(x: f32) -> f32 {
    let x64 = x as f64;
    let cdf = 0.5 * (1.0 + erf_approx(x64 * std::f64::consts::FRAC_1_SQRT_2));
    (x64 * cdf) as f32
}

/// A&S 7.1.26 approximation; max |err| ~1.5e-7 — enough for label argmax parity.
#[inline]
fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

fn gelu_tensor(x: &Tensor) -> Result<Tensor> {
    let data: Vec<f32> = x.as_slice().iter().map(|&v| gelu(v)).collect();
    Tensor::from_vec(data, x.dims().to_vec(), x.device()).map_err(te)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::default_model_dir;

    #[test]
    fn gelu_zero_is_zero() {
        assert!(gelu(0.0).abs() < 1e-7);
    }

    #[test]
    #[ignore = "requires punct model at ~/.rupunct/punct or ~/.gigastt/models/punct"]
    fn test_restore_expected_phrases() {
        let dir = default_model_dir();
        let punct = TensorPunctuator::load(Path::new(&dir)).expect("load punct model");
        let cases: &[(&str, &str)] = &[
            (
                "привет меня зовут анна сколько будет стоить шестьдесят тысяч тенге",
                "Привет меня зовут Анна, Сколько будет стоить шестьдесят тысяч тенге.",
            ),
            (
                "завтра я поеду в москву на встречу с иваном",
                "Завтра я поеду в Москву на встречу с Иваном.",
            ),
            (
                "который сейчас час не могли бы вы подсказать",
                "Который сейчас час не могли бы вы подсказать.",
            ),
            (
                "в россии столица москва а великий новгород это древний город",
                "В России столица Москва, а Великий Новгород — это древний город.",
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(&punct.restore(input), expected, "restore({input:?})");
        }
    }
}
