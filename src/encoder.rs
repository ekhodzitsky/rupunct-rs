//! BERT encoder layer: self-attention + feed-forward.

use crate::config::Config;
use crate::error::{Error, Result};
use crate::gelu::gelu_tensor;
use crate::layer_norm::LayerNorm;
use crate::linear::Linear;
use crate::matmul::{matmul_f32_nn_attn_strided, matmul_f32_nt_strided};
use crate::tensor::Tensor;
use crate::weights::WeightSet;
use crate::ws::{layer_norm_from_ws, linear_from_ws};

pub(crate) struct Attention {
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
    pub(crate) fn load(ws: &WeightSet, prefix: &str, config: &Config) -> Result<Self> {
        let attn = Self {
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
        };
        if attn.query.in_dim() != config.hidden_size
            || attn.query.out_dim() != config.hidden_size
            || attn.key.in_dim() != config.hidden_size
            || attn.key.out_dim() != config.hidden_size
            || attn.value.in_dim() != config.hidden_size
            || attn.value.out_dim() != config.hidden_size
            || attn.dense.in_dim() != config.hidden_size
            || attn.dense.out_dim() != config.hidden_size
        {
            return Err(Error::config(format!(
                "{prefix}: Q/K/V/dense must be [{0}, {0}]",
                config.hidden_size
            )));
        }
        Ok(attn)
    }

    pub(crate) fn forward(&self, hidden_states: &Tensor, attention_mask: &[u32]) -> Result<Tensor> {
        let seq = hidden_states.dim(0)?;
        if attention_mask.len() != seq {
            return Err(Error::model(format!(
                "attention_mask len {} != seq {seq}",
                attention_mask.len()
            )));
        }
        let q = self.query.forward(hidden_states)?;
        let k = self.key.forward(hidden_states)?;
        let v = self.value.forward(hidden_states)?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut context = vec![0.0f32; seq * self.hidden];
        let qs = q.as_slice();
        let ks = k.as_slice();
        let vs = v.as_slice();
        let mut scores = vec![0.0f32; seq * seq];

        for h in 0..self.n_heads {
            let off = h * self.head_dim;
            matmul_f32_nt_strided(seq, self.head_dim, qs, ks, self.hidden, off, &mut scores)?;
            for s in scores.iter_mut() {
                *s *= scale;
            }
            mask_keys(&mut scores, seq, attention_mask);
            softmax_rows(&mut scores, seq);
            matmul_f32_nn_attn_strided(
                seq,
                self.head_dim,
                &scores,
                vs,
                &mut context,
                self.hidden,
                off,
            )?;
        }

        let attn = Tensor::from_vec(context, vec![seq, self.hidden])?;
        let attn = self.dense.forward(&attn)?;
        let residual = hidden_states.add(&attn)?;
        Ok(self.layer_norm.forward(&residual)?)
    }
}

pub(crate) fn mask_keys(scores: &mut [f32], seq: usize, attention_mask: &[u32]) {
    for s in 0..seq {
        if attention_mask.get(s).copied().unwrap_or(0) == 0 {
            for t in 0..seq {
                scores[t * seq + s] += -1.0e9;
            }
        }
    }
}

pub(crate) fn softmax_rows(scores: &mut [f32], seq: usize) {
    for t in 0..seq {
        let row_start = t * seq;
        let mut max_v = f32::NEG_INFINITY;
        for s in 0..seq {
            max_v = max_v.max(scores[row_start + s]);
        }
        let mut sum = 0.0f32;
        for s in 0..seq {
            let e = (scores[row_start + s] - max_v).exp();
            scores[row_start + s] = e;
            sum += e;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for s in 0..seq {
                scores[row_start + s] *= inv;
            }
        }
    }
}

struct FeedForward {
    intermediate: Linear,
    output: Linear,
    layer_norm: LayerNorm,
}

impl FeedForward {
    fn load(ws: &WeightSet, prefix: &str, config: &Config) -> Result<Self> {
        let ff = Self {
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
        };
        if ff.intermediate.out_dim() != config.intermediate_size {
            return Err(crate::error::Error::config(format!(
                "intermediate out {} != intermediate_size {}",
                ff.intermediate.out_dim(),
                config.intermediate_size
            )));
        }
        Ok(ff)
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let x = self.intermediate.forward(hidden_states)?;
        let x = gelu_tensor(&x)?;
        let x = self.output.forward(&x)?;
        let residual = hidden_states.add(&x)?;
        Ok(self.layer_norm.forward(&residual)?)
    }
}

pub(crate) struct BertLayer {
    attention: Attention,
    feed_forward: FeedForward,
}

impl BertLayer {
    pub(crate) fn load(ws: &WeightSet, layer_idx: usize, config: &Config) -> Result<Self> {
        let prefix = format!("bert.encoder.layer.{layer_idx}");
        Ok(Self {
            attention: Attention::load(ws, &prefix, config)?,
            feed_forward: FeedForward::load(ws, &prefix, config)?,
        })
    }

    pub(crate) fn forward(&self, hidden_states: &Tensor, attention_mask: &[u32]) -> Result<Tensor> {
        let x = self.attention.forward(hidden_states, attention_mask)?;
        self.feed_forward.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_and_softmax() {
        let mut scores = vec![1.0f32, 2.0, 3.0, 4.0];
        mask_keys(&mut scores, 2, &[1, 0]);
        assert!(scores[1] < -1000.0);
        assert!(scores[3] < -1000.0);
        softmax_rows(&mut scores, 2);
        let s0 = scores[0] + scores[1];
        assert!((s0 - 1.0).abs() < 1e-5);
    }
}
