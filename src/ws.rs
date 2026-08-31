//! Safetensors helpers shared by BERT load.

use crate::error::{Error, Result};
use crate::layer_norm::LayerNorm;
use crate::linear::Linear;
use crate::q8::Q8Weight;
use crate::tensor::Tensor;
use crate::weights::WeightSet;

#[derive(Clone)]
pub(crate) enum EmbTable {
    F32(Tensor),
    Q8(Q8Weight),
}

pub(crate) fn require_f32(ws: &WeightSet, name: &str) -> Result<Tensor> {
    ws.get_f32(name)
        .cloned()
        .ok_or_else(|| Error::model(format!("missing weight {name}")))
}

pub(crate) fn check_emb_shape(t: &Tensor, rows: usize, cols: usize, what: &str) -> Result<()> {
    if t.rank() != 2 || t.dim(0)? != rows || t.dim(1)? != cols {
        return Err(Error::config(format!(
            "{what} embeddings expected [{rows}, {cols}], got {:?}",
            t.dims()
        )));
    }
    Ok(())
}

pub(crate) fn emb_from_ws(
    ws: &WeightSet,
    name: &str,
    rows: usize,
    cols: usize,
    what: &str,
) -> Result<EmbTable> {
    if let Some(q) = ws.get_q8(name) {
        if q.shape() != [rows, cols] {
            return Err(Error::config(format!(
                "{what} embeddings expected [{rows}, {cols}], got {:?}",
                q.shape()
            )));
        }
        return Ok(EmbTable::Q8(q.clone()));
    }
    let t = require_f32(ws, name)?;
    check_emb_shape(&t, rows, cols, what)?;
    Ok(EmbTable::F32(t))
}

pub(crate) fn linear_from_ws(ws: &WeightSet, w_name: &str, b_name: &str) -> Result<Linear> {
    let b = require_f32(ws, b_name)?;
    if let Some(q) = ws.get_q8(w_name) {
        return Ok(Linear::from_q8(q.clone(), Some(b))?);
    }
    let w = require_f32(ws, w_name)?;
    if w.rank() != 2 {
        return Err(Error::model(format!(
            "{w_name}: expected rank-2, got {:?}",
            w.dims()
        )));
    }
    Ok(Linear::new(w, Some(b))?)
}

pub(crate) fn layer_norm_from_ws(
    ws: &WeightSet,
    w_name: &str,
    b_name: &str,
    eps: f32,
) -> Result<LayerNorm> {
    let w = require_f32(ws, w_name)?;
    let b = require_f32(ws, b_name)?;
    Ok(LayerNorm::new(w, b, eps)?)
}

pub(crate) fn gather_rows(table: &Tensor, ids: &[u32]) -> Result<Tensor> {
    let rows = table.dim(0)?;
    let h = table.dim(1)?;
    let src = table.as_slice();
    let n = ids
        .len()
        .checked_mul(h)
        .ok_or_else(|| Error::model("gather overflow"))?;
    let mut out = vec![0.0f32; n];
    for (i, &id) in ids.iter().enumerate() {
        let id = id as usize;
        if id >= rows {
            return Err(Error::model(format!("embedding index {id} >= rows {rows}")));
        }
        out[i * h..(i + 1) * h].copy_from_slice(&src[id * h..(id + 1) * h]);
    }
    Ok(Tensor::from_vec(out, vec![ids.len(), h])?)
}

pub(crate) fn gather_emb(table: &EmbTable, ids: &[u32]) -> Result<Tensor> {
    match table {
        EmbTable::F32(t) => gather_rows(t, ids),
        EmbTable::Q8(q) => {
            let h = q.shape()[1];
            let n = ids
                .len()
                .checked_mul(h)
                .ok_or_else(|| Error::model("gather overflow"))?;
            let mut out = vec![0.0f32; n];
            q.gather_rows(ids, &mut out)?;
            Ok(Tensor::from_vec(out, vec![ids.len(), h])?)
        }
    }
}

pub(crate) fn add_inplace(dst: &mut [f32], src: &[f32]) {
    assert_eq!(dst.len(), src.len());
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
    }
}
