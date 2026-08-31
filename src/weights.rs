//! F32 / Q8_0 safetensors load (in-tree reader).
//!
//! Q8_0 MatMul weights (`{name}.q8_0` + `{name}.shape`) stay packed on the mmap.
//! Embeddings stay mmap F32. Linear dequants tiles into scratch for F32 GEMM.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::error::{TensorError, TensorResult};
use crate::limits::MAX_MODEL_BYTES;
use crate::q8::Q8Weight;
use crate::st::{self, Dtype, TensorDesc};
use crate::tensor::{MappedFile, Tensor};

#[derive(Debug, Default)]
pub(crate) struct WeightSet {
    pub f32: HashMap<String, Tensor>,
    pub q8: HashMap<String, Q8Weight>,
}

impl WeightSet {
    pub(crate) fn get_f32(&self, name: &str) -> Option<&Tensor> {
        self.f32.get(name)
    }

    pub(crate) fn get_q8(&self, name: &str) -> Option<&Q8Weight> {
        self.q8.get(name)
    }
}

pub(crate) fn load_safetensors_file(path: &Path) -> TensorResult<WeightSet> {
    let file = std::fs::File::open(path)
        .map_err(|e| TensorError::msg(format!("open {}: {e}", path.display())))?;
    let mapped = MappedFile::open(file)?;
    if mapped.len() as u64 > MAX_MODEL_BYTES {
        return Err(TensorError::msg(format!(
            "model larger than {MAX_MODEL_BYTES} bytes"
        )));
    }
    let bytes = mapped.as_bytes();
    let index = st::parse_index(bytes)?;
    let mut by_name = HashMap::new();
    for desc in &index {
        by_name.insert(desc.name.as_str(), desc);
    }
    let mut out = WeightSet::default();
    for desc in &index {
        if desc.dtype == Dtype::F32 {
            out.f32
                .insert(desc.name.clone(), load_f32(&mapped, bytes, desc)?);
            continue;
        }
        if desc.name.ends_with(".q8_0") || desc.name.ends_with(".shape") {
            continue;
        }
        let weight_like = desc.name.ends_with(".weight") || desc.name.ends_with(".bias");
        if weight_like {
            return Err(TensorError::msg(format!(
                "{}: unsupported dtype {:?} (F32 required)",
                desc.name, desc.dtype
            )));
        }
    }
    load_q8_weights(&mapped, bytes, &by_name, &mut out)?;
    Ok(out)
}

fn load_f32(mapped: &Arc<MappedFile>, bytes: &[u8], desc: &TensorDesc) -> TensorResult<Tensor> {
    if !desc.nbytes.is_multiple_of(4) {
        return Err(TensorError::msg(format!(
            "{}: F32 byte len {}",
            desc.name, desc.nbytes
        )));
    }
    let nelem = desc.nbytes / 4;
    if desc.offset.is_multiple_of(4) && desc.offset + desc.nbytes <= mapped.len() && nelem >= 1024 {
        Tensor::from_mmap_f32(Arc::clone(mapped), desc.offset, desc.shape.clone())
    } else {
        let raw = &bytes[desc.offset..desc.offset + desc.nbytes];
        let mut vals = Vec::with_capacity(nelem);
        for chunk in raw.chunks_exact(4) {
            vals.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Tensor::from_vec(vals, desc.shape.clone())
    }
}

fn load_q8_weights(
    mapped: &Arc<MappedFile>,
    bytes: &[u8],
    by_name: &HashMap<&str, &TensorDesc>,
    out: &mut WeightSet,
) -> TensorResult<()> {
    for desc in by_name.values() {
        let Some(base) = desc.name.strip_suffix(".q8_0") else {
            continue;
        };
        if desc.dtype != Dtype::U8 {
            return Err(TensorError::msg(format!("{}: q8_0 must be U8", desc.name)));
        }
        if out.f32.contains_key(base) || out.q8.contains_key(base) {
            return Err(TensorError::msg(format!(
                "{base}: both F32 and q8_0 present"
            )));
        }
        let shape_name = format!("{base}.shape");
        let shape_desc = by_name
            .get(shape_name.as_str())
            .ok_or_else(|| TensorError::msg(format!("{}: missing {shape_name}", desc.name)))?;
        let shape = read_i64_shape(bytes, shape_desc)?;
        out.q8.insert(
            base.to_string(),
            Q8Weight::new(Arc::clone(mapped), desc.offset, desc.nbytes, shape)?,
        );
    }
    Ok(())
}

fn read_i64_shape(bytes: &[u8], desc: &TensorDesc) -> TensorResult<Vec<usize>> {
    if desc.dtype != Dtype::I64 {
        return Err(TensorError::msg(format!(
            "{}: shape must be I64",
            desc.name
        )));
    }
    let raw = &bytes[desc.offset..desc.offset + desc.nbytes];
    let mut shape = Vec::with_capacity(desc.nbytes / 8);
    for chunk in raw.chunks_exact(8) {
        let n = i64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        let n = usize::try_from(n)
            .map_err(|_| TensorError::msg(format!("{}: negative dim", desc.name)))?;
        shape.push(n);
    }
    if shape.is_empty() {
        return Err(TensorError::msg(format!("{}: empty shape", desc.name)));
    }
    Ok(shape)
}

#[cfg(test)]
#[path = "weights_tests.rs"]
mod tests;
