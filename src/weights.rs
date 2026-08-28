//! F32 safetensors load (in-tree reader).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::error::{TensorError, TensorResult};
use crate::limits::MAX_MODEL_BYTES;
use crate::st::{self, Dtype};
use crate::tensor::{MappedFile, Tensor};

#[derive(Debug, Default)]
pub(crate) struct WeightSet {
    pub f32: HashMap<String, Tensor>,
}

impl WeightSet {
    pub(crate) fn get_f32(&self, name: &str) -> Option<&Tensor> {
        self.f32.get(name)
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
    let mut out = WeightSet::default();
    for desc in index {
        if desc.dtype != Dtype::F32 {
            let weight_like = desc.name.ends_with(".weight") || desc.name.ends_with(".bias");
            if weight_like {
                return Err(TensorError::msg(format!(
                    "{}: unsupported dtype {:?} (F32 required)",
                    desc.name, desc.dtype
                )));
            }
            continue;
        }
        if !desc.nbytes.is_multiple_of(4) {
            return Err(TensorError::msg(format!(
                "{}: F32 byte len {}",
                desc.name, desc.nbytes
            )));
        }
        let nelem = desc.nbytes / 4;
        let t = if desc.offset.is_multiple_of(4)
            && desc.offset + desc.nbytes <= mapped.len()
            && nelem >= 1024
        {
            Tensor::from_mmap_f32(Arc::clone(&mapped), desc.offset, desc.shape)?
        } else {
            let raw = &bytes[desc.offset..desc.offset + desc.nbytes];
            let mut vals = Vec::with_capacity(nelem);
            for chunk in raw.chunks_exact(4) {
                vals.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Tensor::from_vec(vals, desc.shape)?
        };
        out.f32.insert(desc.name, t);
    }
    Ok(out)
}

#[cfg(test)]
#[path = "weights_tests.rs"]
mod tests;
