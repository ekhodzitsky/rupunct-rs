//! F32 safetensors load.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;
use safetensors::Dtype as StDtype;
use safetensors::tensor::SafeTensors;

use crate::error::{TensorError, TensorResult};
use crate::tensor::Tensor;

#[derive(Debug, Default)]
pub(crate) struct WeightSet {
    pub f32: HashMap<String, Tensor>,
}

impl WeightSet {
    pub(crate) fn get_f32(&self, name: &str) -> Option<&Tensor> {
        self.f32.get(name)
    }
}

fn open_mmap(path: &Path) -> TensorResult<Arc<Mmap>> {
    let file = std::fs::File::open(path)
        .map_err(|e| TensorError::msg(format!("open {}: {e}", path.display())))?;
    // SAFETY: read-only mapping; file is not truncated while mapped.
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| TensorError::msg(format!("mmap {}: {e}", path.display())))?;
    Ok(Arc::new(mmap))
}

pub(crate) fn load_safetensors_file(path: &Path) -> TensorResult<WeightSet> {
    let map = open_mmap(path)?;
    let bytes: &[u8] = &map;
    let st = SafeTensors::deserialize(bytes)
        .map_err(|e| TensorError::msg(format!("safetensors: {e}")))?;
    let base = bytes.as_ptr() as usize;
    let mut out = WeightSet::default();
    for name in st.names() {
        let view = st
            .tensor(name)
            .map_err(|e| TensorError::msg(format!("tensor {name}: {e}")))?;
        if view.dtype() != StDtype::F32 {
            let weight_like = name.ends_with(".weight") || name.ends_with(".bias");
            if weight_like {
                return Err(TensorError::msg(format!(
                    "{name}: unsupported dtype {:?} (F32 required)",
                    view.dtype()
                )));
            }
            continue;
        }
        let shape: Vec<usize> = view.shape().to_vec();
        let raw = view.data();
        if !raw.len().is_multiple_of(4) {
            return Err(TensorError::msg(format!(
                "{name}: F32 byte len {}",
                raw.len()
            )));
        }
        let nelem = raw.len() / 4;
        let offset = (raw.as_ptr() as usize).wrapping_sub(base);
        let t = if offset.is_multiple_of(4) && offset + raw.len() <= bytes.len() && nelem >= 1024 {
            Tensor::from_mmap_f32(Arc::clone(&map), offset, shape)?
        } else {
            let mut vals = Vec::with_capacity(nelem);
            for chunk in raw.chunks_exact(4) {
                vals.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Tensor::from_vec(vals, shape)?
        };
        out.f32.insert((*name).to_string(), t);
    }
    Ok(out)
}

#[cfg(test)]
#[path = "weights_tests.rs"]
mod tests;
