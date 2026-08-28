use std::fs::File;
use std::sync::Arc;

use memmap2::Mmap;

use crate::error::{TensorError, TensorResult};
use crate::matmul::matmul_f32_bt;
use crate::shape::Shape;

/// Read-only mapping that keeps the `File` alive (Windows requires it).
#[derive(Debug)]
pub(crate) struct MappedFile {
    _file: File,
    map: Mmap,
}

impl MappedFile {
    pub(crate) fn open(file: File) -> TensorResult<Arc<Self>> {
        // SAFETY: read-only mapping; `_file` is kept so the fd is not closed
        // under the map (Windows). Another process truncating the file can
        // still SIGBUS — weights are treated as immutable after load.
        let map =
            unsafe { Mmap::map(&file) }.map_err(|e| TensorError::msg(format!("mmap: {e}")))?;
        Ok(Arc::new(Self { _file: file, map }))
    }

    pub(crate) fn len(&self) -> usize {
        self.map.len()
    }

    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.map
    }
}

/// Contiguous row-major F32 storage: heap or mmap view.
#[derive(Debug, Clone)]
enum F32Store {
    Heap(Arc<Vec<f32>>),
    Mmap {
        map: Arc<MappedFile>,
        offset: usize,
        nelem: usize,
    },
}

/// Contiguous row-major F32 tensor on CPU.
///
/// Heap storage is [`Arc`]-shared so [`Tensor::reshape`] / [`Clone`] are O(1).
/// Large F32 weights use mmap views (reclaimable clean pages).
#[derive(Debug, Clone)]
pub(crate) struct Tensor {
    shape: Shape,
    data: F32Store,
}

impl Tensor {
    pub(crate) fn from_vec(data: Vec<f32>, shape: impl Into<Shape>) -> TensorResult<Self> {
        let shape = shape.into();
        let n = shape.elem_count()?;
        if data.len() != n {
            return Err(TensorError::shape(format!(
                "data len {} != shape {} ({n} elems)",
                data.len(),
                shape
            )));
        }
        Ok(Self {
            shape,
            data: F32Store::Heap(Arc::new(data)),
        })
    }

    /// Zero-copy F32 view into a shared mmap (LE floats).
    pub(crate) fn from_mmap_f32(
        map: Arc<MappedFile>,
        offset: usize,
        shape: impl Into<Shape>,
    ) -> TensorResult<Self> {
        let shape = shape.into();
        let nelem = shape.elem_count()?;
        let nbytes = nelem
            .checked_mul(4)
            .ok_or_else(|| TensorError::shape("f32 mmap overflow"))?;
        if offset
            .checked_add(nbytes)
            .map(|e| e > map.len())
            .unwrap_or(true)
        {
            return Err(TensorError::shape("f32 mmap range OOB"));
        }
        Ok(Self {
            shape,
            data: F32Store::Mmap { map, offset, nelem },
        })
    }

    #[cfg(test)]
    pub(crate) fn zeros(shape: impl Into<Shape>) -> TensorResult<Self> {
        let shape = shape.into();
        let n = shape.elem_count()?;
        Self::from_vec(vec![0.0; n], shape)
    }

    #[cfg(test)]
    pub(crate) fn shape(&self) -> &Shape {
        &self.shape
    }

    pub(crate) fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    pub(crate) fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub(crate) fn dim(&self, i: usize) -> TensorResult<usize> {
        self.shape.dim(i)
    }

    pub(crate) fn elem_count(&self) -> usize {
        match &self.data {
            F32Store::Heap(v) => v.len(),
            F32Store::Mmap { nelem, .. } => *nelem,
        }
    }

    pub(crate) fn as_slice(&self) -> &[f32] {
        match &self.data {
            F32Store::Heap(v) => v.as_slice(),
            F32Store::Mmap { map, offset, nelem } => {
                let bytes = &map.as_bytes()[*offset..*offset + nelem * 4];
                // SAFETY: safetensors F32 payload is aligned LE; offset is 4-byte aligned.
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, *nelem) }
            }
        }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [f32] {
        self.ensure_heap_mut();
        match &mut self.data {
            F32Store::Heap(v) => Arc::make_mut(v).as_mut_slice(),
            F32Store::Mmap { .. } => unreachable!("ensure_heap_mut"),
        }
    }

    fn ensure_heap_mut(&mut self) {
        if let F32Store::Mmap { map, offset, nelem } = &self.data {
            let sl = {
                let bytes = &map.as_bytes()[*offset..*offset + *nelem * 4];
                // SAFETY: same as `as_slice` for the mmap view.
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, *nelem) }
            };
            self.data = F32Store::Heap(Arc::new(sl.to_vec()));
        }
    }

    /// 2D matmul: `self [m,k] @ rhs [k,n] -> [m,n]`.
    #[cfg(test)]
    pub(crate) fn matmul(&self, rhs: &Tensor) -> TensorResult<Tensor> {
        if self.rank() != 2 || rhs.rank() != 2 {
            return Err(TensorError::shape("matmul requires rank-2 tensors"));
        }
        let m = self.dim(0)?;
        let k = self.dim(1)?;
        let k2 = rhs.dim(0)?;
        let n = rhs.dim(1)?;
        if k != k2 {
            return Err(TensorError::shape(format!(
                "matmul inner mismatch {k} vs {k2}"
            )));
        }
        let mut out = vec![0.0f32; m * n];
        crate::matmul::matmul_f32(m, k, n, self.as_slice(), rhs.as_slice(), &mut out)?;
        Tensor::from_vec(out, vec![m, n])
    }

    /// `self [m,k] @ weight^T` where weight is `[n,k]` (Linear layout).
    pub(crate) fn matmul_weight_t(&self, weight: &Tensor) -> TensorResult<Tensor> {
        if self.rank() != 2 || weight.rank() != 2 {
            return Err(TensorError::shape("matmul_weight_t requires rank-2"));
        }
        let m = self.dim(0)?;
        let k = self.dim(1)?;
        let n = weight.dim(0)?;
        let k2 = weight.dim(1)?;
        if k != k2 {
            return Err(TensorError::shape(format!(
                "weight in_dim {k2} != x last {k}"
            )));
        }
        let mut out = vec![0.0f32; m * n];
        matmul_f32_bt(m, k, n, self.as_slice(), weight.as_slice(), &mut out)?;
        Tensor::from_vec(out, vec![m, n])
    }

    /// Elementwise `self + other` (same shape).
    pub(crate) fn add(&self, other: &Tensor) -> TensorResult<Tensor> {
        if self.dims() != other.dims() {
            return Err(TensorError::shape(format!(
                "add shape {} vs {}",
                self.shape, other.shape
            )));
        }
        let a = self.as_slice();
        let b = other.as_slice();
        let mut data = Vec::with_capacity(a.len());
        let n = a.len();
        let mut i = 0;
        while i + 4 <= n {
            data.push(a[i] + b[i]);
            data.push(a[i + 1] + b[i + 1]);
            data.push(a[i + 2] + b[i + 2]);
            data.push(a[i + 3] + b[i + 3]);
            i += 4;
        }
        while i < n {
            data.push(a[i] + b[i]);
            i += 1;
        }
        Tensor::from_vec(data, self.shape.clone())
    }

    /// Broadcast-add a 1D bias `[n]` onto last dim of rank-2 `[m,n]`.
    #[cfg(test)]
    pub(crate) fn add_bias_2d(&self, bias: &Tensor) -> TensorResult<Tensor> {
        let mut out = self.clone();
        out.add_bias_2d_inplace(bias)?;
        Ok(out)
    }

    pub(crate) fn add_bias_2d_inplace(&mut self, bias: &Tensor) -> TensorResult<()> {
        if self.rank() != 2 {
            return Err(TensorError::shape("add_bias_2d requires rank-2 input"));
        }
        if bias.rank() != 1 {
            return Err(TensorError::shape("bias must be rank-1"));
        }
        let m = self.dim(0)?;
        let n = self.dim(1)?;
        if bias.dim(0)? != n {
            return Err(TensorError::shape(format!(
                "bias len {} != n {n}",
                bias.dim(0)?
            )));
        }
        let b = bias.as_slice();
        let data = self.as_mut_slice();
        for i in 0..m {
            for j in 0..n {
                data[i * n + j] += b[j];
            }
        }
        Ok(())
    }

    /// Zero-copy reshape (shared storage) when element count matches.
    #[cfg(test)]
    pub(crate) fn reshape(&self, shape: impl Into<Shape>) -> TensorResult<Tensor> {
        let shape = shape.into();
        let n = shape.elem_count()?;
        if n != self.elem_count() {
            return Err(TensorError::shape(format!(
                "reshape {} -> {} ({} elems, have {})",
                self.shape,
                shape,
                n,
                self.elem_count()
            )));
        }
        Ok(Self {
            shape,
            data: self.data.clone(),
        })
    }
}

#[cfg(test)]
#[path = "tensor_tests.rs"]
mod tests;
