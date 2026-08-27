use std::sync::Arc;

use memmap2::Mmap;

use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Result, TensorError};
use crate::matmul::{matmul_f32, matmul_f32_bt};
use crate::shape::Shape;

/// Contiguous row-major F32 storage: heap or mmap view (product weights).
#[derive(Debug, Clone)]
enum F32Store {
    Heap(Arc<Vec<f32>>),
    /// LE F32 bytes in a shared read-only mmap.
    Mmap {
        map: Arc<Mmap>,
        offset: usize,
        /// Element count (not bytes).
        nelem: usize,
    },
}

/// Contiguous row-major F32 tensor on CPU.
///
/// Heap storage is [`Arc`]-shared so [`reshape`] / [`Clone`] are O(1).
/// Product large F32 weights use mmap views (reclaimable clean pages).
#[derive(Debug, Clone)]
pub struct Tensor {
    shape: Shape,
    data: F32Store,
    device: Device,
}

impl Tensor {
    pub fn from_vec(data: Vec<f32>, shape: impl Into<Shape>, device: Device) -> Result<Self> {
        let shape = shape.into();
        let n = shape.elem_count()?;
        if data.len() != n {
            return Err(TensorError::shape(format!(
                "data len {} != shape {} ({n} elems)",
                data.len(),
                shape
            )));
        }
        if !device.is_cpu() {
            return Err(TensorError::Device("only CPU supported".into()));
        }
        Ok(Self {
            shape,
            data: F32Store::Heap(Arc::new(data)),
            device,
        })
    }

    /// Zero-copy F32 view into a shared mmap (LE floats). Registers mmap for
    /// post-request `madvise` reclaim.
    pub fn from_mmap_f32(
        map: Arc<Mmap>,
        offset: usize,
        shape: impl Into<Shape>,
        device: Device,
    ) -> Result<Self> {
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
        if !device.is_cpu() {
            return Err(TensorError::Device("only CPU supported".into()));
        }
        Ok(Self {
            shape,
            data: F32Store::Mmap { map, offset, nelem },
            device,
        })
    }

    pub fn zeros(shape: impl Into<Shape>, device: Device) -> Result<Self> {
        let shape = shape.into();
        let n = shape.elem_count()?;
        Self::from_vec(vec![0.0; n], shape, device)
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn dim(&self, i: usize) -> Result<usize> {
        self.shape.dim(i)
    }

    pub fn dtype(&self) -> DType {
        DType::F32
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn elem_count(&self) -> usize {
        match &self.data {
            F32Store::Heap(v) => v.len(),
            F32Store::Mmap { nelem, .. } => *nelem,
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        match &self.data {
            F32Store::Heap(v) => v.as_slice(),
            F32Store::Mmap {
                map, offset, nelem, ..
            } => {
                let bytes = &map[*offset..*offset + nelem * 4];
                // SAFETY: product safetensors F32 is aligned LE; offset from
                // tensor payload is 4-byte aligned; nelem floats.
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, *nelem) }
            }
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        self.ensure_heap_mut();
        match &mut self.data {
            F32Store::Heap(v) => Arc::make_mut(v).as_mut_slice(),
            F32Store::Mmap { .. } => unreachable!("ensure_heap_mut"),
        }
    }

    fn ensure_heap_mut(&mut self) {
        if let F32Store::Mmap { map, offset, nelem } = &self.data {
            let sl = {
                let bytes = &map[*offset..*offset + *nelem * 4];
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, *nelem) }
            };
            self.data = F32Store::Heap(Arc::new(sl.to_vec()));
        }
    }

    pub fn into_vec(self) -> Vec<f32> {
        match self.data {
            F32Store::Heap(v) => match Arc::try_unwrap(v) {
                Ok(v) => v,
                Err(arc) => (*arc).clone(),
            },
            F32Store::Mmap { .. } => self.as_slice().to_vec(),
        }
    }

    /// 2D matmul: `self [m,k] @ rhs [k,n] -> [m,n]`.
    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
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
        matmul_f32(m, k, n, self.as_slice(), rhs.as_slice(), &mut out)?;
        Tensor::from_vec(out, vec![m, n], self.device)
    }

    /// `self [m,k] @ weight^T` where weight is `[n,k]` (Linear layout).
    pub fn matmul_weight_t(&self, weight: &Tensor) -> Result<Tensor> {
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
        Tensor::from_vec(out, vec![m, n], self.device)
    }

    /// Elementwise `self + other` (same shape).
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        if self.dims() != other.dims() {
            return Err(TensorError::shape(format!(
                "add shape {} vs {}",
                self.shape, other.shape
            )));
        }
        let a = self.as_slice();
        let b = other.as_slice();
        let mut data = Vec::with_capacity(a.len());
        // Manual loop is faster than zip+collect on large residual maps.
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
        Tensor::from_vec(data, self.shape.clone(), self.device)
    }

    /// In-place `self += other` (copy-on-write if storage is shared).
    pub fn add_assign(&mut self, other: &Tensor) -> Result<()> {
        if self.dims() != other.dims() {
            return Err(TensorError::shape(format!(
                "add_assign shape {} vs {}",
                self.shape, other.shape
            )));
        }
        let b = other.as_slice();
        let a = self.as_mut_slice();
        for (x, &y) in a.iter_mut().zip(b.iter()) {
            *x += y;
        }
        Ok(())
    }

    /// Broadcast-add a 1D bias `[n]` onto last dim of rank-2 `[m,n]`.
    pub fn add_bias_2d(&self, bias: &Tensor) -> Result<Tensor> {
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
        let mut data = self.as_slice().to_vec();
        let b = bias.as_slice();
        for i in 0..m {
            for j in 0..n {
                data[i * n + j] += b[j];
            }
        }
        Tensor::from_vec(data, self.shape.clone(), self.device)
    }

    /// Zero-copy reshape (shared storage) when element count matches.
    pub fn reshape(&self, shape: impl Into<Shape>) -> Result<Tensor> {
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
            device: self.device,
        })
    }

    /// Elementwise SiLU: `x * sigmoid(x)`.
    pub fn silu(&self) -> Result<Tensor> {
        let src = self.as_slice();
        let mut data = Vec::with_capacity(src.len());
        for &x in src {
            data.push(x * (1.0 / (1.0 + (-x).exp())));
        }
        Tensor::from_vec(data, self.shape.clone(), self.device)
    }

    /// Elementwise ReLU.
    pub fn relu(&self) -> Result<Tensor> {
        let data: Vec<f32> = self.as_slice().iter().map(|&x| x.max(0.0)).collect();
        Tensor::from_vec(data, self.shape.clone(), self.device)
    }

    /// Softmax over the last dimension (rank-2).
    pub fn softmax_last(&self) -> Result<Tensor> {
        if self.rank() != 2 {
            return Err(TensorError::shape("softmax_last: need rank-2"));
        }
        let rows = self.dim(0)?;
        let cols = self.dim(1)?;
        let mut out = vec![0.0f32; self.elem_count()];
        let src = self.as_slice();
        for r in 0..rows {
            let row = &src[r * cols..(r + 1) * cols];
            let mut max_v = f32::NEG_INFINITY;
            for &x in row {
                max_v = max_v.max(x);
            }
            let mut sum = 0.0f32;
            for c in 0..cols {
                let e = (row[c] - max_v).exp();
                out[r * cols + c] = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            for c in 0..cols {
                out[r * cols + c] *= inv;
            }
        }
        Tensor::from_vec(out, self.shape.clone(), self.device)
    }

    /// Log-softmax over the last dimension (rank-2: rows independent).
    pub fn log_softmax_last(&self) -> Result<Tensor> {
        if self.rank() != 2 {
            return Err(TensorError::shape("log_softmax_last: need rank-2"));
        }
        let rows = self.dim(0)?;
        let cols = self.dim(1)?;
        let mut out = vec![0.0f32; self.elem_count()];
        let src = self.as_slice();
        for r in 0..rows {
            let row = &src[r * cols..(r + 1) * cols];
            let mut max_v = f32::NEG_INFINITY;
            for &x in row {
                max_v = max_v.max(x);
            }
            let mut sum = 0.0f32;
            for &x in row {
                sum += (x - max_v).exp();
            }
            let log_sum = sum.ln() + max_v;
            for c in 0..cols {
                out[r * cols + c] = row[c] - log_sum;
            }
        }
        Tensor::from_vec(out, self.shape.clone(), self.device)
    }

    /// Elementwise multiply (same shape).
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        if self.dims() != other.dims() {
            return Err(TensorError::shape(format!(
                "mul shape {} vs {}",
                self.shape, other.shape
            )));
        }
        let a = self.as_slice();
        let b = other.as_slice();
        let mut data = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            data.push(a[i] * b[i]);
        }
        Tensor::from_vec(data, self.shape.clone(), self.device)
    }

    /// Elementwise sigmoid.
    pub fn sigmoid(&self) -> Result<Tensor> {
        let data: Vec<f32> = self
            .as_slice()
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        Tensor::from_vec(data, self.shape.clone(), self.device)
    }

    /// Elementwise tanh.
    pub fn tanh(&self) -> Result<Tensor> {
        let data: Vec<f32> = self.as_slice().iter().map(|&x| x.tanh()).collect();
        Tensor::from_vec(data, self.shape.clone(), self.device)
    }

    /// Narrow along `dim` starting at `start` with length `len` (contiguous only).
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Tensor> {
        if dim >= self.rank() {
            return Err(TensorError::shape(format!(
                "narrow dim {dim} rank {}",
                self.rank()
            )));
        }
        let d = self.dim(dim)?;
        if start + len > d {
            return Err(TensorError::shape(format!(
                "narrow start {start}+len {len} > dim {d}"
            )));
        }
        let src = self.as_slice();
        // Contiguous row-major slice for dim 0 only (rank-1 or rank-2 first dim).
        if dim == 0 {
            let stride: usize = self.dims()[1..].iter().product::<usize>().max(1);
            let off = start * stride;
            let n = len * stride;
            let mut shape = self.dims().to_vec();
            shape[0] = len;
            return Tensor::from_vec(src[off..off + n].to_vec(), shape, self.device);
        }
        if self.rank() == 2 && dim == 1 {
            let rows = self.dim(0)?;
            let cols = self.dim(1)?;
            let mut out = Vec::with_capacity(rows * len);
            for r in 0..rows {
                let base = r * cols + start;
                out.extend_from_slice(&src[base..base + len]);
            }
            return Tensor::from_vec(out, vec![rows, len], self.device);
        }
        Err(TensorError::msg(
            "narrow: only dim 0 (any rank contiguous) or dim 1 of rank-2 supported",
        ))
    }

    /// Row `i` of a rank-2 tensor as rank-1.
    pub fn row(&self, i: usize) -> Result<Tensor> {
        if self.rank() != 2 {
            return Err(TensorError::shape("row requires rank-2"));
        }
        self.narrow(0, i, 1)?.reshape(vec![self.dim(1)?])
    }

    /// Add a 1-D bias to a 1-D tensor (same length).
    pub fn add_1d(&self, bias: &Tensor) -> Result<Tensor> {
        if self.rank() != 1 || bias.rank() != 1 || self.dim(0)? != bias.dim(0)? {
            return Err(TensorError::shape("add_1d requires equal rank-1"));
        }
        self.add(bias)
    }

    /// Max absolute difference vs another tensor (same shape).
    pub fn max_abs_diff(&self, other: &Tensor) -> Result<f32> {
        if self.dims() != other.dims() {
            return Err(TensorError::shape("max_abs_diff shape mismatch"));
        }
        let mut m = 0.0f32;
        for (a, b) in self.as_slice().iter().zip(other.as_slice().iter()) {
            m = m.max((a - b).abs());
        }
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_softmax_sums_to_one() {
        let t =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0], vec![2, 3], Device::Cpu).unwrap();
        let y = t.log_softmax_last().unwrap();
        for r in 0..2 {
            let mut s = 0.0f32;
            for c in 0..3 {
                s += y.as_slice()[r * 3 + c].exp();
            }
            assert!((s - 1.0).abs() < 1e-5, "row {r} sum {s}");
        }
    }

    #[test]
    fn zeros_and_add() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::Cpu).unwrap();
        let z = Tensor::zeros(vec![2, 2], Device::Cpu).unwrap();
        let s = a.add(&z).unwrap();
        assert_eq!(s.as_slice(), a.as_slice());
    }

    #[test]
    fn reshape_is_zero_copy() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::Cpu).unwrap();
        let b = a.reshape(vec![4]).unwrap();
        // Same allocation: equal pointer after reshape.
        assert_eq!(a.as_slice().as_ptr(), b.as_slice().as_ptr());
        assert_eq!(b.dims(), &[4]);
    }
}
