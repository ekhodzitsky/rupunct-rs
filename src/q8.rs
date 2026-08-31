//! GGML Q8_0: f16 scale + 32×i8 per block. Last dim is padded to 32 on disk.

use std::cell::RefCell;
use std::sync::Arc;

use crate::error::{TensorError, TensorResult};
use crate::tensor::MappedFile;

pub(crate) const QK8_0: usize = 32;
pub(crate) const BLOCK_Q8_0_BYTES: usize = 34;
const TILE_N: usize = 128;
#[cfg(test)]
const MIN_ELEMENTS: usize = 4096;
#[cfg(test)]
const QMAX: f32 = 127.0;

thread_local! {
    static TILE: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

#[derive(Debug, Clone)]
enum Store {
    Mmap {
        map: Arc<MappedFile>,
        offset: usize,
        nbytes: usize,
    },
    #[cfg(test)]
    Heap(Arc<Vec<u8>>),
}

#[derive(Debug, Clone)]
pub(crate) struct Q8Weight {
    store: Store,
    shape: Vec<usize>,
}

impl Q8Weight {
    pub(crate) fn new(
        map: Arc<MappedFile>,
        offset: usize,
        nbytes: usize,
        shape: Vec<usize>,
    ) -> TensorResult<Self> {
        let expect = packed_bytes(&shape)?;
        if nbytes != expect {
            return Err(TensorError::msg(format!("q8_0 wire {nbytes} != {expect}")));
        }
        let end = offset
            .checked_add(nbytes)
            .ok_or_else(|| TensorError::shape("q8_0 range overflow"))?;
        if end > map.len() {
            return Err(TensorError::shape("q8_0 range OOB"));
        }
        Ok(Self {
            store: Store::Mmap {
                map,
                offset,
                nbytes,
            },
            shape,
        })
    }

    #[cfg(test)]
    pub(crate) fn from_wire(wire: Vec<u8>, shape: Vec<usize>) -> TensorResult<Self> {
        let expect = packed_bytes(&shape)?;
        if wire.len() != expect {
            return Err(TensorError::msg(format!(
                "q8_0 wire {} != {expect}",
                wire.len()
            )));
        }
        Ok(Self {
            store: Store::Heap(Arc::new(wire)),
            shape,
        })
    }

    pub(crate) fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn wire(&self) -> &[u8] {
        match &self.store {
            Store::Mmap {
                map,
                offset,
                nbytes,
            } => &map.as_bytes()[*offset..*offset + *nbytes],
            #[cfg(test)]
            Store::Heap(v) => v.as_slice(),
        }
    }

    #[cfg(test)]
    pub(crate) fn dequant_f32(&self) -> TensorResult<Vec<f32>> {
        dequant(self.wire(), &self.shape)
    }

    pub(crate) fn gather_rows(&self, ids: &[u32], out: &mut [f32]) -> TensorResult<()> {
        if self.shape.len() != 2 {
            return Err(TensorError::shape("q8 gather needs rank-2"));
        }
        let rows = self.shape[0];
        let h = self.shape[1];
        let n = ids
            .len()
            .checked_mul(h)
            .ok_or_else(|| TensorError::shape("q8 gather overflow"))?;
        if out.len() != n {
            return Err(TensorError::shape("q8 gather out len"));
        }
        let wire = self.wire();
        for (i, &id) in ids.iter().enumerate() {
            let id = id as usize;
            if id >= rows {
                return Err(TensorError::shape(format!(
                    "embedding index {id} >= rows {rows}"
                )));
            }
            dequant_rows(wire, rows, h, id, 1, &mut out[i * h..(i + 1) * h])?;
        }
        Ok(())
    }

    pub(crate) fn matmul_nt(&self, m: usize, x: &[f32], out: &mut [f32]) -> TensorResult<()> {
        if self.shape.len() != 2 {
            return Err(TensorError::shape("q8 matmul needs rank-2 weight"));
        }
        let n = self.shape[0];
        let k = self.shape[1];
        let mk = m
            .checked_mul(k)
            .ok_or_else(|| TensorError::shape("m*k overflow"))?;
        let mn = m
            .checked_mul(n)
            .ok_or_else(|| TensorError::shape("m*n overflow"))?;
        if x.len() != mk || out.len() != mn {
            return Err(TensorError::shape("q8 matmul x/out size"));
        }
        let wire = self.wire();
        // Padded last dim (e.g. hidden=312): one GEMM. Multiples of 32 keep L2 tiles.
        let tile = if k.is_multiple_of(QK8_0) { TILE_N } else { n };
        TILE.with(|slot| -> TensorResult<()> {
            let mut buf = slot.borrow_mut();
            let mut n0 = 0usize;
            while n0 < n {
                let nt = (n - n0).min(tile);
                let need = nt
                    .checked_mul(k)
                    .ok_or_else(|| TensorError::shape("tile overflow"))?;
                if buf.len() < need {
                    buf.resize(need, 0.0);
                }
                dequant_rows(wire, n, k, n0, nt, &mut buf[..need])?;
                crate::blas::gemm_nt_weight_panel(m, k, n, n0, nt, x, &buf[..need], out);
                n0 += nt;
            }
            Ok(())
        })
    }
}

fn pad_k(k: usize) -> usize {
    k.div_ceil(QK8_0).saturating_mul(QK8_0)
}

fn packed_bytes(shape: &[usize]) -> TensorResult<usize> {
    let Some(&k) = shape.last() else {
        return Err(TensorError::shape("q8 empty shape"));
    };
    if k == 0 {
        return Err(TensorError::shape("q8 last dim 0"));
    }
    let n = numel(shape)?;
    let rows = n / k;
    let padded = rows
        .checked_mul(pad_k(k))
        .ok_or_else(|| TensorError::shape("q8 padded overflow"))?;
    padded
        .checked_div(QK8_0)
        .and_then(|b| b.checked_mul(BLOCK_Q8_0_BYTES))
        .ok_or_else(|| TensorError::shape("q8 packed overflow"))
}

#[cfg(test)]
pub(crate) fn should_quant(name: &str, shape: &[usize]) -> bool {
    if name.contains("LayerNorm") {
        return false;
    }
    let n = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d));
    matches!(n, Some(n) if n >= MIN_ELEMENTS && shape.len() >= 2)
}

#[cfg(test)]
pub(crate) fn quant(w: &[f32], shape: &[usize]) -> TensorResult<Vec<u8>> {
    let n = numel(shape)?;
    if w.len() != n {
        return Err(TensorError::shape("q8_0 quant len != shape"));
    }
    let k = *shape.last().unwrap_or(&0);
    if k == 0 {
        return Err(TensorError::shape("q8 last dim 0"));
    }
    let rows = n / k;
    let k_pad = pad_k(k);
    let mut padded = vec![0.0f32; rows * k_pad];
    for r in 0..rows {
        padded[r * k_pad..r * k_pad + k].copy_from_slice(&w[r * k..(r + 1) * k]);
    }
    let n_blocks = padded.len() / QK8_0;
    let mut out = vec![0u8; n_blocks * BLOCK_Q8_0_BYTES];
    for i in 0..n_blocks {
        let xs = &padded[i * QK8_0..(i + 1) * QK8_0];
        let mut amax = 0.0f32;
        for &x in xs {
            amax = amax.max(x.abs());
        }
        let d = if amax > 0.0 { amax / QMAX } else { 0.0 };
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let off = i * BLOCK_Q8_0_BYTES;
        out[off..off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        for j in 0..QK8_0 {
            out[off + 2 + j] = (xs[j] * id).round() as i8 as u8;
        }
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) fn dequant(wire: &[u8], shape: &[usize]) -> TensorResult<Vec<f32>> {
    let n = numel(shape)?;
    let mut out = vec![0.0f32; n];
    dequant_into(wire, shape, &mut out)?;
    Ok(out)
}

#[cfg(test)]
fn dequant_into(wire: &[u8], shape: &[usize], out: &mut [f32]) -> TensorResult<()> {
    let n = numel(shape)?;
    if out.len() != n || wire.len() != packed_bytes(shape)? {
        return Err(TensorError::msg("q8_0 dequant size"));
    }
    match shape {
        [rows, k] => dequant_rows(wire, *rows, *k, 0, *rows, out),
        [k] => dequant_rows(wire, 1, *k, 0, 1, out),
        _ => Err(TensorError::shape("q8 dequant rank")),
    }
}

fn dequant_rows(
    wire: &[u8],
    n: usize,
    k: usize,
    n0: usize,
    nt: usize,
    out: &mut [f32],
) -> TensorResult<()> {
    if k == 0 || n0.saturating_add(nt) > n || out.len() != nt.saturating_mul(k) {
        return Err(TensorError::shape("q8 tile"));
    }
    let k_pad = pad_k(k);
    let bpr = k_pad / QK8_0;
    let row_bytes = bpr * BLOCK_Q8_0_BYTES;
    let full = k / QK8_0;
    let rem = k % QK8_0;
    for r in 0..nt {
        let src = (n0 + r) * row_bytes;
        let dst = r * k;
        for bi in 0..full {
            dequant_block(
                &wire[src + bi * BLOCK_Q8_0_BYTES..src + (bi + 1) * BLOCK_Q8_0_BYTES],
                &mut out[dst + bi * QK8_0..dst + (bi + 1) * QK8_0],
            );
        }
        if rem != 0 {
            let mut tail = [0.0f32; QK8_0];
            dequant_block(
                &wire[src + full * BLOCK_Q8_0_BYTES..src + (full + 1) * BLOCK_Q8_0_BYTES],
                &mut tail,
            );
            out[dst + full * QK8_0..dst + k].copy_from_slice(&tail[..rem]);
        }
    }
    Ok(())
}

#[inline(always)]
fn dequant_block(block: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: callers pass a 34-byte block and a 32-wide `out` slice.
        unsafe {
            dequant_qs_neon(block.as_ptr().add(2) as *const i8, d, out.as_mut_ptr());
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for j in 0..QK8_0 {
            out[j] = (block[2 + j] as i8) as f32 * d;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dequant_qs_neon(qs: *const i8, d: f32, dst: *mut f32) {
    use core::arch::aarch64::*;
    // SAFETY: `qs` is 32 i8; `dst` is 32 f32. Unaligned NEON load/store is OK.
    unsafe {
        let dv = vdupq_n_f32(d);
        let i8v0 = vld1q_s8(qs);
        let i8v1 = vld1q_s8(qs.add(16));
        let s = [
            vmovl_s8(vget_low_s8(i8v0)),
            vmovl_s8(vget_high_s8(i8v0)),
            vmovl_s8(vget_low_s8(i8v1)),
            vmovl_s8(vget_high_s8(i8v1)),
        ];
        for (i, si) in s.iter().enumerate() {
            vst1q_f32(
                dst.add(i * 8),
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(*si))), dv),
            );
            vst1q_f32(
                dst.add(i * 8 + 4),
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(*si))), dv),
            );
        }
    }
}

fn numel(shape: &[usize]) -> TensorResult<usize> {
    shape
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))
        .ok_or_else(|| TensorError::shape("q8 shape overflow"))
}

#[cfg(test)]
fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | u16::from(mant != 0);
    }
    let exp = exp - 127 + 15;
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        return sign;
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

fn f16_to_f32(h: u16) -> f32 {
    let s = u32::from(h >> 15);
    let e = u32::from((h >> 10) & 0x1f);
    let m = u32::from(h & 0x3ff);
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            let mut m = m;
            let mut e: i32 = -14;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (s << 31) | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if e == 31 {
        (s << 31) | 0x7f80_0000 | (m << 13)
    } else {
        (s << 31) | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
#[path = "q8_tests.rs"]
mod tests;
