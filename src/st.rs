//! HuggingFace safetensors v0.4 layout: 8-byte LE header length, JSON
//! header, raw little-endian payloads. No `safetensors` crate.

use crate::error::{TensorError, TensorResult};

const MAX_HEADER: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Dtype {
    F32,
    I32,
    I64,
    U8,
    Other,
}

#[derive(Debug, Clone)]
pub(crate) struct TensorDesc {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Absolute byte offset into the file.
    pub offset: usize,
    pub nbytes: usize,
}

pub(crate) fn parse_index(bytes: &[u8]) -> TensorResult<Vec<TensorDesc>> {
    if bytes.len() < 8 {
        return Err(TensorError::msg("safetensors: truncated header size"));
    }
    let mut len_buf = [0u8; 8];
    len_buf.copy_from_slice(&bytes[..8]);
    let header_len = u64::from_le_bytes(len_buf);
    if header_len > MAX_HEADER {
        return Err(TensorError::msg("safetensors: header too large"));
    }
    let header_len = header_len as usize;
    let header_end = 8usize
        .checked_add(header_len)
        .ok_or_else(|| TensorError::msg("safetensors: header overflow"))?;
    if header_end > bytes.len() {
        return Err(TensorError::msg("safetensors: truncated header"));
    }
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..header_end])
        .map_err(|e| TensorError::msg(format!("safetensors header JSON: {e}")))?;
    let obj = header
        .as_object()
        .ok_or_else(|| TensorError::msg("safetensors header is not an object"))?;
    let mut out = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for (name, spec) in obj {
        if name == "__metadata__" {
            continue;
        }
        if !seen.insert(name.as_str()) {
            return Err(TensorError::msg(format!(
                "safetensors: duplicate tensor '{name}'"
            )));
        }
        out.push(parse_tensor(name, spec, header_end, bytes.len())?);
    }
    check_no_overlap(&out)?;
    Ok(out)
}

fn check_no_overlap(descs: &[TensorDesc]) -> TensorResult<()> {
    let mut ranges: Vec<(usize, usize, &str)> = descs
        .iter()
        .filter(|d| d.nbytes > 0)
        .map(|d| (d.offset, d.offset + d.nbytes, d.name.as_str()))
        .collect();
    ranges.sort_by_key(|r| r.0);
    for pair in ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(TensorError::msg(format!(
                "safetensors: overlapping tensors '{}' and '{}'",
                pair[0].2, pair[1].2
            )));
        }
    }
    Ok(())
}

fn parse_tensor(
    name: &str,
    spec: &serde_json::Value,
    data_base: usize,
    file_len: usize,
) -> TensorResult<TensorDesc> {
    let spec = spec
        .as_object()
        .ok_or_else(|| TensorError::msg(format!("{name}: tensor spec is not an object")))?;
    let dtype = match spec.get("dtype").and_then(|v| v.as_str()) {
        Some("F32") => Dtype::F32,
        Some("I32") => Dtype::I32,
        Some("I64") => Dtype::I64,
        Some("U8") => Dtype::U8,
        Some(_) => Dtype::Other,
        None => {
            return Err(TensorError::msg(format!("{name}: missing dtype")));
        }
    };
    let shape = parse_shape(name, spec.get("shape"))?;
    let offsets = spec
        .get("data_offsets")
        .and_then(|v| v.as_array())
        .ok_or_else(|| TensorError::msg(format!("{name}: missing data_offsets")))?;
    if offsets.len() != 2 {
        return Err(TensorError::msg(format!(
            "{name}: data_offsets must be [begin, end]"
        )));
    }
    let begin = json_usize(&offsets[0], name, "data_offsets[0]")?;
    let end = json_usize(&offsets[1], name, "data_offsets[1]")?;
    if end < begin {
        return Err(TensorError::msg(format!(
            "{name}: data_offsets end < begin"
        )));
    }
    let nbytes = end - begin;
    let offset = data_base
        .checked_add(begin)
        .ok_or_else(|| TensorError::msg(format!("{name}: offset overflow")))?;
    let abs_end = offset
        .checked_add(nbytes)
        .ok_or_else(|| TensorError::msg(format!("{name}: payload overflow")))?;
    if abs_end > file_len {
        return Err(TensorError::msg(format!("{name}: payload OOB")));
    }
    if dtype == Dtype::F32 || dtype == Dtype::I32 || dtype == Dtype::I64 || dtype == Dtype::U8 {
        let width = match dtype {
            Dtype::I64 => 8usize,
            Dtype::U8 => 1,
            _ => 4,
        };
        let nelem = shape_product(&shape)?;
        let expect = nelem
            .checked_mul(width)
            .ok_or_else(|| TensorError::msg(format!("{name}: size overflow")))?;
        if expect != nbytes {
            return Err(TensorError::msg(format!(
                "{name}: byte len {nbytes} != {expect}"
            )));
        }
    }
    Ok(TensorDesc {
        name: name.to_string(),
        dtype,
        shape,
        offset,
        nbytes,
    })
}

fn parse_shape(name: &str, v: Option<&serde_json::Value>) -> TensorResult<Vec<usize>> {
    let arr = v
        .and_then(|x| x.as_array())
        .ok_or_else(|| TensorError::msg(format!("{name}: missing shape")))?;
    arr.iter().map(|d| json_usize(d, name, "shape")).collect()
}

fn json_usize(v: &serde_json::Value, name: &str, what: &str) -> TensorResult<usize> {
    v.as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| TensorError::msg(format!("{name}: {what} is not an integer")))
}

fn shape_product(shape: &[usize]) -> TensorResult<usize> {
    shape.iter().try_fold(1usize, |acc, &d| {
        acc.checked_mul(d)
            .ok_or_else(|| TensorError::msg("safetensors: shape overflow"))
    })
}

#[cfg(test)]
#[derive(Debug)]
pub(crate) enum WriteTensor {
    F32 {
        name: String,
        shape: Vec<usize>,
        data: Vec<f32>,
    },
    I32 {
        name: String,
        shape: Vec<usize>,
        data: Vec<i32>,
    },
    I64 {
        name: String,
        shape: Vec<usize>,
        data: Vec<i64>,
    },
    U8 {
        name: String,
        shape: Vec<usize>,
        data: Vec<u8>,
    },
}

#[cfg(test)]
pub(crate) fn serialize(tensors: &[WriteTensor]) -> TensorResult<Vec<u8>> {
    let mut items: Vec<(String, &'static str, Vec<usize>, Vec<u8>)> = tensors
        .iter()
        .map(|t| match t {
            WriteTensor::F32 { name, shape, data } => (
                name.clone(),
                "F32",
                shape.clone(),
                data.iter().flat_map(|x| x.to_le_bytes()).collect(),
            ),
            WriteTensor::I32 { name, shape, data } => (
                name.clone(),
                "I32",
                shape.clone(),
                data.iter().flat_map(|x| x.to_le_bytes()).collect(),
            ),
            WriteTensor::I64 { name, shape, data } => (
                name.clone(),
                "I64",
                shape.clone(),
                data.iter().flat_map(|x| x.to_le_bytes()).collect(),
            ),
            WriteTensor::U8 { name, shape, data } => {
                (name.clone(), "U8", shape.clone(), data.clone())
            }
        })
        .collect();
    items.sort_by(|a, b| a.0.cmp(&b.0));

    let mut header = serde_json::Map::new();
    let mut body = Vec::new();
    let mut cursor = 0usize;
    for (name, dtype, shape, bytes) in &items {
        let end = cursor
            .checked_add(bytes.len())
            .ok_or_else(|| TensorError::msg("safetensors: write overflow"))?;
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [cursor, end]
            }),
        );
        body.extend_from_slice(bytes);
        cursor = end;
    }
    let mut header_bytes = serde_json::to_vec(&serde_json::Value::Object(header))
        .map_err(|e| TensorError::msg(format!("safetensors header encode: {e}")))?;
    while !header_bytes.len().is_multiple_of(8) {
        header_bytes.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + header_bytes.len() + body.len());
    let n = u64::try_from(header_bytes.len())
        .map_err(|_| TensorError::msg("safetensors: header length does not fit u64"))?;
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&body);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_f32() {
        let bytes = serialize(&[WriteTensor::F32 {
            name: "w".into(),
            shape: vec![2],
            data: vec![1.0, 2.0],
        }])
        .unwrap();
        let idx = parse_index(&bytes).unwrap();
        assert_eq!(idx.len(), 1);
        assert_eq!(idx[0].dtype, Dtype::F32);
        assert_eq!(idx[0].shape, vec![2]);
        assert_eq!(idx[0].nbytes, 8);
        let sl = &bytes[idx[0].offset..idx[0].offset + 8];
        assert_eq!(f32::from_le_bytes(sl[..4].try_into().unwrap()), 1.0);
    }

    #[test]
    fn rejects_truncated_and_non_object() {
        assert!(parse_index(&[1, 2, 3]).is_err());
        let mut buf = 2u64.to_le_bytes().to_vec();
        buf.extend_from_slice(b"[]");
        assert!(parse_index(&buf).is_err());
        let mut buf = (MAX_HEADER + 1).to_le_bytes().to_vec();
        buf.extend_from_slice(&[0u8; 8]);
        assert!(parse_index(&buf).is_err());
        let mut buf = 4u64.to_le_bytes().to_vec();
        buf.extend_from_slice(b"{");
        assert!(parse_index(&buf).is_err());
    }

    #[test]
    fn skips_metadata_and_rejects_bad_specs() {
        let mut header =
            br#"{"__metadata__":{"k":"v"},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#
                .to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        let idx = parse_index(&bytes).unwrap();
        assert_eq!(idx.len(), 1);

        let mut header = br#"{"w":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#.to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&[0, 0]);
        let idx = parse_index(&bytes).unwrap();
        assert_eq!(idx[0].dtype, Dtype::Other);

        for spec in [
            br#"{"w":1}"# as &[u8],
            br#"{"w":{"shape":[1],"data_offsets":[0,4]}}"#,
            br#"{"w":{"dtype":"F32","data_offsets":[0,4]}}"#,
            br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0]}}"#,
            br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}}"#,
            br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,8]}}"#,
            br#"{"w":{"dtype":"F32","shape":["x"],"data_offsets":[0,4]}}"#,
        ] {
            let mut header = spec.to_vec();
            while !header.len().is_multiple_of(8) {
                header.push(b' ');
            }
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(&header);
            bytes.extend_from_slice(&[0u8; 16]);
            assert!(
                parse_index(&bytes).is_err(),
                "{}",
                String::from_utf8_lossy(spec)
            );
        }
    }

    #[test]
    fn write_integers() {
        let bytes = serialize(&[
            WriteTensor::I32 {
                name: "i".into(),
                shape: vec![1],
                data: vec![7],
            },
            WriteTensor::I64 {
                name: "j".into(),
                shape: vec![1],
                data: vec![-1],
            },
        ])
        .unwrap();
        let idx = parse_index(&bytes).unwrap();
        assert_eq!(idx.len(), 2);
        assert!(idx.iter().any(|t| t.dtype == Dtype::I32));
        assert!(idx.iter().any(|t| t.dtype == Dtype::I64));
    }

    #[test]
    fn rejects_overlap() {
        let mut header =
            br#"{"a":{"dtype":"I32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"I32","shape":[2],"data_offsets":[4,12]}}"#
                .to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&[0u8; 16]);
        let err = parse_index(&bytes).unwrap_err();
        assert!(err.to_string().contains("overlapping"), "{err}");
    }

    #[test]
    fn rejects_integer_size_mismatch() {
        let mut header = br#"{"a":{"dtype":"I32","shape":[2],"data_offsets":[0,4]}}"#.to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&[0u8; 8]);
        let err = parse_index(&bytes).unwrap_err();
        assert!(err.to_string().contains("byte len"), "{err}");
    }
}
