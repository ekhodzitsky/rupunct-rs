use super::*;
use crate::fixture::{TinySpec, write_tiny_model};
use std::path::Path;

#[test]
fn load_tiny_weights_has_classifier() {
    let tmp = tempfile::tempdir().unwrap();
    write_tiny_model(tmp.path(), &TinySpec::default()).unwrap();
    let path = tmp.path().join(crate::MODEL_FILE);
    let ws = load_safetensors_file(&path).unwrap();
    assert!(ws.get_f32("classifier.bias").is_some());
    assert!(ws.get_f32("missing").is_none());
    let emb = ws
        .get_f32("bert.embeddings.word_embeddings.weight")
        .unwrap();
    assert!(emb.elem_count() >= 1024);
}

#[test]
fn load_missing_file_errors() {
    assert!(load_safetensors_file(Path::new("/no/such/file.safetensors")).is_err());
}

#[test]
fn load_rejects_garbage() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("bad.safetensors");
    std::fs::write(&path, b"not a safetensors file").unwrap();
    assert!(load_safetensors_file(&path).is_err());
}

#[test]
fn load_rejects_non_f32_named_weight() {
    use crate::st::{WriteTensor, serialize};
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("w.safetensors");
    let ser = serialize(&[WriteTensor::I32 {
        name: "classifier.weight".into(),
        shape: vec![1],
        data: vec![1],
    }])
    .unwrap();
    std::fs::write(&path, ser).unwrap();
    let err = load_safetensors_file(&path).unwrap_err();
    assert!(err.to_string().contains("unsupported dtype"), "{err}");
}

#[test]
fn load_q8_0_dequants_to_f32() {
    use crate::q8;
    use crate::st::{WriteTensor, serialize};
    let mut w = vec![0.0f32; 128 * 32];
    for (i, x) in w.iter_mut().enumerate() {
        *x = (i as f32).sin() * 0.05;
    }
    let shape = vec![128usize, 32];
    let wire = q8::quant(&w, &shape).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("q.safetensors");
    let ser = serialize(&[
        WriteTensor::U8 {
            name: "classifier.weight.q8_0".into(),
            shape: vec![wire.len()],
            data: wire,
        },
        WriteTensor::I64 {
            name: "classifier.weight.shape".into(),
            shape: vec![2],
            data: vec![128, 32],
        },
        WriteTensor::F32 {
            name: "classifier.bias".into(),
            shape: vec![2],
            data: vec![0.1, -0.2],
        },
    ])
    .unwrap();
    std::fs::write(&path, ser).unwrap();
    let ws = load_safetensors_file(&path).unwrap();
    let q = ws.get_q8("classifier.weight").unwrap();
    let got = q.dequant_f32().unwrap();
    assert_eq!(got.len(), w.len());
    let mut max_err = 0.0f32;
    for (a, b) in w.iter().zip(got.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    assert!(max_err < 0.01, "{max_err}");
    assert!(ws.get_f32("classifier.bias").is_some());
    let layer = crate::linear::Linear::from_q8(q.clone(), None).unwrap();
    let x = crate::tensor::Tensor::from_vec(vec![0.25; 32], vec![1, 32]).unwrap();
    assert_eq!(layer.forward(&x).unwrap().dims(), &[1, 128]);
    let mut set = WeightSet::default();
    set.q8.insert(
        "classifier.weight".into(),
        ws.get_q8("classifier.weight").unwrap().clone(),
    );
    set.f32.insert(
        "classifier.bias".into(),
        crate::tensor::Tensor::from_vec(vec![0.0; 128], vec![128]).unwrap(),
    );
    let via_ws = crate::ws::linear_from_ws(&set, "classifier.weight", "classifier.bias").unwrap();
    assert_eq!(via_ws.out_dim(), 128);
}

#[test]
fn load_q8_rejects_missing_shape() {
    use crate::st::{WriteTensor, serialize};
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("q.safetensors");
    let ser = serialize(&[WriteTensor::U8 {
        name: "classifier.weight.q8_0".into(),
        shape: vec![34],
        data: vec![0u8; 34],
    }])
    .unwrap();
    std::fs::write(&path, ser).unwrap();
    let err = load_safetensors_file(&path).unwrap_err();
    assert!(err.to_string().contains("missing"), "{err}");
}

#[test]
fn load_q8_rejects_f32_and_packed() {
    use crate::q8;
    use crate::st::{WriteTensor, serialize};
    let w = vec![0.0f32; 128 * 32];
    let shape = vec![128usize, 32];
    let wire = q8::quant(&w, &shape).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("q.safetensors");
    let ser = serialize(&[
        WriteTensor::F32 {
            name: "classifier.weight".into(),
            shape: shape.clone(),
            data: w,
        },
        WriteTensor::U8 {
            name: "classifier.weight.q8_0".into(),
            shape: vec![wire.len()],
            data: wire,
        },
        WriteTensor::I64 {
            name: "classifier.weight.shape".into(),
            shape: vec![2],
            data: vec![128, 32],
        },
    ])
    .unwrap();
    std::fs::write(&path, ser).unwrap();
    let err = load_safetensors_file(&path).unwrap_err();
    assert!(err.to_string().contains("both F32 and q8_0"), "{err}");
}

#[test]
fn load_q8_rejects_non_u8() {
    use crate::st::{WriteTensor, serialize};
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("q.safetensors");
    let ser = serialize(&[
        WriteTensor::I32 {
            name: "classifier.weight.q8_0".into(),
            shape: vec![1],
            data: vec![1],
        },
        WriteTensor::I64 {
            name: "classifier.weight.shape".into(),
            shape: vec![2],
            data: vec![1, 1],
        },
    ])
    .unwrap();
    std::fs::write(&path, ser).unwrap();
    let err = load_safetensors_file(&path).unwrap_err();
    assert!(err.to_string().contains("q8_0 must be U8"), "{err}");
}

#[test]
fn load_q8_rejects_bad_wire() {
    use crate::st::{WriteTensor, serialize};
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("q.safetensors");
    let ser = serialize(&[
        WriteTensor::U8 {
            name: "classifier.weight.q8_0".into(),
            shape: vec![3],
            data: vec![1, 2, 3],
        },
        WriteTensor::I64 {
            name: "classifier.weight.shape".into(),
            shape: vec![2],
            data: vec![128, 32],
        },
    ])
    .unwrap();
    std::fs::write(&path, ser).unwrap();
    assert!(load_safetensors_file(&path).is_err());
}

#[test]
fn load_skips_integer_non_weight_tensor() {
    use crate::st::{WriteTensor, serialize};
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("ids.safetensors");
    let ser = serialize(&[WriteTensor::I64 {
        name: "bert.embeddings.position_ids".into(),
        shape: vec![1],
        data: vec![0],
    }])
    .unwrap();
    std::fs::write(&path, ser).unwrap();
    let ws = load_safetensors_file(&path).unwrap();
    assert!(ws.get_f32("bert.embeddings.position_ids").is_none());
}
