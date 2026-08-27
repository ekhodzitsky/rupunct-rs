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
    use safetensors::Dtype;
    use safetensors::tensor::TensorView;
    use std::collections::BTreeMap;
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("w.safetensors");
    let bytes = 1i32.to_le_bytes();
    let view = TensorView::new(Dtype::I32, vec![1], &bytes).unwrap();
    let mut map = BTreeMap::new();
    map.insert("classifier.weight".to_string(), view);
    let ser = safetensors::serialize(&map, None).unwrap();
    std::fs::write(&path, ser).unwrap();
    let err = load_safetensors_file(&path).unwrap_err();
    assert!(err.to_string().contains("unsupported dtype"), "{err}");
}

#[test]
fn load_skips_integer_non_weight_tensor() {
    use safetensors::Dtype;
    use safetensors::tensor::TensorView;
    use std::collections::BTreeMap;
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("ids.safetensors");
    let bytes = 0i64.to_le_bytes();
    let view = TensorView::new(Dtype::I64, vec![1], &bytes).unwrap();
    let mut map = BTreeMap::new();
    map.insert("bert.embeddings.position_ids".to_string(), view);
    let ser = safetensors::serialize(&map, None).unwrap();
    std::fs::write(&path, ser).unwrap();
    let ws = load_safetensors_file(&path).unwrap();
    assert!(ws.get_f32("bert.embeddings.position_ids").is_none());
}
