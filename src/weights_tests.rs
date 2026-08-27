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
