use super::*;

#[test]
fn io_preserves_kind_and_path() {
    let err = Error::io(
        "/tmp/missing.json",
        io::Error::from(io::ErrorKind::NotFound),
    );
    assert_eq!(
        err,
        Error::Io {
            path: "/tmp/missing.json".to_string(),
            kind: io::ErrorKind::NotFound,
        }
    );
    let msg = err.to_string();
    assert!(msg.contains("missing.json"), "{msg}");
    assert!(
        msg.contains("not found") || msg.contains("NotFound") || msg.contains("os error"),
        "{msg}"
    );
}

#[test]
fn constructors_round_trip_display() {
    assert_eq!(
        Error::config("bad hidden_size").to_string(),
        "bad hidden_size"
    );
    assert_eq!(
        Error::tokenizer("unsupported model type").to_string(),
        "unsupported model type"
    );
    assert_eq!(
        Error::model("missing weight x").to_string(),
        "missing weight x"
    );
}

#[test]
fn tensor_error_maps_to_model() {
    let err = Error::from(TensorError::shape("rank"));
    assert_eq!(err, Error::Model("shape error: rank".to_string()));
    let err = Error::from(TensorError::msg("mmap"));
    assert_eq!(err, Error::Model("mmap".to_string()));
}
