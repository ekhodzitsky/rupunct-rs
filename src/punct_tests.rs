use super::*;
use crate::fixture::{
    ID2LABEL, LABEL_LOWER_O, LABEL_UPPER_PERIOD, TinySpec, expected_for_label, write_tiny_model,
};
use crate::model::{CONFIG_FILE, TOKENIZER_FILE};

#[test]
fn load_missing_dir_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let missing = tmp.path().join("does-not-exist");
    assert!(Punctuator::load(&missing).is_err());
}

#[test]
fn load_valid_config_invalid_tokenizer_errors() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(
        tmp.path().join(CONFIG_FILE),
        r#"{"id2label": {"0": "LOWER_O"}, "hidden_size": 2, "num_attention_heads": 1,
           "num_hidden_layers": 1, "intermediate_size": 2, "vocab_size": 3,
           "max_position_embeddings": 8, "type_vocab_size": 2, "layer_norm_eps": 1e-12}"#,
    )
    .unwrap();
    std::fs::write(tmp.path().join(TOKENIZER_FILE), "{ not valid json").unwrap();
    let err = match Punctuator::load(tmp.path()) {
        Ok(_) => panic!("malformed tokenizer must error"),
        Err(e) => e,
    };
    let full = err.to_string();
    assert!(
        full.contains("tokenizer") || full.contains("JSON"),
        "got {full}"
    );
}

#[test]
fn load_missing_weights_errors() {
    let tmp = tempfile::tempdir().unwrap();
    write_tiny_model(tmp.path(), &TinySpec::default()).unwrap();
    std::fs::remove_file(tmp.path().join(crate::MODEL_FILE)).unwrap();
    assert!(Punctuator::load(tmp.path()).is_err());
}

#[test]
fn restore_empty_is_identity() {
    let tmp = tempfile::tempdir().unwrap();
    write_tiny_model(tmp.path(), &TinySpec::default()).unwrap();
    let p = Punctuator::load(tmp.path()).unwrap();
    assert_eq!(p.restore("").unwrap(), "");
    assert_eq!(p.restore("   \n").unwrap(), "   \n");
}

#[test]
fn restore_lower_o_is_identity_words() {
    let tmp = tempfile::tempdir().unwrap();
    write_tiny_model(tmp.path(), &TinySpec::default()).unwrap();
    let p = Punctuator::load(tmp.path()).unwrap();
    let out = p.restore("привет мир").unwrap();
    assert_eq!(
        out,
        expected_for_label("привет мир", ID2LABEL[LABEL_LOWER_O])
    );
    assert_eq!(p.restore_lossy("привет мир"), out);
}

#[test]
fn restore_upper_period_applies_label() {
    let tmp = tempfile::tempdir().unwrap();
    let spec = TinySpec {
        bias_label: LABEL_UPPER_PERIOD,
        ..TinySpec::default()
    };
    write_tiny_model(tmp.path(), &spec).unwrap();
    let p = Punctuator::load(tmp.path()).unwrap();
    assert_eq!(p.restore("привет мир").unwrap(), "Привет. Мир.");
}

#[test]
fn restore_batch_and_lossy() {
    let tmp = tempfile::tempdir().unwrap();
    write_tiny_model(tmp.path(), &TinySpec::default()).unwrap();
    let p = Punctuator::load(tmp.path()).unwrap();
    let batch = p.restore_batch(&["привет", "мир", ""]).unwrap();
    assert_eq!(batch.len(), 3);
    assert_eq!(batch[2], "");
}

#[test]
fn restore_windows_long_input() {
    let tmp = tempfile::tempdir().unwrap();
    let spec = TinySpec {
        max_pos: 8,
        ..TinySpec::default()
    };
    write_tiny_model(tmp.path(), &spec).unwrap();
    let p = Punctuator::load(tmp.path()).unwrap();
    let words: Vec<String> = (0..20).map(|i| format!("w{i}")).collect();
    let text = words.join(" ");
    let out = p.restore(&text).unwrap();
    assert_eq!(out.split_whitespace().count(), 20);
}

#[test]
#[ignore = "requires RUPunct weights at $RUPUNCT_MODEL or ~/.rupunct/punct"]
fn test_restore_reference_string() {
    let dir = crate::model::live_model_dir().expect("live model");
    let punct = Punctuator::load(&dir).expect("load punct model");
    let out = punct
        .restore("привет меня зовут анна сколько будет стоить шестьдесят тысяч тенге")
        .expect("restore");
    assert_eq!(
        out,
        "Привет меня зовут Анна, Сколько будет стоить шестьдесят тысяч тенге."
    );
}
