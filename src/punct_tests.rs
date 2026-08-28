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
    let opts = crate::RestoreOpts {
        max_window_tokens: 8,
        overlap_tokens: 2,
        ..crate::RestoreOpts::default()
    };
    let batch = p.restore_batch_with(&["привет"], opts).unwrap();
    assert_eq!(batch.len(), 1);
}

#[test]
fn restore_windows_long_input() {
    let tmp = tempfile::tempdir().unwrap();
    let spec = TinySpec {
        max_pos: 8,
        bias_label: LABEL_UPPER_PERIOD,
        ..TinySpec::default()
    };
    write_tiny_model(tmp.path(), &spec).unwrap();
    let p = Punctuator::load(tmp.path()).unwrap();
    let words: Vec<String> = (0..20).map(|i| format!("w{i}")).collect();
    let text = words.join(" ");
    let out = p
        .restore_with(
            &text,
            crate::RestoreOpts {
                max_window_tokens: 8,
                overlap_tokens: 7,
                ..crate::RestoreOpts::default()
            },
        )
        .unwrap();
    let want: String = (0..20)
        .map(|i| format!("W{i}."))
        .collect::<Vec<_>>()
        .join(" ");
    assert_eq!(out, want);
}

#[test]
#[ignore = "requires RUPunct weights at $RUPUNCT_MODEL or ~/.rupunct/punct"]
fn test_restore_reference_string() {
    let Some(dir) = crate::model::live_model_dir() else {
        panic!("ignored test requires weights at $RUPUNCT_MODEL or ~/.rupunct/punct");
    };
    let punct = Punctuator::load(&dir).expect("load punct model");
    let out = punct
        .restore("привет меня зовут анна сколько будет стоить шестьдесят тысяч тенге")
        .expect("restore");
    assert_eq!(
        out,
        "Привет меня зовут Анна, Сколько будет стоить шестьдесят тысяч тенге."
    );
}

#[test]
fn restore_words_reports_label_and_score() {
    let tmp = tempfile::tempdir().unwrap();
    write_tiny_model(tmp.path(), &TinySpec::default()).unwrap();
    let p = Punctuator::load(tmp.path()).unwrap();
    let words = p.restore_words("привет мир").unwrap();
    assert_eq!(words.len(), 2);
    assert_eq!(words[0].word, "привет");
    assert_eq!(words[0].label, ID2LABEL[LABEL_LOWER_O]);
    assert_eq!(words[0].text, "привет");
    assert!(words[0].score > 0.5, "score {}", words[0].score);
}

#[test]
fn strip_punct_drops_isolated_punctuation() {
    let tmp = tempfile::tempdir().unwrap();
    write_tiny_model(tmp.path(), &TinySpec::default()).unwrap();
    let p = Punctuator::load(tmp.path()).unwrap();
    let opts = crate::RestoreOpts {
        strip_punct: true,
        ..crate::RestoreOpts::default()
    };
    assert_eq!(p.restore_with("привет, мир", opts).unwrap(), "привет мир");
    let words = p.restore_words_with("привет, мир", opts).unwrap();
    assert_eq!(
        words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>(),
        ["привет", "мир"]
    );
}

#[test]
fn lowercase_option_lowercases_classifier_input() {
    let tmp = tempfile::tempdir().unwrap();
    write_tiny_model(tmp.path(), &TinySpec::default()).unwrap();
    let p = Punctuator::load(tmp.path()).unwrap();
    let opts = crate::RestoreOpts {
        lowercase: true,
        ..crate::RestoreOpts::default()
    };
    assert_eq!(p.restore_with("Привет", opts).unwrap(), "привет");
}

#[test]
fn control_only_input_is_identity() {
    let tmp = tempfile::tempdir().unwrap();
    write_tiny_model(tmp.path(), &TinySpec::default()).unwrap();
    let p = Punctuator::load(tmp.path()).unwrap();
    assert_eq!(p.restore("  \u{0}  ").unwrap(), "  \u{0}  ");
}

#[test]
fn restore_rejects_oversized_input() {
    let tmp = tempfile::tempdir().unwrap();
    write_tiny_model(tmp.path(), &TinySpec::default()).unwrap();
    let p = Punctuator::load(tmp.path()).unwrap();
    let huge = "a".repeat(crate::limits::MAX_RESTORE_BYTES + 1);
    let err = p.restore(&huge).unwrap_err().to_string();
    assert!(err.contains("bytes"), "{err}");
}

#[test]
fn encoder_logits_depend_on_input() {
    let tmp = tempfile::tempdir().unwrap();
    let spec = TinySpec {
        peak_classifier_bias: false,
        ..TinySpec::default()
    };
    write_tiny_model(tmp.path(), &spec).unwrap();
    let p = Punctuator::load(tmp.path()).unwrap();
    let a = p.debug_logits("мир").unwrap();
    let b = p.debug_logits("как").unwrap();
    assert_ne!(a.as_slice(), b.as_slice());
    assert!(a.as_slice().iter().any(|x| x.abs() > 1e-8));
}

#[test]
#[ignore = "requires RUPunct weights at $RUPUNCT_MODEL or ~/.rupunct/punct"]
fn test_restore_short_greeting() {
    let Some(dir) = crate::model::live_model_dir() else {
        panic!("ignored test requires weights at $RUPUNCT_MODEL or ~/.rupunct/punct");
    };
    let punct = Punctuator::load(&dir).expect("load punct model");
    assert_eq!(
        punct.restore("привет как дела").unwrap(),
        "Привет, как дела?"
    );
    let opts = crate::RestoreOpts {
        strip_punct: true,
        lowercase: true,
        ..crate::RestoreOpts::default()
    };
    assert_eq!(
        punct.restore_with("привет, как дела", opts).unwrap(),
        "Привет, как дела?"
    );
    let words = punct.restore_words("привет как дела").unwrap();
    assert_eq!(words[0].text, "Привет,");
    assert!(words[0].score > 0.0);
}
