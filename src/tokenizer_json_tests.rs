use super::*;

#[test]
fn null_parts_are_ok() {
    assert!(matches!(parse_normalizer(None).unwrap(), Normalizer::None));
    assert!(!parse_pre_tokenizer(None).unwrap());
    assert_eq!(parse_post_processor(None).unwrap(), (None, None));
}

#[test]
fn post_processor_rejects_bad_kind_and_template() {
    let v = serde_json::json!({"type": "ByteLevel"});
    assert!(parse_post_processor(Some(v)).is_err());
    let v = serde_json::json!({
        "type": "TemplateProcessing",
        "single": [{"Sequence": {"id": "A", "type_id": 0}}],
        "special_tokens": {}
    });
    assert!(parse_post_processor(Some(v)).is_err());
    let v = serde_json::json!({
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
            {"Sequence": {"id": "B", "type_id": 0}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
        ],
        "special_tokens": {
            "[CLS]": {"ids": [2]},
            "[SEP]": {"ids": [3]}
        }
    });
    assert!(parse_post_processor(Some(v)).is_err());
}

#[test]
fn post_processor_rejects_bad_specials() {
    let v = serde_json::json!({
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "[CLS]", "type_id": 1}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
        ],
        "special_tokens": {
            "[CLS]": {"ids": [2]},
            "[SEP]": {"ids": [3]}
        }
    });
    assert!(parse_post_processor(Some(v)).is_err());
    let v = serde_json::json!({
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
        ],
        "special_tokens": {
            "[SEP]": {"ids": [3]}
        }
    });
    assert!(parse_post_processor(Some(v)).is_err());
    let v = serde_json::json!({
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
        ],
        "special_tokens": {
            "[CLS]": {"ids": [2, 9]},
            "[SEP]": {"ids": [3]}
        }
    });
    assert!(parse_post_processor(Some(v)).is_err());
}

#[test]
fn pre_tokenizer_missing_type() {
    let v = serde_json::json!({"foo": 1});
    assert!(parse_pre_tokenizer(Some(v)).is_err());
}

#[test]
fn from_value_reads_minimal_wordpiece() {
    let v = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 1, "content": "[UNK]", "single_word": false,
             "lstrip": false, "rstrip": false, "normalized": false}
        ],
        "normalizer": {"type": "BertNormalizer", "lowercase": false, "strip_accents": false},
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "model": {
            "type": "WordPiece",
            "unk_token": "[UNK]",
            "vocab": {"[UNK]": 1, "a": 2}
        }
    });
    let f = TokenizerFile::from_value(v).unwrap();
    assert_eq!(f.version.as_deref(), Some("1.0"));
    assert!(f.truncation.is_none());
    assert_eq!(f.added_tokens.len(), 1);
    assert_eq!(f.model.vocab.get("a"), Some(&2));
    assert!(matches!(
        parse_normalizer(f.normalizer),
        Ok(Normalizer::Bert { .. })
    ));
    assert!(parse_pre_tokenizer(f.pre_tokenizer).unwrap());
}

#[test]
fn from_value_rejects_bad_shapes() {
    assert!(TokenizerFile::from_value(serde_json::json!([])).is_err());
    assert!(TokenizerFile::from_value(serde_json::json!({"model": 1})).is_err());
    assert!(
        TokenizerFile::from_value(serde_json::json!({"model": {"type": "WordPiece"}})).is_err()
    );
    assert!(
        TokenizerFile::from_value(serde_json::json!({
            "model": {"type": "WordPiece", "vocab": {"a": "nope"}}
        }))
        .is_err()
    );
    assert!(parse_added_tokens_err());
}

fn parse_added_tokens_err() -> bool {
    TokenizerFile::from_value(serde_json::json!({
        "added_tokens": {"id": 1},
        "model": {"type": "WordPiece", "vocab": {"a": 1}}
    }))
    .is_err()
}
