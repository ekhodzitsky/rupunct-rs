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
