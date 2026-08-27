use super::*;

#[test]
fn test_capitalize_python_semantics() {
    assert_eq!(capitalize("привет"), "Привет");
    assert_eq!(capitalize("ПРИВЕТ"), "Привет");
    assert_eq!(capitalize("пРиВеТ"), "Привет");
    assert_eq!(capitalize(""), "");
    assert_eq!(capitalize("a"), "A");
}

#[test]
fn test_process_token_lower_modes() {
    assert_eq!(process_token("слово", "LOWER_O"), "слово");
    assert_eq!(process_token("слово", "LOWER_PERIOD"), "слово.");
    assert_eq!(process_token("слово", "LOWER_COMMA"), "слово,");
    assert_eq!(process_token("слово", "LOWER_QUESTION"), "слово?");
    assert_eq!(process_token("слово", "LOWER_VOSKL"), "слово!");
    assert_eq!(process_token("слово", "LOWER_DVOETOCHIE"), "слово:");
    assert_eq!(process_token("слово", "LOWER_PERIODCOMMA"), "слово;");
    assert_eq!(process_token("слово", "LOWER_DEFIS"), "слово-");
    assert_eq!(process_token("слово", "LOWER_MNOGOTOCHIE"), "слово...");
    assert_eq!(process_token("слово", "LOWER_QUESTIONVOSKL"), "слово?!");
}

#[test]
fn test_process_token_upper_capitalizes_first_lowercases_rest() {
    assert_eq!(process_token("анна", "UPPER_O"), "Анна");
    assert_eq!(process_token("анна", "UPPER_COMMA"), "Анна,");
    assert_eq!(process_token("ПРИВЕТ", "UPPER_PERIOD"), "Привет.");
}

#[test]
fn test_process_token_upper_total_uppercases_all() {
    assert_eq!(process_token("ооо", "UPPER_TOTAL_O"), "ООО");
    assert_eq!(process_token("ссср", "UPPER_TOTAL_PERIOD"), "СССР.");
    assert_eq!(process_token("ооо", "UPPER_TOTAL_COMMA"), "ООО,");
}

#[test]
fn test_process_token_tire_spacing_quirk() {
    assert_eq!(process_token("это", "LOWER_TIRE"), "это—");
    assert_eq!(process_token("это", "UPPER_TIRE"), "Это —");
    assert_eq!(process_token("это", "UPPER_TOTAL_TIRE"), "ЭТО —");
}

#[test]
fn test_process_token_unknown_label_is_identity() {
    assert_eq!(process_token("слово", "GARBAGE"), "слово");
    assert_eq!(process_token("слово", "LOWER_BOGUS"), "слово");
}

#[test]
fn test_first_subword_labels_picks_first_subtoken() {
    let word_ids = vec![None, Some(0), Some(0), Some(1), None];
    let argmax = vec![0, 3, 9, 7, 0];
    let labels = first_subword_labels(&word_ids, &argmax, 2, 31);
    assert_eq!(labels, vec![3, 7]);
}

#[test]
fn test_first_subword_labels_missing_word_defaults_lower_o() {
    let word_ids = vec![None, Some(0), None];
    let argmax = vec![0, 5, 0];
    let labels = first_subword_labels(&word_ids, &argmax, 2, 31);
    assert_eq!(labels, vec![5, 31]);
}

#[test]
fn test_lower_o_index() {
    let labels = vec!["UPPER_O".into(), "LOWER_O".into()];
    assert_eq!(lower_o_index(&labels).unwrap(), 1);
    assert!(lower_o_index(&["UPPER_O".into()]).is_err());
}

#[test]
fn test_argmax_returns_index_of_max() {
    assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
    assert_eq!(argmax(&[5.0, 1.0, 2.0]), 0);
    assert_eq!(argmax(&[1.0, 1.0, 3.0]), 2);
    assert_eq!(argmax(&[]), 0);
}

#[test]
fn test_load_id2label_parses_contiguous_map() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg = tmp.path().join("config.json");
    std::fs::write(
        &cfg,
        r#"{"id2label": {"0": "UPPER_PERIOD", "1": "LOWER_PERIOD", "2": "UPPER_TOTAL_PERIOD"}}"#,
    )
    .unwrap();
    let labels = load_id2label(&cfg).expect("parse");
    assert_eq!(
        labels,
        vec!["UPPER_PERIOD", "LOWER_PERIOD", "UPPER_TOTAL_PERIOD"]
    );
}

#[test]
fn test_load_id2label_rejects_gap() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg = tmp.path().join("config.json");
    std::fs::write(&cfg, r#"{"id2label": {"0": "A", "2": "C"}}"#).unwrap();
    assert!(load_id2label(&cfg).is_err());
}

#[test]
fn test_load_id2label_rejects_bad_json_and_types() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg = tmp.path().join("config.json");
    std::fs::write(&cfg, "{ not json").unwrap();
    assert!(load_id2label(&cfg).is_err());
    std::fs::write(&cfg, r#"{"id2label": {"x": "A"}}"#).unwrap();
    assert!(load_id2label(&cfg).is_err());
    std::fs::write(&cfg, r#"{"id2label": {"0": 1}}"#).unwrap();
    assert!(load_id2label(&cfg).is_err());
    std::fs::write(&cfg, r#"{}"#).unwrap();
    assert!(load_id2label(&cfg).is_err());
    assert!(load_id2label(&tmp.path().join("missing.json")).is_err());
}
