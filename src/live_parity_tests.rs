//! Replay `tests/fixtures/hf_parity_golden.json` against live weights.

use super::*;
use crate::RestoreOpts;
use crate::model::live_model_dir;
use std::path::Path;

fn golden_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hf_parity_golden.json")
}

fn opts_from_flags(flags: &[serde_json::Value]) -> RestoreOpts {
    let mut opts = RestoreOpts::default();
    let xs: Vec<&str> = flags.iter().filter_map(|v| v.as_str()).collect();
    let mut i = 0;
    while i < xs.len() {
        match xs[i] {
            "--strip-punct" => opts.strip_punct = true,
            "--lowercase" => opts.lowercase = true,
            "--max-seq" => {
                i += 1;
                opts.max_window_tokens = xs[i].parse().expect("max-seq");
            }
            "--overlap" => {
                i += 1;
                opts.overlap_tokens = xs[i].parse().expect("overlap");
            }
            other => panic!("unknown golden flag {other}"),
        }
        i += 1;
    }
    opts
}

#[test]
#[ignore = "requires RUPunct weights at $RUPUNCT_MODEL or ~/.rupunct/punct"]
fn test_hf_parity_golden() {
    let Some(dir) = live_model_dir() else {
        panic!("ignored test requires weights at $RUPUNCT_MODEL or ~/.rupunct/punct");
    };
    let punct = Punctuator::load(&dir).expect("load punct model");
    let raw = std::fs::read_to_string(golden_path()).expect("read hf_parity_golden.json");
    let value: serde_json::Value = serde_json::from_str(&raw).expect("parse golden");
    let cases = value["cases"].as_array().expect("cases array");
    assert!(!cases.is_empty(), "golden is empty");
    for case in cases {
        let id = case["id"].as_str().expect("id");
        let input = case["input"].as_str().expect("input");
        let expect = case["expect"].as_str().unwrap_or("text");
        let flags = case["flags"].as_array().cloned().unwrap_or_default();
        let opts = opts_from_flags(&flags);
        let words = punct.restore_words_with(input, opts).expect(id);
        let got: Vec<&str> = words.iter().map(|w| w.label.as_str()).collect();
        let want: Vec<&str> = case["rupunct_labels"]
            .as_array()
            .expect("rupunct_labels")
            .iter()
            .map(|v| v.as_str().expect("label"))
            .collect();
        assert_eq!(got, want, "labels id={id}");
        if expect == "text" {
            let out = punct.restore_with(input, opts).expect(id);
            let want_text = case["rupunct_text"].as_str().expect("rupunct_text");
            assert_eq!(out, want_text, "text id={id}");
        }
    }
}
