use super::*;
use crate::fixture::{TinySpec, write_tiny_model};
use std::io::Cursor;

fn tiny_dir() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    write_tiny_model(tmp.path(), &TinySpec::default()).unwrap();
    tmp
}

fn run_capture(args: &[&str], stdin: &str) -> (Result<(), String>, String, String) {
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let result = run_from(&args, Cursor::new(stdin), &mut stdout, &mut stderr);
    (
        result,
        String::from_utf8(stdout).unwrap(),
        String::from_utf8(stderr).unwrap(),
    )
}

#[test]
fn help_and_version_skip_model() {
    let (r, _out, err) = run_capture(&["--help"], "");
    assert!(r.is_ok());
    assert!(err.contains("Usage: rupunct"));
    let (r, out, _) = run_capture(&["-V"], "");
    assert!(r.is_ok());
    assert!(out.contains("rupunct "));
    let (r, _, _) = run_capture(&["--unknown"], "");
    assert!(r.unwrap_err().contains("unknown argument"));
}

#[test]
fn missing_flag_values() {
    assert!(
        run_capture(&["--text"], "")
            .0
            .unwrap_err()
            .contains("--text")
    );
    assert!(
        run_capture(&["--model-dir"], "")
            .0
            .unwrap_err()
            .contains("--model-dir")
    );
}

#[test]
fn one_shot_text() {
    let dir = tiny_dir();
    let (r, out, _) = run_capture(
        &[
            "--model-dir",
            dir.path().to_str().unwrap(),
            "--text",
            "привет мир",
        ],
        "",
    );
    assert!(r.is_ok(), "{r:?}");
    assert_eq!(out.trim(), "привет мир");
}

#[test]
fn jsonl_ready_and_frames() {
    let dir = tiny_dir();
    let stdin = "{\"text\":\"привет мир\"}\n\n{\"text\":\"\"}\nnot-json\n{\"foo\":1}\n";
    let (r, out, _) = run_capture(&["--model-dir", dir.path().to_str().unwrap()], stdin);
    assert!(r.is_ok(), "{r:?}");
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines[0], r#"{"ready":true}"#);
    assert!(lines[1].contains(r#""text":"#), "{}", lines[1]);
    assert!(lines[2].contains("text is empty"), "{}", lines[2]);
    assert!(lines[3].contains("JSONL protocol error"), "{}", lines[3]);
    assert!(lines[4].contains("JSONL protocol error"), "{}", lines[4]);
}

#[test]
fn json_escapes_quotes() {
    assert!(json_text("a\"b").contains(r#"\"b"#) || json_text("a\"b").contains("\\\""));
    assert!(json_error("boom").contains("boom"));
}

#[test]
fn load_missing_model_dir_errors() {
    let (r, _, _) = run_capture(
        &["--model-dir", "/no/such/rupunct-model", "--text", "x"],
        "",
    );
    assert!(r.unwrap_err().contains("load punct model"));
}
