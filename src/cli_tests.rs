use super::*;
use crate::fixture::{TinySpec, write_tiny_model};
use std::io::Cursor;

fn tiny_dir() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    write_tiny_model(tmp.path(), &TinySpec::default()).unwrap();
    tmp
}

fn run_capture(args: &[&str], stdin: &str, tty: bool) -> (Result<(), CliError>, String, String) {
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let result = run_from(&args, Cursor::new(stdin), &mut stdout, &mut stderr, tty);
    (
        result,
        String::from_utf8(stdout).unwrap(),
        String::from_utf8(stderr).unwrap(),
    )
}

#[test]
fn help_and_version_skip_model() {
    let (r, out, _) = run_capture(&["--help"], "", true);
    assert!(r.is_ok());
    assert!(out.contains("Usage:"), "{out}");
    assert!(out.contains("Exit status:"), "{out}");
    let (r, out, _) = run_capture(&["-V"], "", true);
    assert!(r.is_ok());
    assert!(out.contains("rupunct "));
    let (r, _, _) = run_capture(&["--unknown"], "", true);
    let err = r.unwrap_err();
    assert!(err.is_usage());
    assert!(err.to_string().contains("unknown argument"));
}

#[test]
fn tty_without_input_is_usage() {
    let err = run_capture(&[], "", true).0.unwrap_err();
    assert!(err.is_usage());
    assert!(err.to_string().contains("missing input"));
}

#[test]
fn missing_flag_values() {
    assert!(
        run_capture(&["--text"], "", true)
            .0
            .unwrap_err()
            .to_string()
            .contains("--text")
    );
    assert!(
        run_capture(&["--model-dir"], "", true)
            .0
            .unwrap_err()
            .to_string()
            .contains("--model-dir")
    );
}

#[test]
fn one_shot_text_and_positional() {
    let dir = tiny_dir();
    let d = dir.path().to_str().unwrap();
    let (r, out, _) = run_capture(&["--model-dir", d, "--text", "привет мир"], "", true);
    assert!(r.is_ok(), "{r:?}");
    assert_eq!(out.trim(), "привет мир");
    let (r, out, _) = run_capture(&["-m", d, "привет мир"], "", true);
    assert!(r.is_ok(), "{r:?}");
    assert_eq!(out.trim(), "привет мир");
    let (r, out, _) = run_capture(&["--model-dir", d, "--text=привет мир"], "", false);
    assert!(r.is_ok(), "{r:?}");
    assert_eq!(out.trim(), "привет мир");
}

#[test]
fn one_shot_file_and_stdin_dash() {
    let dir = tiny_dir();
    let d = dir.path().to_str().unwrap();
    let path = dir.path().join("in.txt");
    std::fs::write(&path, "привет мир").unwrap();
    let (r, out, _) = run_capture(&["-m", d, "--file", path.to_str().unwrap()], "", true);
    assert!(r.is_ok(), "{r:?}");
    assert_eq!(out.trim(), "привет мир");
    let (r, out, _) = run_capture(&["-m", d, "--text", "-"], "привет мир", true);
    assert!(r.is_ok(), "{r:?}");
    assert_eq!(out.trim(), "привет мир");
}

#[test]
fn jsonl_ready_and_frames() {
    let dir = tiny_dir();
    let stdin = "{\"text\":\"привет мир\"}\n\n{\"text\":\"\"}\nnot-json\n{\"foo\":1}\n{\"id\":7,\"text\":\"\"}\n";
    let (r, out, _) = run_capture(&["--model-dir", dir.path().to_str().unwrap()], stdin, false);
    assert!(r.is_ok(), "{r:?}");
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines[0], r#"{"ready":true}"#);
    assert!(lines[1].contains(r#""text":"#), "{}", lines[1]);
    assert!(lines[2].contains("text is empty"), "{}", lines[2]);
    assert!(lines[3].contains("JSONL protocol error"), "{}", lines[3]);
    assert!(lines[4].contains("JSONL protocol error"), "{}", lines[4]);
    assert!(lines[5].contains(r#""id":7"#), "{}", lines[5]);
    assert!(lines[5].contains("text is empty"), "{}", lines[5]);
}

#[test]
fn jsonl_rejects_oversized_line() {
    let dir = tiny_dir();
    let huge = "x".repeat(MAX_JSONL_LINE + 1);
    let stdin = format!("{huge}\n");
    let (r, out, _) = run_capture(
        &["--model-dir", dir.path().to_str().unwrap()],
        &stdin,
        false,
    );
    assert!(r.is_ok(), "{r:?}");
    assert!(out.contains("line too long"), "{out}");
}

#[test]
fn json_escapes_quotes_and_id() {
    assert!(json_text(None, "a\"b").contains("text"));
    assert!(json_error(None, "boom").contains("boom"));
    let id = serde_json::json!("abc");
    assert!(json_text(Some(&id), "hi").contains(r#""id":"abc""#));
}

#[test]
fn load_missing_model_dir_errors() {
    let (r, _, _) = run_capture(
        &["--model-dir", "/no/such-rupunct-model", "--text", "x"],
        "",
        true,
    );
    let err = r.unwrap_err();
    assert!(!err.is_usage());
    assert!(err.to_string().contains("load punct model"));
}

#[test]
fn complete_bash_and_unknown_shell() {
    let (r, out, _) = run_capture(&["--complete", "bash"], "", true);
    assert!(r.is_ok(), "{r:?}");
    assert!(out.contains("complete -F _rupunct rupunct"), "{out}");
    let err = run_capture(&["--complete", "cmd"], "", true).0.unwrap_err();
    assert!(err.is_usage());
}
