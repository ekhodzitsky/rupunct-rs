//! JSONL worker and one-shot CLI.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use rupunct::{Punctuator, default_model_dir};

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut text: Option<String> = None;
    let mut model_dir: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_help();
                return Ok(());
            }
            "-V" | "--version" => {
                println!("rupunct {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "--text" => {
                text = Some(args.next().ok_or("--text needs a value")?);
            }
            "--model-dir" => {
                model_dir = Some(PathBuf::from(
                    args.next().ok_or("--model-dir needs a value")?,
                ));
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    let dir = model_dir.unwrap_or_else(default_model_dir);
    let punct = Punctuator::load(&dir)
        .map_err(|e| format!("load punct model from {}: {e:#}", dir.display()))?;
    if let Some(t) = text {
        println!("{}", punct.restore(&t));
        return Ok(());
    }
    jsonl(&punct)
}

fn jsonl(punct: &Punctuator) -> Result<(), String> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    writeln!(stdout, r#"{{"ready":true}}"#).map_err(io_err)?;
    stdout.flush().map_err(io_err)?;
    for line in stdin.lock().lines() {
        let line = line.map_err(io_err)?;
        if line.trim().is_empty() {
            continue;
        }
        match handle(punct, &line) {
            Ok(out) => writeln!(stdout, "{}", json_text(&out)).map_err(io_err)?,
            Err(e) => writeln!(stdout, "{}", json_error(&e)).map_err(io_err)?,
        }
        stdout.flush().map_err(io_err)?;
    }
    Ok(())
}

fn handle(punct: &Punctuator, line: &str) -> Result<String, String> {
    let v: serde_json::Value =
        serde_json::from_str(line).map_err(|_| "JSONL protocol error".to_string())?;
    let text = v
        .get("text")
        .and_then(|t| t.as_str())
        .ok_or_else(|| "JSONL protocol error".to_string())?;
    if text.trim().is_empty() {
        return Err("text is empty".to_string());
    }
    Ok(punct.restore(text))
}

fn json_text(text: &str) -> String {
    match serde_json::to_string(text) {
        Ok(s) => format!(r#"{{"text":{s}}}"#),
        Err(_) => r#"{"error":"internal error"}"#.to_string(),
    }
}

fn json_error(msg: &str) -> String {
    match serde_json::to_string(msg) {
        Ok(s) => format!(r#"{{"error":{s}}}"#),
        Err(_) => r#"{"error":"internal error"}"#.to_string(),
    }
}

fn io_err(e: io::Error) -> String {
    format!("I/O failed ({})", e.kind())
}

fn print_help() {
    eprintln!(
        "\
On-device Russian punctuation restoration

Usage: rupunct [OPTIONS]

Options:
      --text <TEXT>            One-shot source (omit for JSONL worker)
      --model-dir <DIR>        Weights dir (env RUPUNCT_MODEL; default ~/.rupunct/punct)
  -h, --help                   Print help
  -V, --version                Print version"
    );
}
