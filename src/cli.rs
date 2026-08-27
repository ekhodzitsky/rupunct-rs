//! JSONL worker and one-shot CLI.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use crate::{Punctuator, default_model_dir};

pub fn run() -> i32 {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run_from(
        &args,
        io::stdin().lock(),
        &mut io::stdout(),
        &mut io::stderr(),
    ) {
        Ok(()) => 0,
        Err(err) => {
            eprintln!("{err}");
            2
        }
    }
}

pub(crate) fn run_from<R, W, E>(
    args: &[String],
    stdin: R,
    stdout: &mut W,
    stderr: &mut E,
) -> Result<(), String>
where
    R: BufRead,
    W: Write,
    E: Write,
{
    let mut text: Option<String> = None;
    let mut model_dir: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                write_help(stderr);
                return Ok(());
            }
            "-V" | "--version" => {
                writeln!(stdout, "rupunct {}", env!("CARGO_PKG_VERSION")).map_err(io_err)?;
                return Ok(());
            }
            "--text" => {
                i += 1;
                text = Some(args.get(i).cloned().ok_or("--text needs a value")?);
            }
            "--model-dir" => {
                i += 1;
                model_dir = Some(PathBuf::from(
                    args.get(i).cloned().ok_or("--model-dir needs a value")?,
                ));
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }
    let dir = model_dir.unwrap_or_else(default_model_dir);
    let punct = Punctuator::load(&dir)
        .map_err(|e| format!("load punct model from {}: {e}", dir.display()))?;
    if let Some(t) = text {
        writeln!(stdout, "{}", punct.restore_lossy(&t)).map_err(io_err)?;
        return Ok(());
    }
    jsonl(&punct, stdin, stdout)
}

fn jsonl<R, W>(punct: &Punctuator, stdin: R, stdout: &mut W) -> Result<(), String>
where
    R: BufRead,
    W: Write,
{
    writeln!(stdout, r#"{{"ready":true}}"#).map_err(io_err)?;
    stdout.flush().map_err(io_err)?;
    for line in stdin.lines() {
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
    Ok(punct.restore_lossy(text))
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

fn write_help(err: &mut impl Write) {
    let _ = writeln!(
        err,
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

#[cfg(test)]
#[path = "cli_tests.rs"]
mod tests;
