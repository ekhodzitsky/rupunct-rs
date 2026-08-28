//! JSONL worker and one-shot CLI.

use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;

use crate::{Punctuator, RestoreOpts, default_model_dir};

mod args;
mod capped;
mod help;
use args::{Cmd, OneShot, Worker, parse};
use capped::{NextLine, ReadAll, read_capped_line, read_capped_to_string};
use help::{write_complete, write_help};

/// Reject oversized JSONL frames so a single line cannot exhaust RAM.
pub(crate) const MAX_JSONL_LINE: usize = 1_048_576;
/// Cap for `--text -` / `--file` one-shot bodies.
pub(crate) const MAX_ONESHOT: usize = 8 * 1024 * 1024;

#[derive(Debug)]
pub(crate) enum CliError {
    Usage(String),
    Fail(String),
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Usage(m) | Self::Fail(m) => f.write_str(m),
        }
    }
}

impl CliError {
    fn usage(m: impl Into<String>) -> Self {
        Self::Usage(m.into())
    }
    fn fail(m: impl Into<String>) -> Self {
        Self::Fail(m.into())
    }
    fn is_usage(&self) -> bool {
        matches!(self, Self::Usage(_))
    }
}

pub fn run() -> i32 {
    let mut args = Vec::new();
    for (i, arg) in std::env::args_os().skip(1).enumerate() {
        match arg.into_string() {
            Ok(s) => args.push(s),
            Err(_) => {
                eprintln!("rupunct: argument {} is not valid UTF-8", i + 1);
                eprintln!("Try 'rupunct --help' for more information.");
                return 2;
            }
        }
    }
    let tty = io::stdin().is_terminal();
    match run_from(
        &args,
        io::stdin().lock(),
        &mut io::stdout(),
        &mut io::stderr(),
        tty,
    ) {
        Ok(()) => 0,
        Err(err) => {
            eprintln!("rupunct: {err}");
            if err.is_usage() {
                eprintln!("Try 'rupunct --help' for more information.");
                2
            } else {
                1
            }
        }
    }
}

pub(crate) fn run_from<R, W, E>(
    args: &[String],
    stdin: R,
    stdout: &mut W,
    _stderr: &mut E,
    stdin_is_tty: bool,
) -> Result<(), CliError>
where
    R: BufRead,
    W: Write,
    E: Write,
{
    match parse(args, stdin_is_tty).map_err(CliError::usage)? {
        Cmd::Help => write_help(stdout).map_err(io_err),
        Cmd::Version => writeln!(stdout, "rupunct {}", env!("CARGO_PKG_VERSION")).map_err(io_err),
        Cmd::Complete(shell) => write_complete(&shell, stdout),
        Cmd::OneShot(cmd) => oneshot(cmd, stdin, stdout),
        Cmd::Jsonl(cmd) => jsonl(cmd, stdin, stdout),
    }
}

fn load(dir: Option<PathBuf>) -> Result<Punctuator, CliError> {
    let dir = dir.unwrap_or_else(default_model_dir);
    Punctuator::load(&dir)
        .map_err(|e| CliError::fail(format!("load punct model from {}: {e}", dir.display())))
}

fn apply(
    punct: &Punctuator,
    text: &str,
    lossy: bool,
    opts: RestoreOpts,
) -> Result<String, CliError> {
    if lossy {
        Ok(punct
            .restore_with(text, opts)
            .unwrap_or_else(|_| text.to_string()))
    } else {
        punct
            .restore_with(text, opts)
            .map_err(|e| CliError::fail(e.to_string()))
    }
}

fn oneshot<R, W>(cmd: OneShot, stdin: R, stdout: &mut W) -> Result<(), CliError>
where
    R: BufRead,
    W: Write,
{
    let source = if cmd.from_stdin {
        read_all(stdin)?
    } else if cmd.from_file {
        let f = std::fs::File::open(&cmd.text)
            .map_err(|e| CliError::fail(format!("failed to read {}: {e}", cmd.text)))?;
        read_all(std::io::BufReader::new(f))?
    } else {
        if cmd.text.len() > MAX_ONESHOT {
            return Err(CliError::fail(format!(
                "input larger than {MAX_ONESHOT} bytes"
            )));
        }
        cmd.text
    };
    let punct = load(cmd.model_dir)?;
    let out = apply(&punct, &source, cmd.lossy, cmd.opts)?;
    writeln!(stdout, "{out}").map_err(io_err)
}

fn jsonl<R, W>(cmd: Worker, stdin: R, stdout: &mut W) -> Result<(), CliError>
where
    R: BufRead,
    W: Write,
{
    let punct = load(cmd.model_dir)?;
    writeln!(stdout, r#"{{"ready":true}}"#).map_err(io_err)?;
    stdout.flush().map_err(io_err)?;
    let mut stdin = stdin;
    loop {
        match read_capped_line(&mut stdin, MAX_JSONL_LINE).map_err(io_err)? {
            NextLine::Eof => break,
            NextLine::TooLong => {
                writeln!(stdout, "{}", json_error(None, "line too long")).map_err(io_err)?;
            }
            NextLine::InvalidUtf8 => {
                writeln!(stdout, "{}", json_error(None, "invalid UTF-8")).map_err(io_err)?;
            }
            NextLine::Ok(line) => {
                if line.trim().is_empty() {
                    continue;
                }
                match handle(&punct, &line, cmd.lossy, cmd.opts) {
                    Ok((id, out)) => {
                        writeln!(stdout, "{}", json_text(id.as_ref(), &out)).map_err(io_err)?;
                    }
                    Err((id, e)) => {
                        writeln!(stdout, "{}", json_error(id.as_ref(), &e)).map_err(io_err)?;
                    }
                }
            }
        }
        stdout.flush().map_err(io_err)?;
    }
    Ok(())
}

fn read_all(mut r: impl std::io::Read) -> Result<String, CliError> {
    match read_capped_to_string(&mut r, MAX_ONESHOT).map_err(io_err)? {
        ReadAll::Ok(s) => Ok(s),
        ReadAll::TooLong => Err(CliError::fail(format!(
            "input larger than {MAX_ONESHOT} bytes"
        ))),
        ReadAll::InvalidUtf8 => Err(CliError::fail("input is not valid UTF-8")),
    }
}

fn handle(
    punct: &Punctuator,
    line: &str,
    lossy: bool,
    opts: RestoreOpts,
) -> Result<(Option<serde_json::Value>, String), (Option<serde_json::Value>, String)> {
    let v: serde_json::Value =
        serde_json::from_str(line).map_err(|_| (None, "JSONL protocol error".to_string()))?;
    let id = v.get("id").cloned();
    let text = v
        .get("text")
        .and_then(|t| t.as_str())
        .ok_or_else(|| (id.clone(), "JSONL protocol error".to_string()))?;
    if text.trim().is_empty() {
        return Err((id, "text is empty".to_string()));
    }
    if lossy {
        Ok((
            id,
            punct
                .restore_with(text, opts)
                .unwrap_or_else(|_| text.to_string()),
        ))
    } else {
        punct
            .restore_with(text, opts)
            .map(|s| (id.clone(), s))
            .map_err(|e| (id, e.to_string()))
    }
}

fn json_text(id: Option<&serde_json::Value>, text: &str) -> String {
    frame(id, "text", text)
}

fn json_error(id: Option<&serde_json::Value>, msg: &str) -> String {
    frame(id, "error", msg)
}

fn frame(id: Option<&serde_json::Value>, key: &str, value: &str) -> String {
    let Ok(v) = serde_json::to_string(value) else {
        return r#"{"error":"internal error"}"#.to_string();
    };
    match id {
        Some(id) => format!(r#"{{"id":{id},"{key}":{v}}}"#),
        None => format!(r#"{{"{key}":{v}}}"#),
    }
}

fn io_err(e: io::Error) -> CliError {
    CliError::fail(format!("I/O failed ({})", e.kind()))
}

#[cfg(test)]
#[path = "cli_tests.rs"]
mod tests;
