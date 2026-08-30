//! JSONL worker and one-shot CLI.

use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;

use crate::limits::MAX_RESTORE_BYTES;
use crate::{Punctuator, RestoreOpts, default_model_dir};

mod args;
mod capped;
mod help;
mod jsonl;
use args::{Cmd, OneShot, parse};
use capped::{ReadAll, read_capped_to_string};
use help::{write_complete, write_help};

/// Reject oversized JSONL frames so a single line cannot exhaust RAM.
pub(crate) const MAX_JSONL_LINE: usize = 1_048_576;
/// Cap for `--text -` / `--file` one-shot bodies (same as library restore).
pub(crate) const MAX_ONESHOT: usize = MAX_RESTORE_BYTES;

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
        io::BufReader::new(io::stdin()),
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
    R: BufRead + Send,
    W: Write,
    E: Write,
{
    match parse(args, stdin_is_tty).map_err(CliError::usage)? {
        Cmd::Help => write_help(stdout).map_err(io_err),
        Cmd::Version => writeln!(stdout, "rupunct {}", env!("CARGO_PKG_VERSION")).map_err(io_err),
        Cmd::Complete(shell) => write_complete(&shell, stdout),
        Cmd::OneShot(cmd) => oneshot(cmd, stdin, stdout),
        Cmd::Jsonl(cmd) => jsonl::run(cmd, stdin, stdout),
    }
}

pub(super) fn check_window(
    punct: &Punctuator,
    opts: RestoreOpts,
    explicit: bool,
) -> Result<(), CliError> {
    if !explicit {
        return Ok(());
    }
    let cap = punct.max_position_embeddings();
    if opts.max_window_tokens > cap {
        return Err(CliError::fail(format!(
            "--max-seq ({}) exceeds checkpoint max_position_embeddings ({cap})",
            opts.max_window_tokens
        )));
    }
    Ok(())
}

pub(super) fn load(dir: Option<PathBuf>) -> Result<Punctuator, CliError> {
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
    check_window(&punct, cmd.opts, cmd.explicit_max_seq)?;
    let out = apply(&punct, &source, cmd.lossy, cmd.opts)?;
    writeln!(stdout, "{out}").map_err(io_err)
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

#[cfg(test)]
use jsonl::{json_error, json_text};

pub(super) fn io_err(e: io::Error) -> CliError {
    CliError::fail(format!("I/O failed ({})", e.kind()))
}

#[cfg(test)]
#[path = "cli_tests.rs"]
mod tests;
