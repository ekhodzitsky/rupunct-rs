//! JSONL stdin/stdout worker.

use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};
use std::sync::Mutex;
use std::sync::mpsc;

use super::args::Worker;
use super::capped::{NextLine, read_capped_line};
use super::{CliError, MAX_JSONL_LINE, check_window, io_err, load};
use crate::{Punctuator, RestoreOpts, RestoredWord};

enum Kind {
    Line(String),
    TooLong,
    InvalidUtf8,
}

struct Job {
    seq: usize,
    kind: Kind,
}

pub(super) fn run<R, W>(cmd: Worker, stdin: R, stdout: &mut W) -> Result<(), CliError>
where
    R: BufRead + Send,
    W: Write,
{
    let punct = load(cmd.model_dir)?;
    check_window(&punct, cmd.opts, cmd.explicit_max_seq)?;
    writeln!(stdout, r#"{{"ready":true}}"#).map_err(io_err)?;
    stdout.flush().map_err(io_err)?;
    let jobs = crate::par::worker_count(cmd.opts.threads, usize::MAX);
    if jobs <= 1 {
        sequential(&punct, stdin, stdout, cmd.lossy, cmd.opts)
    } else {
        let mut item_opts = cmd.opts;
        item_opts.threads = 1;
        parallel(&punct, stdin, stdout, cmd.lossy, item_opts, jobs)
    }
}

fn sequential<R, W>(
    punct: &Punctuator,
    mut stdin: R,
    stdout: &mut W,
    lossy: bool,
    opts: RestoreOpts,
) -> Result<(), CliError>
where
    R: BufRead,
    W: Write,
{
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
                match handle(punct, &line, lossy, opts) {
                    Ok((id, out, words)) => {
                        writeln!(stdout, "{}", json_ok(id.as_ref(), &out, &words))
                            .map_err(io_err)?;
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

fn parallel<R, W>(
    punct: &Punctuator,
    stdin: R,
    stdout: &mut W,
    lossy: bool,
    opts: RestoreOpts,
    jobs: usize,
) -> Result<(), CliError>
where
    R: BufRead + Send,
    W: Write,
{
    let (work_tx, work_rx) = mpsc::sync_channel::<Job>(jobs);
    let (res_tx, res_rx) = mpsc::channel::<(usize, String)>();
    let work_rx = Mutex::new(work_rx);
    let io_fail = Mutex::new(None::<io::Error>);

    std::thread::scope(|scope| {
        scope.spawn(|| {
            read_jobs(stdin, work_tx, &io_fail);
        });
        for _ in 0..jobs {
            let work_rx = &work_rx;
            let res_tx = res_tx.clone();
            scope.spawn(move || {
                loop {
                    let job = match work_rx.lock() {
                        Ok(rx) => match rx.recv() {
                            Ok(j) => j,
                            Err(_) => return,
                        },
                        Err(_) => return,
                    };
                    let frame = frame_job(punct, lossy, opts, job.kind);
                    if res_tx.send((job.seq, frame)).is_err() {
                        return;
                    }
                }
            });
        }
        drop(res_tx);
        write_ordered(res_rx, stdout)?;
        match io_fail.lock() {
            Ok(mut slot) => {
                if let Some(e) = slot.take() {
                    return Err(io_err(e));
                }
            }
            Err(_) => return Err(CliError::fail("jsonl worker lock poisoned")),
        }
        Ok(())
    })
}

fn read_jobs<R: BufRead>(
    mut stdin: R,
    work_tx: mpsc::SyncSender<Job>,
    io_fail: &Mutex<Option<io::Error>>,
) {
    let mut seq = 0usize;
    loop {
        let line = match read_capped_line(&mut stdin, MAX_JSONL_LINE) {
            Ok(v) => v,
            Err(e) => {
                if let Ok(mut slot) = io_fail.lock() {
                    *slot = Some(e);
                }
                break;
            }
        };
        let kind = match line {
            NextLine::Eof => break,
            NextLine::TooLong => Kind::TooLong,
            NextLine::InvalidUtf8 => Kind::InvalidUtf8,
            NextLine::Ok(s) if s.trim().is_empty() => continue,
            NextLine::Ok(s) => Kind::Line(s),
        };
        if work_tx.send(Job { seq, kind }).is_err() {
            break;
        }
        seq += 1;
    }
}

fn write_ordered<W: Write>(
    res_rx: mpsc::Receiver<(usize, String)>,
    stdout: &mut W,
) -> Result<(), CliError> {
    let mut next = 0usize;
    let mut pending = BTreeMap::new();
    while let Ok((seq, frame)) = res_rx.recv() {
        pending.insert(seq, frame);
        drain_ready(&mut pending, &mut next, stdout)?;
    }
    drain_ready(&mut pending, &mut next, stdout)?;
    if pending.is_empty() {
        Ok(())
    } else {
        Err(CliError::fail("jsonl worker lost a frame"))
    }
}

fn drain_ready<W: Write>(
    pending: &mut BTreeMap<usize, String>,
    next: &mut usize,
    stdout: &mut W,
) -> Result<(), CliError> {
    while let Some(frame) = pending.remove(next) {
        writeln!(stdout, "{frame}").map_err(io_err)?;
        stdout.flush().map_err(io_err)?;
        *next += 1;
    }
    Ok(())
}

fn frame_job(punct: &Punctuator, lossy: bool, opts: RestoreOpts, kind: Kind) -> String {
    match kind {
        Kind::TooLong => json_error(None, "line too long"),
        Kind::InvalidUtf8 => json_error(None, "invalid UTF-8"),
        Kind::Line(line) => match handle(punct, &line, lossy, opts) {
            Ok((id, out, words)) => json_ok(id.as_ref(), &out, &words),
            Err((id, e)) => json_error(id.as_ref(), &e),
        },
    }
}

type HandleOk = (Option<serde_json::Value>, String, Vec<RestoredWord>);
type HandleErr = (Option<serde_json::Value>, String);

fn handle(
    punct: &Punctuator,
    line: &str,
    lossy: bool,
    opts: RestoreOpts,
) -> Result<HandleOk, HandleErr> {
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
    match punct.restore_bundle(text, opts) {
        Ok((out, words)) => Ok((id, out, words)),
        Err(_) if lossy => Ok((id, text.to_string(), Vec::new())),
        Err(e) => Err((id, e.to_string())),
    }
}

fn json_ok(id: Option<&serde_json::Value>, text: &str, words: &[RestoredWord]) -> String {
    let mut map = serde_json::Map::new();
    if let Some(id) = id {
        map.insert("id".into(), id.clone());
    }
    map.insert("text".into(), serde_json::Value::String(text.to_string()));
    let arr: Vec<serde_json::Value> = words
        .iter()
        .map(|w| {
            serde_json::json!({
                "word": w.word,
                "text": w.text,
                "label": w.label,
                "score": w.score,
                "start": w.start,
                "end": w.end,
            })
        })
        .collect();
    map.insert("words".into(), serde_json::Value::Array(arr));
    serde_json::Value::Object(map).to_string()
}

#[cfg(test)]
pub(super) fn json_text(id: Option<&serde_json::Value>, text: &str) -> String {
    frame(id, "text", text)
}

pub(super) fn json_error(id: Option<&serde_json::Value>, msg: &str) -> String {
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
