use std::path::PathBuf;

use crate::RestoreOpts;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Cmd {
    Help,
    Version,
    Complete(String),
    OneShot(OneShot),
    Jsonl(Worker),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct OneShot {
    pub text: String,
    pub from_stdin: bool,
    pub from_file: bool,
    pub model_dir: Option<PathBuf>,
    pub lossy: bool,
    pub opts: RestoreOpts,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Worker {
    pub model_dir: Option<PathBuf>,
    pub lossy: bool,
    pub opts: RestoreOpts,
}

pub(super) fn parse(args: &[String], stdin_is_tty: bool) -> Result<Cmd, String> {
    let mut text: Option<String> = None;
    let mut file: Option<String> = None;
    let mut positional: Option<String> = None;
    let mut model_dir: Option<PathBuf> = None;
    let mut jsonl = false;
    let mut lossy = false;
    let mut max_seq: Option<usize> = None;
    let mut overlap: Option<usize> = None;
    let mut complete: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        if a == "--" {
            i += 1;
            if i >= args.len() {
                return Err("expected a text operand after --".into());
            }
            set_once(&mut positional, args[i].clone(), "positional text")?;
            if i + 1 < args.len() {
                return Err(format!("unexpected argument: {}", args[i + 1]));
            }
            break;
        }
        let (flag, inline) = split_flag(a);
        match flag {
            "-h" | "--help" => {
                reject_inline(flag, inline)?;
                return Ok(Cmd::Help);
            }
            "-V" | "--version" => {
                reject_inline(flag, inline)?;
                return Ok(Cmd::Version);
            }
            "--complete" => {
                complete = Some(need_value(args, &mut i, flag, inline)?);
            }
            "--text" => {
                text = Some(need_value(args, &mut i, flag, inline)?);
            }
            "-f" | "--file" => {
                file = Some(need_value(args, &mut i, flag, inline)?);
            }
            "-m" | "--model-dir" => {
                model_dir = Some(PathBuf::from(need_value(args, &mut i, flag, inline)?));
            }
            "--jsonl" => {
                reject_inline(flag, inline)?;
                jsonl = true;
            }
            "--lossy" => {
                reject_inline(flag, inline)?;
                lossy = true;
            }
            "--max-seq" => {
                max_seq = Some(parse_usize(&need_value(args, &mut i, flag, inline)?, flag)?);
            }
            "--overlap" => {
                overlap = Some(parse_usize(&need_value(args, &mut i, flag, inline)?, flag)?);
            }
            other if other.starts_with('-') && other != "-" => {
                return Err(format!("unknown argument: {other}"));
            }
            _ => {
                set_once(&mut positional, a.to_string(), "positional text")?;
            }
        }
        i += 1;
    }

    if let Some(shell) = complete {
        if text.is_some() || file.is_some() || positional.is_some() || jsonl {
            return Err("--complete cannot be combined with restore options".into());
        }
        return Ok(Cmd::Complete(shell));
    }

    let opts = RestoreOpts {
        max_window_tokens: max_seq.unwrap_or(RestoreOpts::default().max_window_tokens),
        overlap_tokens: overlap.unwrap_or(RestoreOpts::default().overlap_tokens),
    };

    let sources = usize::from(text.is_some())
        + usize::from(file.is_some())
        + usize::from(positional.is_some());
    if sources > 1 {
        return Err("use only one of --text, --file, or a positional string".into());
    }
    if jsonl && sources > 0 {
        return Err("--jsonl cannot be combined with --text, --file, or positional text".into());
    }

    if jsonl {
        return Ok(Cmd::Jsonl(Worker {
            model_dir,
            lossy,
            opts,
        }));
    }

    if let Some(t) = text {
        let from_stdin = t == "-";
        return Ok(Cmd::OneShot(OneShot {
            text: t,
            from_stdin,
            from_file: false,
            model_dir,
            lossy,
            opts,
        }));
    }
    if let Some(path) = file {
        let from_stdin = path == "-";
        return Ok(Cmd::OneShot(OneShot {
            text: path,
            from_stdin,
            from_file: true,
            model_dir,
            lossy,
            opts,
        }));
    }
    if let Some(t) = positional {
        let from_stdin = t == "-";
        return Ok(Cmd::OneShot(OneShot {
            text: t,
            from_stdin,
            from_file: false,
            model_dir,
            lossy,
            opts,
        }));
    }

    if stdin_is_tty {
        return Err(
            "missing input (pass TEXT, --text, --file, or --jsonl; pipes default to JSONL)".into(),
        );
    }
    Ok(Cmd::Jsonl(Worker {
        model_dir,
        lossy,
        opts,
    }))
}

fn split_flag(arg: &str) -> (&str, Option<&str>) {
    if let Some(rest) = arg.strip_prefix("--")
        && let Some((k, v)) = rest.split_once('=')
    {
        return (&arg[..2 + k.len()], Some(v));
    }
    (arg, None)
}

fn reject_inline(flag: &str, inline: Option<&str>) -> Result<(), String> {
    if inline.is_some() {
        return Err(format!("{flag} does not take a value"));
    }
    Ok(())
}

fn need_value(
    args: &[String],
    i: &mut usize,
    flag: &str,
    inline: Option<&str>,
) -> Result<String, String> {
    if let Some(v) = inline {
        return Ok(v.to_string());
    }
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

fn set_once(slot: &mut Option<String>, value: String, what: &str) -> Result<(), String> {
    if slot.is_some() {
        return Err(format!("multiple {what} operands"));
    }
    *slot = Some(value);
    Ok(())
}

fn parse_usize(raw: &str, flag: &str) -> Result<usize, String> {
    raw.parse::<usize>()
        .map_err(|_| format!("{flag} needs a positive integer, got '{raw}'"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn help_version_complete() {
        assert!(matches!(parse(&a(&["-h"]), true), Ok(Cmd::Help)));
        assert!(matches!(parse(&a(&["--version"]), false), Ok(Cmd::Version)));
        let Cmd::Complete(s) = parse(&a(&["--complete=zsh"]), true).unwrap() else {
            panic!("complete");
        };
        assert_eq!(s, "zsh");
    }

    #[test]
    fn equals_and_short_flags() {
        let Cmd::OneShot(o) = parse(&a(&["--text=hi", "-m", "/w"]), true).unwrap() else {
            panic!("oneshot");
        };
        assert_eq!(o.text, "hi");
        assert_eq!(
            o.model_dir.as_deref().map(|p| p.as_os_str()),
            Some(std::ffi::OsStr::new("/w"))
        );
    }

    #[test]
    fn positional_and_tty_missing() {
        let Cmd::OneShot(o) = parse(&a(&["привет"]), true).unwrap() else {
            panic!("pos");
        };
        assert_eq!(o.text, "привет");
        let err = parse(&a(&[]), true).unwrap_err();
        assert!(err.contains("missing input"), "{err}");
        assert!(matches!(parse(&a(&[]), false), Ok(Cmd::Jsonl(_))));
    }

    #[test]
    fn jsonl_conflict() {
        assert!(parse(&a(&["--jsonl", "--text", "x"]), false).is_err());
        assert!(parse(&a(&["--text", "a", "b"]), true).is_err());
    }

    #[test]
    fn max_seq_and_double_dash() {
        let Cmd::OneShot(o) = parse(&a(&["--max-seq", "128", "--", "--text"]), true).unwrap()
        else {
            panic!("dd");
        };
        assert_eq!(o.text, "--text");
        assert_eq!(o.opts.max_window_tokens, 128);
    }
}
