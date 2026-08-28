# rupunct — Agent Guide

On-device Russian punctuation and casing restoration. One binary. No cloud.

Input: unpunctuated Russian text.
Output: the same words with punctuation and capitalization restored.

## How it runs

```sh
scripts/fetch-model.sh
cargo build --release
rupunct --text "привет как дела"
rupunct "привет как дела"
rupunct -m ~/.rupunct/punct --jsonl
```

JSONL (stdin/stdout, one object per line; default for pipes):

```text
{"ready": true}
→ {"text": "…"} / {"id":1,"text":"…"}
← {"text": "…"} / {"id":1,"error":"…"}
```

Request-level failures are `{"error": "…"}` frames; the loop continues.
Only broken stdin/stdout aborts the worker. A TTY with no input is a usage
error (exit 2). `--text` uses `restore` (exit 1 on load/restore failure).
JSONL frames larger than 1 MiB are rejected.

`--text` / `[TEXT]` / `--file` is one-shot. Pipes without those flags are
the JSONL worker (`--jsonl` forces it). Weights:
`model.safetensors` + `tokenizer.json` + `config.json`.
`RUPUNCT_MODEL` overrides the default `~/.rupunct/punct` (`USERPROFILE` on
Windows if `HOME` is unset).

Compatible checkpoint: HuggingFace `RUPunct/RUPunct_small` at the SHA pinned
in `scripts/fetch-model.sh` (convert `pytorch_model.bin` → safetensors).

Restore windows: min(512, `max_position_embeddings`) tokens, 64-token overlap
(`--max-seq` / `--overlap` / `RestoreOpts`).

Direct crate deps stay tiny: `serde_json`, `memmap2`, and `gemm` only on
non-macOS (Accelerate on macOS). Safetensors and `tokenizer.json` are parsed
in-tree (no `safetensors` crate, no `serde::Deserialize`). No clap, thiserror,
rayon, or libm.

## Forbidden

- Cloud APIs
- Shipping weights in git
- Coupling this crate to a specific downstream app
- `unwrap` / `expect` in `src/**/*.rs` (tests return `Result` and use `?` where practical)
- `println!` in library modules (the binary writes through `cli`)

## Hard rules

- English for comments, docs, commits
- Implementation `src/**/*.rs`: max 400 lines. Test files (`*_tests.rs`): max 500
- `lib.rs` / `main.rs`: module tree / dispatch only
- Tests live in sibling `*_tests.rs` (or a `#[cfg(test)]` module in a tiny file)
- Git never contains weights

## Checks

```sh
cargo fmt --check
cargo test --lib --bins
cargo clippy --all-targets -- -D warnings
python3 scripts/check-file-size.py
python3 scripts/check-changelog.py
cargo llvm-cov --lib --bins --fail-under-lines 90
```

Live restore against real weights is local only:

```sh
cargo test --lib -- --ignored
```

## Versioning

Keep a Changelog + SemVer. `Cargo.toml` version is the product version.
`rupunct -V` is that string. Every user-visible change adds a bullet under
`## [Unreleased]`.
