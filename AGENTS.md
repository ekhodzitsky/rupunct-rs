# rupunct — Agent Guide

On-device Russian punctuation and casing restoration. One binary. No cloud.

Input: unpunctuated Russian text.
Output: the same words with punctuation and capitalization restored.

## How it runs

```sh
scripts/fetch-model.sh
cargo build --release
rupunct --text "привет как дела"
rupunct --model-dir ~/.rupunct/punct
```

JSONL (stdin/stdout, one object per line):

```text
{"ready": true}
→ {"text": "…"}
← {"text": "…"}
← {"error": "…"}
```

Request-level failures are `{"error": "…"}` frames; the loop continues.
Only broken stdin/stdout aborts the worker.

`--text` is one-shot. Omit it for the JSONL worker. Weights:
`model.safetensors` + `tokenizer.json` + `config.json`.
`RUPUNCT_MODEL` overrides the default `~/.rupunct/punct`.

Compatible checkpoint: HuggingFace `RUPunct/RUPunct_small` after converting
`pytorch_model.bin` to safetensors (`scripts/fetch-model.sh`).

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
