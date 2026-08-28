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

JSONL (stdin/stdout, one object per line; default when stdin is not a TTY):

```text
{"ready": true}
→ {"text": "…"} / {"id":1,"text":"…"}
← {"text": "…","words":[…]} / {"id":1,"error":"…"}
```

Request-level failures are `{"error": "…"}` frames; the loop continues.
Only broken stdin/stdout aborts the worker. A TTY with no TEXT/`--text`/
`--file`/`--jsonl` is a usage error (exit 2), not help. `--text` uses
`restore` (exit 1 on load/restore failure). `--lossy` emits the original
text and exits 0 on restore failure. JSONL empty `"text"` is an error
frame; oneshot empty input is identity. JSONL frames larger than 1 MiB
or invalid UTF-8 are `{"error"}` frames; the loop continues. One-shot
`--text` / `--file` / stdin bodies are capped at 8 MiB. Redirected stdin
without those flags is JSONL; `--text -` is oneshot from stdin.

`--text` / `[TEXT]` / `--file` is one-shot. Non-TTY stdin without those
flags is the JSONL worker (`--jsonl` forces it). Weights:
`model.safetensors` + `tokenizer.json` + `config.json`.
`RUPUNCT_MODEL` overrides the default `~/.rupunct/punct` (`USERPROFILE` on
Windows if `HOME` is unset). Non-UTF-8 argv is usage (exit 2).

Compatible checkpoint: HuggingFace `RUPunct/RUPunct_small` at the SHA pinned
in `scripts/fetch-model.sh` (convert `pytorch_model.bin` → safetensors).

Restore windows: min(512, `max_position_embeddings`) tokens, 64-token overlap
(`--max-seq` ≥ 3, `--overlap` < `--max-seq`, or `RestoreOpts`). Overlap is
shrunk so each window commits at least one new word. Output is spliced into
the original string (newlines/tabs/hyphens stay). `--strip-punct` /
`RestoreOpts.strip_punct` deletes isolated punctuation spans.
`--lowercase` / `RestoreOpts.lowercase` lowercases words before encode.
JSONL success frames include `words` (`word` / `text` / `label` / `score` /
`start` / `end` byte offsets). `Punctuator::restore_words` is the library
form. `RUPUNCT_REPO=RUPunct/RUPunct_big` installs under
`~/.rupunct/RUPunct_big` (does not overwrite small).

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

Live restore against real weights is local only. Ignored tests **fail**
if the checkpoint is missing:

```sh
cargo test --lib -- --ignored
```

Local HuggingFace parity and F1 (weights required; transformers only for parity):

```sh
scripts/hf-parity.py
scripts/f1.py
scripts/fetch-eval.py
scripts/f1.py --gold ~/.rupunct/eval/f1_gold.json --quiet
RUPUNCT_MODEL=~/.rupunct/RUPunct_big scripts/f1.py --gold ~/.rupunct/eval/f1_gold.json --quiet
```

## Versioning

Keep a Changelog + SemVer. `Cargo.toml` version is the product version.
`rupunct -V` is that string. Every user-visible change adds a bullet under
`## [Unreleased]`.
