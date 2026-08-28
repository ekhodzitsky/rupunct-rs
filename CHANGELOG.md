# Changelog

The format is [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-08-28

### Changed

- In-tree safetensors reader/writer. The `safetensors` crate is gone.
- `tokenizer.json` is walked as `serde_json::Value`. No `serde::Deserialize`
  and no direct `serde` dependency (JSONL/`config.json` still use `serde_json`).

## [0.4.0] - 2026-08-27

### Added

- POSIX/GNU CLI: positional `TEXT`, `--text=`, `-m`/`--model-dir=`, `-f`/`--file`,
  `--jsonl`, `--lossy`, `--max-seq`, `--overlap`, `--complete {bash,zsh,fish}`.
- `--text -` / `--file -` / positional `-` read one-shot source from stdin.
- JSONL responses echo an optional request `"id"`.
- Public `RestoreOpts` and `Punctuator::restore_with`.

### Changed

- Exit status: `0` success, `1` load/restore failure, `2` usage error.
  `--help` / `--version` write to stdout. A TTY with no input is usage, not
  a hanging JSONL worker.
- Dropped `thiserror`, `libm`, and `rayon`. `gemm` is a target dependency
  only outside macOS (Apple Accelerate on macOS). GELU uses in-tree erf.
  LayerNorm is sequential (no pointer-splitting rayon).

## [0.3.0] - 2026-08-27

### Added

- JSONL worker rejects frames larger than 1 MiB (`line too long`) instead of
  buffering an unbounded line.
- `scripts/fetch-model.sh` pins `RUPunct/RUPunct_small` by commit SHA and
  prefers the `hf` CLI when present.

### Changed

- `--text` and the JSONL worker call `restore` (not `restore_lossy`). Inference
  failures are errors (`exit 2` / `{"error":…}`), matching the documented
  protocol.
- Long inputs are packed by **token** budget: windows cap at 512 tokens (or
  `max_position_embeddings` if smaller) with a 64-token overlap. Packing uses
  per-word subword counts instead of re-encoding on every expansion.
- Missing-word labels default to `LOWER_O`, not index 0 (`UPPER_PERIOD`).
- `encode` is `split_words` + `encode_words`, so added tokens stay whole words
  in every window.
- CJK spacing range matches HuggingFace `is_chinese_char` (`U+2B820`).
- BERT GELU uses `libm::erff` (same erf form as PyTorch).
- Default model dir falls back to `%USERPROFILE%` when `HOME` is unset.
- Non-F32 `.weight` / `.bias` tensors fail at load. Extra I64 tensors such as
  `position_ids` are still skipped.
- `config.json` no longer invents `num_labels = 33`. Missing labels is an error.
  Unsupported `hidden_act` (not `gelu`) is rejected.
- Attention key mask uses `-1e9` (HuggingFace-scale) instead of `-1e4`.

### Removed

- Stub `Device` enum (CPU is the only backend).

### Fixed

- `torch.load` in `fetch-model.sh` uses `weights_only=True`.
- `.gitignore` covers `*.bin` / `pytorch_model.bin`, not only `*.safetensors`.

## [0.2.0] - 2026-08-27

### Added

- Public `Error` / `Result`. `Punctuator::restore` returns `Result<String, Error>`.
- `restore_lossy` (identity fallback) and `restore_batch`.
- Word-aligned BERT windows so input longer than `max_position_embeddings` is
  restored in overlapping chunks instead of being silently truncated.
- `scripts/fetch-model.sh` to install RUPunct_small under `~/.rupunct/punct`.
- CI: fmt, clippy `-D warnings`, tests, file-size, changelog, line coverage.

### Changed

- Independent product: default model dir is `$RUPUNCT_MODEL` or `~/.rupunct/punct`.
- F32 runtime keeps only ops used by BERT restore.
- `restore` splits on tokenizer words, not raw whitespace, so existing
  punctuation is a separate word (HuggingFace pipeline semantics).

### Removed

- Silent restore fallback that returned the original string on BERT failure
  (`restore_lossy` is the explicit opt-in).

## [0.1.0] - 2026-08-27

### Added

- RUPunct BERT restore: `Punctuator::load` / `restore`, JSONL worker, `--text`
  one-shot.
- WordPiece tokenizer and F32 CPU BERT.
