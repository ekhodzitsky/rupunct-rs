# Changelog

The format is [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Root README is a short landing page. CLI/JSONL/library, F1 tables, and
  weight fetch live in `docs/usage.md`, `docs/quality.md`, `docs/weights.md`.

## [0.16.0] - 2026-08-31

### Added

- `scripts/f1.py --system silero` scores Silero TE (`v2_4lang_q`, lan=ru) on the
  same gold. On the 800-sentence cache `RUPunct_big` beats it (punct micro-F1
  0.85 vs 0.75, exact 58% vs 37%). Quantized TE needs `qnnpack` on Apple
  Silicon; some hyphen inputs crash inside the package and score as identity.

## [0.15.0] - 2026-08-31

### Added

- `restore_batch` and the JSONL worker run independent texts in parallel
  (`std::thread`). `--threads` is a shared budget: batch items and window
  workers do not multiply. JSONL responses stay in input order. `--threads 1`
  is sequential.
- `scripts/f1.py --system kontur` scores `kontur-ai/sbert_punc_case_ru` on the
  same gold as rupunct (torch+transformers, not CI). On the 800-sentence
  cache `RUPunct_big` beats it (punct micro-F1 0.85 vs 0.79, exact 58% vs 49%).

## [0.14.0] - 2026-08-31

### Added

- Overlapping restore windows run in parallel (`std::thread`, no rayon),
  capped at `available_parallelism`. One-window inputs stay on the caller
  thread. `--threads N` / `RestoreOpts.threads` (`0` = auto, `1` = sequential).
  Overlap merge is unchanged: higher softmax wins; ties keep the earlier window.

## [0.13.0] - 2026-08-30

### Changed

- Default checkpoint is `RUPunct/RUPunct_big` at `~/.rupunct/RUPunct_big`.
  `scripts/fetch-model.sh` downloads big unless `RUPUNCT_REPO=RUPunct/RUPunct_small`
  (still `~/.rupunct/punct`, SHA-256 pinned). Ignored HF goldens still load
  small. `--max-seq` default 512 matches big `max_position_embeddings`.

## [0.12.1] - 2026-08-28

### Fixed

- Explicit `--max-seq` larger than the checkpoint's `max_position_embeddings`
  is an error (exit 1), not a silent clamp. Default 512 still clamps in the
  library so tiny test models keep working. `--max-seq 2048` is valid on
  small, not on big (512).

## [0.12.0] - 2026-08-28

### Added

- `scripts/fetch-eval.py` builds a local 800-sentence F1 cache: UD
  Russian-SynTagRus test (written, CC BY-NC-SA) and Common Voice ru prompts
  (CC0). Corpora stay off git.

### Changed

- On that cache `RUPunct_big` beats small (punct micro-F1 0.85 vs 0.77,
  exact 58% vs 46%). The 31-sentence in-tree set ranked them the other way
  and is only a smoke test. Golos transcripts have no punctuation, so they
  are not F1 gold. Default checkpoint remains small.

## [0.11.0] - 2026-08-28

### Added

- `scripts/hf-parity.py` covers 39 cases (hyphen, newlines/tabs, `--strip-punct`,
  `--lowercase`, 600-token `--max-seq 2048`). Frozen outputs live in
  `tests/fixtures/hf_parity_golden.json`; ignored tests replay them.
- `scripts/f1.py` scores trailing punctuation and casing on the 31-sentence
  in-tree set `tests/fixtures/f1_gold.json` (written + ASR-style).

## [0.10.0] - 2026-08-28

### Changed

- `restore` / CLI / JSONL `text` splice into the original string: newlines,
  tabs, and hyphens stay where they were. `--strip-punct` deletes those
  spans instead of leaving a hole that `join(' ')` used to hide.
- Overlapping restore windows keep the higher-softmax label instead of
  always taking the first-commit window.
- `scripts/hf-parity.py` compares a built `rupunct` to the HuggingFace
  pipeline when transformers is installed (not run in CI).

## [0.9.0] - 2026-08-28

### Added

- `RestoredWord.start` / `end` byte offsets into the original input (JSONL
  `words` includes them). Leading whitespace is no longer trimmed before
  restore, so offsets match the caller string.
- `scripts/fetch-model.sh` installs `RUPunct_big` under `~/.rupunct/RUPunct_big`
  (does not overwrite small) and pins that repo's 2024-05-01 revision.

### Changed

- `--strip-punct` keeps `-` between words and `.` between digits (`кто-то`,
  `42.5`). Isolated commas/periods are still dropped.
- CLI oneshot cap shares `MAX_RESTORE_BYTES`. CI installs `llvm-tools-preview`.

## [0.8.0] - 2026-08-28

### Added

- `Punctuator::restore_words` / `restore_words_with` and `RestoredWord`
  (`word`, `text`, `label`, softmax `score` of the first subtoken).
- JSONL success frames include a `words` array (existing `text` is unchanged).
- `--strip-punct` / `RestoreOpts.strip_punct` drops isolated punctuation
  tokens so already-punctuated input is not double-marked.
- `--lowercase` / `RestoreOpts.lowercase` lowercases words before the encoder
  (ASR-style). Default off (HuggingFace parity).
- `scripts/fetch-model.sh` SHA-256 of `model.safetensors` (rebuilds from
  `pytorch_model.bin` on mismatch). `RUPUNCT_REPO=RUPunct/RUPunct_big` downloads
  that repo at `main` (override with `RUPUNCT_REVISION`; hashes are only for
  the pinned small revision).
- `BertNormalizer` `lowercase=true` (needed to load `RUPunct_big`).
- Library `restore` / `restore_words` reject bodies larger than 8 MiB.

### Changed

- `RestoreOpts::resolve` clamps the window to ≥ 3 (same as `--max-seq`).
- `config.json` rejects `model_type` other than `bert`, non-absolute
  `position_embedding_type`, `max_position_embeddings` < 3, non-positive
  `layer_norm_eps`, and oversized hidden/layer/vocab fields.
- Tokenizer/config files and the mapped weights have size caps. FFN load
  checks both linear shapes. GEMM buffers use `checked_mul`; BLAS length
  checks are on in release builds.
- CI installs the pinned 1.88 toolchain (matches `rust-toolchain.toml`).

### Fixed

- Whitespace-only and control-only input (normalizer drops the chars) is
  identity of the original string, not a trimmed leftover.

## [0.7.0] - 2026-08-28

### Fixed

- Token windows shrink overlap so each window still commits a new word.
  `--max-seq` too small with a large `--overlap` no longer labels the new
  word as `LOWER_O` (identity).
- `--max-seq` must be ≥ 3. `--overlap` must be less than `--max-seq`
  (including the 64 default). Completions match the parser (no zsh stacked
  shorts; fish valued flags take an argument).
- Non-UTF-8 argv is a usage error (exit 2), not a panic.
- `config.json` rejects `hidden_size` / `num_attention_heads` of 0.
  F32 mmap views require a 4-byte-aligned offset. Integer safetensors
  payloads must match `shape × dtype width`.
- `scripts/fetch-model.sh` skips SHA-256 when `RUPUNCT_REVISION` is not the
  pinned commit (hashes are only valid for that pin).

### Changed

- A TTY with no input is a usage error (docs now match). Non-TTY stdin
  without `--text` / `--file` is JSONL (pipe **or** redirect). `--lossy`
  restore failure is exit 0 / `{"text"}`. One-shot 8 MiB cap includes
  `--text` argv. Ignored live tests fail if the checkpoint is missing.

### Added

- CI jobs on Windows (mmap `File` lifetime) and a pinned Rust toolchain on
  Ubuntu / macOS / Windows.

## [0.6.0] - 2026-08-28

### Fixed

- JSONL reads a capped buffer before allocating a line. Invalid UTF-8 is an
  `{"error"}` frame; the worker keeps running. A trailing CR is stripped so
  CRLF frames parse.
- `--text` / `--file` / `--model-dir` reject a following flag as a value
  (`--text=--help` still works). Repeated flags error. `--max-seq` must be ≥ 2.
- Safetensors load rejects overlapping payloads. The mapped file handle is kept
  alive (Windows).
- Attention runs heads in-place on `[seq, hidden]` (no per-head Q/K/V copies).

### Added

- `Punctuator::restore_batch_with`. One-shot stdin/file bodies capped at 8 MiB.
- `scripts/fetch-model.sh` verifies SHA-256 of `config.json`, `tokenizer.json`,
  and `pytorch_model.bin` at the pinned revision.
- CI job on macOS (Accelerate GEMM).

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
