# Changelog

The format is [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
