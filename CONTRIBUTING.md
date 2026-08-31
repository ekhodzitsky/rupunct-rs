# Contributing

User docs: [README.md](README.md), [docs/usage.md](docs/usage.md),
[docs/quality.md](docs/quality.md), [docs/weights.md](docs/weights.md).
Keep the root README short; details go in `docs/`.

Layout, forbidden dependencies, and the check list live in [AGENTS.md](AGENTS.md).
User-visible changes add a bullet under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md).

```sh
cargo fmt --check
cargo test --lib --bins
cargo clippy --all-targets -- -D warnings
python3 scripts/check-file-size.py
python3 scripts/check-changelog.py
```

Live restore against real weights is local only (`cargo test --lib -- --ignored`).
Weights must not enter git.
