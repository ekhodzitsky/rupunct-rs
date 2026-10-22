# Publishing rupunct

The GitHub repository and crates.io package are separate publications. Preparing
this checkout does not change repository visibility or upload a crate.
The first public package version is 0.19.1.

## Validate the candidate

Use Rust 1.88 (the pinned toolchain). Install cargo-llvm-cov and the Python
requirements in [weights.md](weights.md#fetch). From a clean checkout:

```sh
cargo fmt --check
cargo test --workspace --lib --bins --locked
cargo clippy --all-targets --locked -- -D warnings
python3 scripts/check-file-size.py
python3 scripts/check-changelog.py
python3 scripts/test-convert-q8.py
cargo llvm-cov --lib --bins --fail-under-lines 90
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --locked
cargo package --list --locked
cargo publish --dry-run --locked
```

`cargo package` builds the extracted archive, so missing source files or fixtures
cannot be hidden by the working tree. The package includes setup scripts and
documentation, but no weights or CI configuration. Real-model parity and F1 are
local checks; CI's synthetic tests do not establish linguistic quality.

Before the first public release, review the reachable Git history and GitHub
metadata for material that should remain private. Review the archive file list
and NOTICE. Public source visibility does not publish model weights. Keep the
current limitations of the quality measurements visible in docs/quality.md.

## Release

1. Choose an unpublished release version, bump Cargo.toml, update Cargo.lock,
   and move Unreleased notes into the matching dated changelog entry.
2. Run the checks above, review and commit the candidate, and wait for CI on all
   three platforms. Verify the version is still available on crates.io.
3. Make the GitHub repository public when the owner approves publication.
4. Authenticate to crates.io locally or configure trusted publishing. Never put
   credentials in the repository. Run `cargo publish --locked` from the verified
   clean commit. Published crate versions cannot be overwritten.
5. Tag that exact commit as `v<version>` and create a GitHub release with the
   changelog. Update the README with the now-available `cargo install rupunct
   --version <version> --locked` command and library dependency example.
6. Verify installation in a fresh directory. Fetch weights separately using the
   scripts from the matching tag and run the README example. Cargo installation
   installs the executable, not the model-fetch scripts or model weights.

## Model artifacts

A prepacked Q8 release is a separate deliverable. Record the upstream revision,
conversion command and tool versions, SHA-256 of each final file, and upstream
license notices. Check real-model parity before publishing it. Current source
setup converts locally; only small's F32 inputs have embedded SHA-256 checks.
Do not advertise checksummed ready-made Q8 bundles until those artifacts exist.
