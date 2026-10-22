# rupunct

[![CI](https://github.com/ekhodzitsky/rupunct-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/ekhodzitsky/rupunct-rs/actions/workflows/ci.yml)

On-device Russian punctuation and casing restoration. One binary. No cloud.

```text
привет как дела  →  Привет, как дела?
```

Local BERT. Default checkpoint [RUPunct_big](https://huggingface.co/RUPunct/RUPunct_big).
Weights stay off git. The process never opens a network socket.

## Install

Rust **1.88+**. Install the CLI from crates.io:

```sh
cargo install rupunct --version 0.19.1 --locked
```

For Rust applications, add `rupunct = "0.19.1"` to `[dependencies]`.
Weights are installed separately. Fetch them from the matching source release
(or build the executable from source):

```sh
git clone --branch v0.19.1 https://github.com/ekhodzitsky/rupunct-rs.git
cd rupunct-rs
scripts/fetch-model.sh   # Q8_0 into ~/.rupunct/RUPunct_big (~180 MB)
cargo build --release
./target/release/rupunct --text "привет как дела"
```

Fetching weights requires the Hugging Face CLI and Python conversion tools;
see [setup prerequisites](docs/weights.md#fetch). Inference itself needs no Python.
The shell fetch script runs on Unix or under WSL; native Windows users can
copy a prepared model directory and pass `-m`.

macOS uses Accelerate; Linux and Windows use `gemm`.

## Use

```sh
rupunct --text "привет как дела"
rupunct --jsonl          # default when stdin is not a TTY
```

```rust
use rupunct::{Punctuator, default_model_dir};

let p = Punctuator::load(&default_model_dir())?;
p.restore("привет как дела")?;
```

Flags, JSONL, library, windows, `--threads`: [docs/usage.md](docs/usage.md)

## Quality

Local 800-sentence cache (UD SynTagRus test + Common Voice ru, **not CI**):

| system | exact | punct F1 |
|--------|------:|---------:|
| rupunct `RUPunct_big` | **58%** | **0.85** |
| [Kontur SBERT-large](https://huggingface.co/kontur-ai/sbert_punc_case_ru) | 49% | 0.79 |
| [Silero TE](https://github.com/snakers4/silero-models) | 37% | 0.75 |

This gold is written + read-speech, not interview ASR. Kontur only predicts `. , ?`.
The engine matches HuggingFace `aggregation_strategy="first"` **per word** (39/39 on small).
`RUPunct_small` is the CPU/size option, not the quality default. The ceiling is the checkpoint.

Tables, gold, scripts: [docs/quality.md](docs/quality.md)

## Weights

Default: `~/.rupunct/RUPunct_big` (`RUPUNCT_MODEL` or `-m`).
Small: `RUPUNCT_REPO=RUPunct/RUPunct_small` → `~/.rupunct/punct`.

[docs/weights.md](docs/weights.md)

## More

[Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [License](LICENSE) · [NOTICE](NOTICE)

MIT (engine). The upstream small and big checkpoint cards declare MIT;
weights are distributed separately. See [NOTICE](NOTICE).
