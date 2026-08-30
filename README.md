# rupunct

[![CI](https://github.com/ekhodzitsky/rupunct-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/ekhodzitsky/rupunct-rs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rustc-1.88%2B-orange?logo=rust)](rust-toolchain.toml)
[![Checkpoint](https://img.shields.io/badge/checkpoint-RUPunct__small-yellow)](https://huggingface.co/RUPunct/RUPunct_small)

On-device **Russian punctuation and casing** restoration. One binary. No cloud.

Input is unpunctuated Russian text. Output is the same words with punctuation
and capitalization restored. Inference is a local BERT token-classifier
compatible with [RUPunct/RUPunct_small](https://huggingface.co/RUPunct/RUPunct_small)
(`aggregation_strategy="first"` on **each word**, not HuggingFace entity
grouping of consecutive identical tags).

Weights stay off git. The engine never opens a network socket.

The **checkpoint** is a 3-layer BERT-tiny, not SOTA restore. This repo is a
faithful on-device engine for that checkpoint (and `RUPunct_big`).

## Contents

- [Install](#install)
- [CLI](#cli)
- [JSONL worker](#jsonl-worker)
- [Library](#library)
- [Quality](#quality)
- [Weights](#weights)
- [License](#license)

## Install

Rust **1.88+** (`rust-toolchain.toml`). macOS uses Accelerate; Linux and Windows
use `gemm`.

```sh
scripts/fetch-model.sh
cargo build --release
./target/release/rupunct --text "привет как дела"
```

Checkpoint layout (three files in `$RUPUNCT_MODEL` or `~/.rupunct/punct`):

`model.safetensors` · `tokenizer.json` · `config.json`

Not on crates.io (`publish = false`). Do not `cargo install` from a registry.

## CLI

```sh
./target/release/rupunct --text "привет меня зовут анна"
./target/release/rupunct "привет как дела"
./target/release/rupunct --strip-punct --lowercase --text "привет, как дела"
./target/release/rupunct --complete bash >> ~/.bash_completion
```

| Flag / mode | Behavior |
|-------------|----------|
| One-shot | `--text`, `[TEXT]`, or `--file` (`-` = stdin) |
| Worker | non-TTY stdin, or `--jsonl` |
| TTY with no input | usage error **2**, not help |
| `--strip-punct` | drop isolated punctuation tokens (off by default) |
| `--lowercase` | lowercase words before the encoder (off by default) |
| `--lossy` | on restore failure, emit the original text (exit **0**) |
| `--max-seq` | BERT window, ≥ 3 and ≤ checkpoint (`2048` on small, `512` on big) |

```
  0  success (--lossy restore failure is success)
  1  load or restore failure
  2  usage error
```

`--text` / `--file` / stdin bodies are capped at **8 MiB**. JSONL frames at **1 MiB**.
Non-UTF-8 argv is usage (exit **2**).

`--strip-punct` fixes `привет, как` → double marks. `--lowercase` fixes `ООО` →
`Ооо` titlecase. Both default **off** so the HuggingFace card path is unchanged.

## JSONL worker

Stdin that is not a TTY (pipe or redirect) without `--text` / `--file` / `[TEXT]`
is the worker. `--jsonl` forces it. `--text -` is oneshot from stdin.

```text
{"ready":true}
{"id":1,"text":"привет как дела"}
{"id":1,"text":"Привет, как дела?","words":[{"word":"привет","text":"Привет,","label":"UPPER_COMMA","score":0.99}, ...]}
{"text":""}
{"error":"text is empty"}
```

Request-level failures (bad JSON, empty `text`, restore error, oversized line,
invalid UTF-8) are `{"error":"…"}` frames; the loop continues. Optional `"id"`
is echoed. Only a broken stdin/stdout aborts the worker. JSONL `--lossy` emits
`{"text":…}` instead of `{"error":…}`. Empty JSONL `"text"` is an error frame;
oneshot empty input is identity.

Each `words[]` entry has `word` / `text` / `label` / `score` / `start` / `end`
(byte offsets into the original string). `score` is softmax, not calibrated.

## Library

```rust
use rupunct::{Punctuator, RestoreOpts, default_model_dir};

let p = Punctuator::load(&default_model_dir())?;
let ru = p.restore("привет как дела")?;
let ru = p.restore_with("привет как дела", RestoreOpts {
    max_window_tokens: 256,
    overlap_tokens: 32,
    ..RestoreOpts::default()
})?;
let words = p.restore_words("привет как дела")?;
// words[i].start / end are byte offsets into the original string
```

`restore` returns `Err` on inference failure. `restore_lossy` returns the
original string instead. Output is spliced into the original string (newlines,
tabs, and hyphens stay; `--strip-punct` deletes those spans).

Inputs longer than 512 WordPiece tokens (or `max_position_embeddings` if
smaller) are split on word boundaries with a 64-token overlap. The small
checkpoint allows **2048** positions; the default 512 window is a speed choice,
not bit-identical to a full-length HuggingFace `pipeline` on long inputs.
`--max-seq 2048` matches **small**; big allows **512**. An explicit `--max-seq`
larger than the checkpoint is an error (the library still clamps `RestoreOpts`
for tiny test models). `--overlap` must be smaller than `--max-seq`.

## Quality

`RUPunct_small` is a 3-layer BERT-tiny (hidden 312, ~111 MB F32). The card
targets simple CPU texts, **not** SOTA restore. Typical outputs on the pinned
checkpoint:

| Input | Output |
|-------|--------|
| `привет как дела` | `Привет, как дела?` |
| `ты пойдешь со мной в кино` | `Ты пойдешь со мной в кино?` |
| already-punctuated `привет, как дела` | double marks unless `--strip-punct` |

Local HuggingFace parity (`scripts/hf-parity.py`, not CI) is **39/39** on
`tests/fixtures/hf_parity_golden.json`: first-subtoken labels match the encoder.
Space-separated inputs also match a per-word `process_token` join. Splice keeps
hyphens, newlines, and tabs; the join does not. Consecutive identical tags are
**not** merged the way `pipeline(..., aggregation_strategy="first")` groups
entities.

F1 is `scripts/f1.py` (not CI). The 31-sentence in-tree set is a **smoke** test
and ranked small above big; that ranking does **not** hold on public corpora.

External gold (local cache, not git): 400 sentences from the
[UD Russian-SynTagRus](https://github.com/UniversalDependencies/UD_Russian-SynTagRus)
test split (CC BY-NC-SA 4.0) and 400 [Common Voice](https://github.com/common-voice/common-voice)
ru prompts (CC0). Golos transcripts have **no** punctuation, so they cannot
be F1 gold.

| checkpoint | split | n | exact | punct micro-F1 (excl. `O`) | case acc. |
|------------|-------|---|-------|----------------------------|-----------|
| small | SynTagRus test | 400 | 40% | 0.78 | 97% |
| small | Common Voice ru | 400 | 53% | 0.76 | 98% |
| small | all | 800 | 46% | 0.77 | 98% |
| big | SynTagRus test | 400 | 54% | 0.86 | 98% |
| big | Common Voice ru | 400 | 62% | 0.83 | 98% |
| big | all | 800 | 58% | 0.85 | 98% |

Default stays **small** (111 MB, 2048 positions, CPU). Big is the quality
option (~711 MB F32, `max_position_embeddings` **512**, `lowercase=true`
tokenizer, same 33 labels). Live goldens (`cargo test --lib -- --ignored`)
lock the small checkpoint’s strings; they are not a linguistic oracle.

```sh
scripts/fetch-eval.py
scripts/f1.py --gold ~/.rupunct/eval/f1_gold.json --quiet
RUPUNCT_REPO=RUPunct/RUPunct_big scripts/fetch-model.sh
# -> ~/.rupunct/RUPunct_big  (small stays in ~/.rupunct/punct)
rupunct -m ~/.rupunct/RUPunct_big --text "привет как дела"
RUPUNCT_MODEL=~/.rupunct/RUPunct_big scripts/f1.py --gold ~/.rupunct/eval/f1_gold.json --quiet
```

SHA-256 pins are only for `RUPunct_small`. Override dest with `RUPUNCT_MODEL`.

## Weights

Download [RUPunct/RUPunct_small](https://huggingface.co/RUPunct/RUPunct_small)
(MIT) and convert `pytorch_model.bin` → `model.safetensors`:

```sh
scripts/fetch-model.sh
```

Override the search path with `--model-dir` / `-m` or `RUPUNCT_MODEL`. Do not
commit `*.safetensors`.

## License

MIT (engine). Checkpoint: see the RUPunct model card before commercial use.
Eval corpora used by `scripts/fetch-eval.py` stay off git; see [NOTICE](NOTICE).
