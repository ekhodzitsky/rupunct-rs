# rupunct

On-device **Russian punctuation and casing** restoration. One binary. No cloud.

Input is unpunctuated Russian text. Output is the same words with punctuation
and capitalization restored. Inference is a local BERT token-classifier
compatible with [RUPunct/RUPunct_small](https://huggingface.co/RUPunct/RUPunct_small)
(`aggregation_strategy="first"` on **each word**, not HuggingFace entity
grouping of consecutive identical tags).

Weights stay off git. The engine never opens a network socket.

## Quick start

```sh
# Checkpoint layout (three files):
#   model.safetensors  tokenizer.json  config.json
# Default dir: $RUPUNCT_MODEL, else ~/.rupunct/punct
scripts/fetch-model.sh

cargo build --release
./target/release/rupunct --text "привет меня зовут анна"
./target/release/rupunct "привет как дела"
./target/release/rupunct --strip-punct --lowercase --text "привет, как дела"
./target/release/rupunct --complete bash >> ~/.bash_completion
```

A TTY with no `TEXT` / `--text` / `--file` / `--jsonl` is a **usage error**
(exit **2**), not help. Stdin that is not a TTY (pipe **or** redirect) without
those flags is the JSONL worker. Oneshot from stdin: `--text -` / `--file -`.

```text
{"ready":true}
{"id":1,"text":"привет как дела"}
{"id":1,"text":"Привет, как дела?","words":[{"word":"привет","text":"Привет,","label":"UPPER_COMMA","score":0.99}, ...]}
{"text":""}
{"error":"text is empty"}
```

Request-level failures (bad JSON, empty `text`, restore error, oversized line,
invalid UTF-8) are `{"error":"…"}` frames; the loop continues. Optional `"id"`
is echoed on the response. Only a broken stdin/stdout aborts the worker.

`--text` prints the restored line or exits **1** on load/restore failure.
`--lossy` emits the original text and exits **0** on restore failure (load
failure is still **1**). JSONL `--lossy` emits `{"text":…}` instead of
`{"error":…}`. Empty JSONL `"text"` is an error frame; oneshot empty input
is identity. One-shot `--text` / `--file` / stdin bodies are capped at 8 MiB.

`--strip-punct` drops isolated punctuation tokens (fixes `привет, как` →
`Привет! , как`). `--lowercase` lowercases words before the encoder (fixes
`ООО` → `Ооо` titlecase). Both default **off** so the HuggingFace card path
is unchanged.

```
  0  success (--lossy restore failure is success)
  1  load or restore failure
  2  usage error
```

Library:

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
```

`restore` returns `Err` on inference failure. `restore_lossy` returns the
original string instead. `restore_words` keeps per-word `label` and softmax
`score` (not calibrated). Output is space-joined words (newlines/tabs are
not preserved). Inputs longer than 512 WordPiece tokens (or the checkpoint
`max_position_embeddings` if smaller) are split on word boundaries with a
64-token overlap. The small checkpoint allows 2048 positions; the default
512 window is a speed choice, not bit-identical to a full-length HuggingFace
`pipeline` on long inputs. `--max-seq 2048` matches the checkpoint. `--max-seq`
is ≥ 3 and `--overlap` must be smaller.

## Quality

`RUPunct_small` is a 3-layer BERT-tiny (hidden 312, ~111 MB F32). The card
targets simple CPU texts, not SOTA restore. Typical outputs on the pinned
checkpoint:

| Input | Output |
|-------|--------|
| `привет как дела` | `Привет, как дела?` |
| `ты пойдешь со мной в кино` | `Ты пойдешь со мной в кино?` |
| already-punctuated `привет, как дела` | double marks unless `--strip-punct` |

There is no in-tree class-wise F1 set yet. Live goldens (`cargo test --lib --
--ignored`) lock the pinned checkpoint’s strings; they are not a linguistic
oracle. `RUPunct_big` uses the same 33 labels and a `lowercase=true`
WordPiece tokenizer; this engine loads it:

```sh
RUPUNCT_REPO=RUPunct/RUPunct_big scripts/fetch-model.sh
```

(Uses HuggingFace `main` unless `RUPUNCT_REVISION` is set. SHA-256 pins are
only for `RUPunct_small` at the commit in `scripts/fetch-model.sh`.)

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
