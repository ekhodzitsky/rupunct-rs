# rupunct

On-device **Russian punctuation and casing** restoration. One binary. No cloud.

Input is unpunctuated Russian text. Output is the same words with punctuation
and capitalization restored. Inference is a local BERT token-classifier
compatible with [RUPunct/RUPunct_small](https://huggingface.co/RUPunct/RUPunct_small)
(`aggregation_strategy="first"`).

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
./target/release/rupunct --complete bash >> ~/.bash_completion
```

A TTY with no `TEXT` / `--text` / `--file` / `--jsonl` is a **usage error**
(exit **2**), not help. Stdin that is not a TTY (pipe **or** redirect) without
those flags is the JSONL worker. Oneshot from stdin: `--text -` / `--file -`.

```text
{"ready":true}
{"id":1,"text":"привет как дела"}
{"id":1,"text":"Привет, как дела?"}
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
})?;
```

`restore` returns `Err` on inference failure. `restore_lossy` returns the
original string instead. Output is space-joined words (newlines/tabs are
not preserved). Inputs longer than 512 WordPiece tokens (or the checkpoint
`max_position_embeddings` if smaller) are split on word boundaries with a
64-token overlap. `--max-seq` is ≥ 3 and `--overlap` must be smaller.
Override with `--max-seq` / `--overlap` or `RestoreOpts`.

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
