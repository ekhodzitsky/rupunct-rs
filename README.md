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
```

JSONL worker (stdin/stdout, one object per line):

```text
{"ready":true}
{"text":"привет как дела"}
{"text":"Привет, как дела?"}
{"text":""}
{"error":"text is empty"}
```

Request-level failures (bad JSON, empty `text`, restore error, oversized line)
are `{"error":"…"}` frames; the loop continues. Only a broken stdin/stdout
aborts the worker. `--text` prints the restored line or exits 2 on failure.

Library:

```rust
use rupunct::{Punctuator, default_model_dir};

let p = Punctuator::load(&default_model_dir())?;
let ru = p.restore("привет как дела")?;
```

`restore` returns `Err` on inference failure. `restore_lossy` returns the
original string instead. Inputs longer than 512 WordPiece tokens (or the
checkpoint `max_position_embeddings` if smaller) are split on word boundaries
with a 64-token overlap.

## Weights

Download [RUPunct/RUPunct_small](https://huggingface.co/RUPunct/RUPunct_small)
(MIT) and convert `pytorch_model.bin` → `model.safetensors`:

```sh
scripts/fetch-model.sh
```

Override the search path with `--model-dir` or `RUPUNCT_MODEL`. Do not commit
`*.safetensors`.

## License

MIT (engine). Checkpoint: see the RUPunct model card before commercial use.
