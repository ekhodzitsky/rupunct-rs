# rupunct

On-device **Russian punctuation and casing** restoration. One binary. No cloud.

Extracted from [gigastt-pro](https://github.com/ekhodzitsky/gigastt-pro): RUPunct BERT
token-classification (`aggregation_strategy="first"`) plus the in-tree WordPiece
tokenizer and F32 CPU runtime.

Downstream **dnmt** (and TTS) consume the restored line. Weights stay off git.

## Quick start

```sh
# Weights: copy a converted RUPunct_small dir (same layout as gigastt-pro punct/)
#   model.safetensors  tokenizer.json  config.json
# Default search: $RUPUNCT_MODEL, then ~/.rupunct/punct, then ~/.gigastt/models/punct

cargo build --release
./target/release/rupunct --text "забери спину пройди закрытый гард не теряй фрейм"
```

JSONL worker (stdin/stdout, same shape as dnmt):

```text
{"ready":true}
{"text":"забери спину пройди закрытый гард"}
{"text":"Забери спину, пройди закрытый гард."}
{"text":""}
{"error":"text is empty"}
```

Library:

```rust
use rupunct::{Punctuator, default_model_dir};

let p = Punctuator::load(&default_model_dir())?;
let ru = p.restore("привет как дела");
```

## Weights

RUPunct_small (MIT) as used by gigastt-pro. Convert with
`gigastt-pro/scripts/convert_rupunct_to_candle.py`, or reuse
`~/.gigastt/models/punct` if you already run gigastt-pro.

Do not commit `*.safetensors`.

## License

MIT (engine). RUPunct weights: see the model card; confirm before commercial use.
