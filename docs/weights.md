# Weights

Three files in `$RUPUNCT_MODEL` or `~/.rupunct/RUPunct_big`:

`model.safetensors` · `tokenizer.json` · `config.json`

Never commit `*.safetensors`. Git never contains weights.

## Fetch

Needs `huggingface-cli` or `hf`. Conversion of `pytorch_model.bin` needs Python
with `torch` + `safetensors`. Packing Q8_0 needs `numpy`. `RUPUNCT_Q8=0` keeps
F32. Fetch skips the `.bin` when `model.safetensors` is already local. SHA-256
of small is the F32 file; small is not packed.

```sh
scripts/fetch-model.sh
# -> ~/.rupunct/RUPunct_big   (RUPunct/RUPunct_big, pinned, packed Q8_0)
RUPUNCT_Q8=0 scripts/fetch-model.sh
# -> same dir, F32
```

Small (does not overwrite big):

```sh
RUPUNCT_REPO=RUPunct/RUPunct_small scripts/fetch-model.sh
# -> ~/.rupunct/punct  (F32; hidden 312)
```

Override dest with `RUPUNCT_MODEL`. Search path at runtime: `--model-dir` / `-m`
or `RUPUNCT_MODEL` (`USERPROFILE` on Windows if `HOME` is unset).

SHA-256 pins are only for `RUPunct_small`. Pins live in `scripts/fetch-model.sh`.

## Checkpoints

| | big (default) | small |
|--|---------------|-------|
| HF | [RUPunct_big](https://huggingface.co/RUPunct/RUPunct_big) | [RUPunct_small](https://huggingface.co/RUPunct/RUPunct_small) |
| Dir | `~/.rupunct/RUPunct_big` | `~/.rupunct/punct` |
| Size | ~180 MB Q8 (~711 MB F32) | ~111 MB F32 |
| Layers × hidden | 12×768 | 3×312 |
| `max_position_embeddings` | 512 | 2048 |
| Labels | 33 | 33 |

Default tokenizer for big has `lowercase=true`. Same `aggregation_strategy="first"`
per word, not HuggingFace entity grouping.

Checkpoint license: see the model card before commercial use. Engine is MIT.

## Q8_0

`scripts/fetch-model.sh` packs large MatMul and embedding tables as GGML Q8_0
(`{name}.q8_0` + `{name}.shape`) and drops `pytorch_model.bin`. Packed tensors
stay mmap'd. Linear GEMM dequants an output tile; embedding gather dequants
selected rows. LayerNorm and tiny tensors stay F32. On RUPunct_big: ~678 MB →
~180 MB on disk, warm RSS ~361 MB → ~127 MB. Punct micro-F1 on the 800-sentence
cache matches F32 (0.845). `RUPunct_small` is left F32 (hidden 312 is not a
multiple of 32).

```sh
scripts/convert-q8.py ~/.rupunct/RUPunct_big/model.safetensors /tmp/model.q8.safetensors
# already-packed files are left as-is
```
