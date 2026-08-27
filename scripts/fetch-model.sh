#!/bin/sh
# Install a RUPunct_small checkpoint into $RUPUNCT_MODEL or ~/.rupunct/punct.
# Needs huggingface-cli. Conversion of pytorch_model.bin needs Python with
# torch + safetensors.
set -eu

DEST="${RUPUNCT_MODEL:-${HOME:?HOME unset}/.rupunct/punct}"
mkdir -p "$DEST"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "fetch-model: huggingface-cli not on PATH" >&2
  echo "  pip install -U huggingface_hub" >&2
  echo "  then: huggingface-cli download RUPunct/RUPunct_small --local-dir $DEST" >&2
  exit 1
fi

huggingface-cli download RUPunct/RUPunct_small --local-dir "$DEST"

if [ ! -f "$DEST/model.safetensors" ] && [ -f "$DEST/pytorch_model.bin" ]; then
  if ! command -v python3 >/dev/null 2>&1; then
    echo "fetch-model: python3 required to convert pytorch_model.bin" >&2
    exit 1
  fi
  python3 - "$DEST" <<'PY'
import sys
from pathlib import Path

dest = Path(sys.argv[1])
src = dest / "pytorch_model.bin"
out = dest / "model.safetensors"
try:
    import torch
    from safetensors.torch import save_file
except ImportError as e:
    raise SystemExit(f"need torch and safetensors: {e}") from e
state = torch.load(src, map_location="cpu")
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
tensors = {k: v.contiguous().cpu() for k, v in state.items()}
save_file(tensors, str(out))
print(f"wrote {out}")
PY
fi

for f in model.safetensors tokenizer.json config.json; do
  if [ ! -f "$DEST/$f" ]; then
    echo "fetch-model: missing $DEST/$f" >&2
    exit 1
  fi
done
echo "ok $DEST"
