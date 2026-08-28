#!/bin/sh
# Install a pinned RUPunct_small checkpoint into $RUPUNCT_MODEL or ~/.rupunct/punct.
# Needs huggingface-cli (or `hf`). Conversion of pytorch_model.bin needs Python
# with torch + safetensors.
set -eu

# Pin: RUPunct/RUPunct_small @ 2024-05-01 (config hidden=312, max_pos=2048).
REVISION="${RUPUNCT_REVISION:-97aa9196b313e931424a5401fa42bc69dd15acd9}"
DEST="${RUPUNCT_MODEL:-${HOME:?HOME unset}/.rupunct/punct}"
mkdir -p "$DEST"

HF_BIN=""
if command -v hf >/dev/null 2>&1; then
  HF_BIN=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_BIN=huggingface-cli
else
  echo "fetch-model: hf / huggingface-cli not on PATH" >&2
  echo "  pip install -U huggingface_hub" >&2
  echo "  then: hf download RUPunct/RUPunct_small --revision $REVISION --local-dir $DEST" >&2
  exit 1
fi

if [ "$HF_BIN" = hf ]; then
  hf download RUPunct/RUPunct_small --revision "$REVISION" --local-dir "$DEST"
else
  huggingface-cli download RUPunct/RUPunct_small --revision "$REVISION" --local-dir "$DEST"
fi

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
state = torch.load(src, map_location="cpu", weights_only=True)
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
tensors = {k: v.contiguous().cpu() for k, v in state.items()}
save_file(tensors, str(out))
print(f"wrote {out}")
PY
fi

check_sha256() {
  python3 - "$1" "$2" <<'PY'
import hashlib, sys
path, expect = sys.argv[1], sys.argv[2]
with open(path, "rb") as f:
    got = hashlib.sha256(f.read()).hexdigest()
if got != expect:
    raise SystemExit(f"{path}: sha256 {got} != {expect}")
print(f"ok sha256 {path}")
PY
}

for f in model.safetensors tokenizer.json config.json; do
  if [ ! -f "$DEST/$f" ]; then
    echo "fetch-model: missing $DEST/$f" >&2
    exit 1
  fi
done

# Content hashes at RUPunct/RUPunct_small @ $REVISION
check_sha256 "$DEST/config.json" "ec5503b1b544ee19563895c7c73884a4ec94b52b17eda172c11dcdc0eb924f03"
check_sha256 "$DEST/tokenizer.json" "7ca617388c2092a3a84272025c52bbf3c6db0aee225c0351186295c0b5d3ddc6"
if [ -f "$DEST/pytorch_model.bin" ]; then
  check_sha256 "$DEST/pytorch_model.bin" "5b05a696d0a9629e0938852e221abcd84cc75ab67a4445fb0b4ea40630dd6b59"
fi
echo "ok $DEST (revision $REVISION)"
