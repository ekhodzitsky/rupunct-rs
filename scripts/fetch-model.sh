#!/bin/sh
# Install a pinned RUPunct checkpoint into $RUPUNCT_MODEL or ~/.rupunct/<name>.
# Needs huggingface-cli (or `hf`). Conversion of pytorch_model.bin needs Python
# with torch + safetensors. Q8_0 packing needs numpy (`scripts/convert-q8.py`).
# RUPUNCT_Q8=0 keeps F32. pytorch_model.bin is dropped once safetensors exists.
#
# Default: RUPunct/RUPunct_big at the pinned revision into ~/.rupunct/RUPunct_big,
# packed Q8_0. Skips pytorch_model.bin when model.safetensors is already present.
# RUPunct_small goes to ~/.rupunct/punct (SHA-256 of F32, then packed; last dim
# padded to 32). Does not overwrite big. Override dest with RUPUNCT_MODEL.
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

PINNED_REPO="RUPunct/RUPunct_small"
# Pin: RUPunct/RUPunct_small @ 2024-05-01 (config hidden=312, max_pos=2048).
PINNED_REVISION="97aa9196b313e931424a5401fa42bc69dd15acd9"
BIG_REPO="RUPunct/RUPunct_big"
# Pin: RUPunct/RUPunct_big @ 2024-05-01.
BIG_REVISION="d05f73afd84b57a45b83238c35b866bc625fe247"
REPO="${RUPUNCT_REPO:-$BIG_REPO}"
if [ -n "${RUPUNCT_REVISION:-}" ]; then
  REVISION="$RUPUNCT_REVISION"
elif [ "$REPO" = "$PINNED_REPO" ]; then
  REVISION="$PINNED_REVISION"
elif [ "$REPO" = "$BIG_REPO" ]; then
  REVISION="$BIG_REVISION"
else
  REVISION="main"
fi
HOME_DIR="${HOME:?HOME unset}"
if [ -n "${RUPUNCT_MODEL:-}" ]; then
  DEST="$RUPUNCT_MODEL"
elif [ "$REPO" = "$PINNED_REPO" ]; then
  DEST="$HOME_DIR/.rupunct/punct"
else
  REPO_TAIL=$(echo "$REPO" | awk -F/ '{print $NF}')
  DEST="$HOME_DIR/.rupunct/$REPO_TAIL"
fi
mkdir -p "$DEST"

HF_BIN=""
if command -v hf >/dev/null 2>&1; then
  HF_BIN=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_BIN=huggingface-cli
else
  echo "fetch-model: hf / huggingface-cli not on PATH" >&2
  echo "  pip install -U huggingface_hub" >&2
  echo "  then: hf download $REPO --revision $REVISION --local-dir $DEST" >&2
  exit 1
fi

hf_snapshot() {
  if [ "$HF_BIN" = hf ]; then
    hf download "$REPO" --revision "$REVISION" --local-dir "$DEST" "$@"
  else
    huggingface-cli download "$REPO" --revision "$REVISION" --local-dir "$DEST" "$@"
  fi
}

hf_file() {
  if [ "$HF_BIN" = hf ]; then
    hf download "$REPO" "$1" --revision "$REVISION" --local-dir "$DEST"
  else
    huggingface-cli download "$REPO" "$1" --revision "$REVISION" --local-dir "$DEST"
  fi
}

hf_snapshot --exclude pytorch_model.bin
if [ ! -f "$DEST/model.safetensors" ]; then
  echo "fetch-model: no model.safetensors yet; downloading pytorch_model.bin" >&2
  hf_file pytorch_model.bin
fi

convert_bin() {
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
}

if [ ! -f "$DEST/model.safetensors" ] && [ -f "$DEST/pytorch_model.bin" ]; then
  convert_bin
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

PINNED=0
if [ "$REPO" = "$PINNED_REPO" ] && [ "$REVISION" = "$PINNED_REVISION" ]; then
  PINNED=1
fi

if [ "$PINNED" -ne 1 ]; then
  echo "fetch-model: skipping SHA-256 (repo=$REPO revision=$REVISION; hashes are for $PINNED_REPO@$PINNED_REVISION)" >&2
else
  # Content hashes at RUPunct/RUPunct_small @ $PINNED_REVISION
  check_sha256 "$DEST/config.json" "ec5503b1b544ee19563895c7c73884a4ec94b52b17eda172c11dcdc0eb924f03"
  check_sha256 "$DEST/tokenizer.json" "7ca617388c2092a3a84272025c52bbf3c6db0aee225c0351186295c0b5d3ddc6"
  if [ -f "$DEST/pytorch_model.bin" ]; then
    check_sha256 "$DEST/pytorch_model.bin" "5b05a696d0a9629e0938852e221abcd84cc75ab67a4445fb0b4ea40630dd6b59"
  fi
  if python3 "$HERE/convert-q8.py" --is-q8 "$DEST/model.safetensors"; then
    echo "fetch-model: model.safetensors already Q8_0; skipping F32 sha256" >&2
  elif ! check_sha256 "$DEST/model.safetensors" "1b9d9c252616e79d44eb48883200223e9e77c333a2fa481f618ea9fd32f73c97"; then
    if [ -f "$DEST/pytorch_model.bin" ]; then
      echo "fetch-model: model.safetensors hash mismatch; reconverting from pytorch_model.bin" >&2
      convert_bin
      check_sha256 "$DEST/model.safetensors" "1b9d9c252616e79d44eb48883200223e9e77c333a2fa481f618ea9fd32f73c97"
    else
      echo "fetch-model: model.safetensors hash mismatch and no pytorch_model.bin to rebuild" >&2
      exit 1
    fi
  fi
fi

pack_q8() {
  case "${RUPUNCT_Q8:-1}" in
    0 | false | no | off)
      echo "fetch-model: skipping Q8 (RUPUNCT_Q8=0)" >&2
      return 0
      ;;
  esac
  if ! command -v python3 >/dev/null 2>&1; then
    echo "fetch-model: python3 required to pack Q8_0" >&2
    exit 1
  fi
  conv="$HERE/convert-q8.py"
  if [ ! -f "$conv" ]; then
    echo "fetch-model: missing $conv" >&2
    exit 1
  fi
  tmp="$DEST/model.q8.safetensors"
  python3 "$conv" "$DEST/model.safetensors" "$tmp"
  if [ -f "$tmp" ]; then
    mv "$tmp" "$DEST/model.safetensors"
  fi
}

pack_q8
rm -f "$DEST/pytorch_model.bin"
echo "ok $DEST ($REPO revision $REVISION)"
