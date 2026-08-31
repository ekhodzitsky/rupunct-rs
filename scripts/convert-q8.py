#!/usr/bin/env python3
"""Pack large F32 MatMul and embedding weights as GGML Q8_0. LayerNorm / tiny stay F32.

Runtime keeps Q8_0 packed on the mmap; Linear tiles and embedding rows dequant to F32.

  scripts/convert-q8.py ~/.rupunct/RUPunct_big/model.safetensors
  scripts/convert-q8.py in.safetensors out.safetensors

Needs numpy. Writes `{name}.q8_0` (U8) + `{name}.shape` (I64) plus leftover F32.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

QK = 32
BLOCK = 34
QMAX = 127.0
MIN_ELEMENTS = 4096


def parse_st(path: Path) -> tuple[dict, bytes, int]:
    raw = path.read_bytes()
    header_len = struct.unpack_from("<Q", raw, 0)[0]
    header = json.loads(raw[8 : 8 + header_len])
    base = 8 + header_len
    return header, raw, base


def should_quant(name: str, shape: list[int], arr: np.ndarray) -> bool:
    if "LayerNorm" in name:
        return False
    if arr.dtype != np.float32:
        return False
    if arr.size < MIN_ELEMENTS or len(shape) < 2:
        return False
    return int(shape[-1]) % QK == 0


def quant_q8_0(w: np.ndarray) -> bytes:
    blocks = w.astype(np.float32, copy=False).reshape(-1, QK)
    amax = np.max(np.abs(blocks), axis=1)
    d = np.where(amax > 0.0, amax / QMAX, 0.0).astype(np.float16)
    d32 = d.astype(np.float32)
    id_ = np.divide(1.0, d32, out=np.zeros_like(d32), where=d32 != 0.0)
    qs = np.rint(blocks * id_[:, None]).astype(np.int8)
    out = np.empty((blocks.shape[0], BLOCK), dtype=np.uint8)
    out[:, :2] = np.frombuffer(d.tobytes(), dtype=np.uint8).reshape(-1, 2)
    out[:, 2:] = qs.view(np.uint8)
    return out.reshape(-1).tobytes()


def write_st(path: Path, tensors: list[tuple[str, str, list[int], bytes]]) -> None:
    tensors = sorted(tensors, key=lambda t: t[0])
    header: dict = {}
    body = bytearray()
    cursor = 0
    for name, dtype, shape, data in tensors:
        end = cursor + len(data)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, end]}
        body.extend(data)
        cursor = end
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    while len(header_bytes) % 8:
        header_bytes += b" "
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + bytes(body))


def convert(src: Path, dst: Path) -> None:
    header, raw, base = parse_st(src)
    out: list[tuple[str, str, list[int], bytes]] = []
    n_q = 0
    n_f = 0
    src_bytes = 0
    dst_bytes = 0
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        shape = [int(x) for x in spec["shape"]]
        begin, end = spec["data_offsets"]
        payload = raw[base + begin : base + end]
        src_bytes += len(payload)
        dtype = spec["dtype"]
        if dtype == "F32":
            arr = np.frombuffer(payload, dtype="<f4").reshape(shape)
            if should_quant(name, shape, arr):
                wire = quant_q8_0(arr)
                out.append((f"{name}.q8_0", "U8", [len(wire)], wire))
                out.append(
                    (
                        f"{name}.shape",
                        "I64",
                        [len(shape)],
                        np.asarray(shape, dtype="<i8").tobytes(),
                    )
                )
                n_q += 1
                dst_bytes += len(wire) + 8 * len(shape)
                continue
        out.append((name, dtype, shape, bytes(payload)))
        n_f += 1
        dst_bytes += len(payload)
    dst.parent.mkdir(parents=True, exist_ok=True)
    write_st(dst, out)
    print(
        f"convert-q8: {src} -> {dst}  quantized={n_q} f32={n_f}  "
        f"{src_bytes} -> {dst.stat().st_size} bytes",
        file=sys.stderr,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path, nargs="?")
    args = ap.parse_args()
    dst = args.dst
    if dst is None:
        dst = args.src.with_name("model.q8.safetensors")
        if args.src.name != "model.safetensors":
            dst = args.src.with_suffix(".q8.safetensors")
    convert(args.src, dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
