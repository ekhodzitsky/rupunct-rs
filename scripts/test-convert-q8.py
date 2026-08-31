#!/usr/bin/env python3
"""Smoke convert-q8.py: pack, already-packed, padded last dim, nothing-to-pack."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONV = ROOT / "scripts" / "convert-q8.py"


def load_conv():
    spec = importlib.util.spec_from_file_location("convert_q8", CONV)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {CONV}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def f32(name: str, arr: np.ndarray) -> tuple[str, str, list[int], bytes]:
    a = np.ascontiguousarray(arr, dtype="<f4")
    return name, "F32", [int(x) for x in a.shape], a.tobytes()


def names(mod, path: Path) -> set[str]:
    header, _, _ = mod.parse_st(path)
    return {n for n in header if n != "__metadata__"}


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CONV), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    mod = load_conv()
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        pack_src = tmp / "pack.safetensors"
        pack_dst = tmp / "pack.q8.safetensors"
        w = rng.standard_normal((128, 32), dtype=np.float32) * 0.05
        ln = rng.standard_normal((32,), dtype=np.float32)
        bias = rng.standard_normal((2, 8), dtype=np.float32)
        mod.write_st(
            pack_src,
            [
                f32("classifier.weight", w),
                f32("bert.encoder.layer.0.attention.output.LayerNorm.weight", ln),
                f32("tiny.bias", bias),
            ],
        )
        conv = run_cli(str(pack_src), str(pack_dst))
        if conv.returncode != 0:
            raise SystemExit(conv.stderr)
        got = names(mod, pack_dst)
        if "classifier.weight.q8_0" not in got or "classifier.weight.shape" not in got:
            raise SystemExit(f"pack missing q8 tensors: {got}")
        if "classifier.weight" in got:
            raise SystemExit("pack left F32 classifier.weight")
        if "bert.encoder.layer.0.attention.output.LayerNorm.weight" not in got:
            raise SystemExit("LayerNorm was packed")
        if run_cli("--is-q8", str(pack_dst)).returncode != 0:
            raise SystemExit("--is-q8 packed should be 0")
        if run_cli("--is-q8", str(pack_src)).returncode == 0:
            raise SystemExit("--is-q8 F32 should be 1")

        again = tmp / "already.safetensors"
        conv = run_cli(str(pack_dst), str(again))
        if conv.returncode != 0:
            raise SystemExit(conv.stderr)
        if "already packed" not in conv.stderr:
            raise SystemExit(f"expected already packed: {conv.stderr}")
        if again.exists():
            raise SystemExit("already-packed wrote dest")

        pad_src = tmp / "pad.safetensors"
        pad_dst = tmp / "pad.q8.safetensors"
        wp = rng.standard_normal((128, 40), dtype=np.float32) * 0.05
        mod.write_st(pad_src, [f32("dense.weight", wp)])
        conv = run_cli(str(pad_src), str(pad_dst))
        if conv.returncode != 0:
            raise SystemExit(conv.stderr)
        header, raw, base = mod.parse_st(pad_dst)
        spec = header["dense.weight.q8_0"]
        begin, end = spec["data_offsets"]
        # 128 rows × pad_k(40)=64 → 2 blocks/row × 34
        expect = 128 * 2 * 34
        if end - begin != expect:
            raise SystemExit(f"padded wire {end - begin} != {expect}")
        sb0, sb1 = header["dense.weight.shape"]["data_offsets"]
        shape = np.frombuffer(raw[base + sb0 : base + sb1], dtype="<i8")
        if list(shape) != [128, 40]:
            raise SystemExit(f"stored shape {list(shape)} != [128, 40]")

        skip_src = tmp / "skip.safetensors"
        skip_dst = tmp / "skip.q8.safetensors"
        tiny = rng.standard_normal((8, 32), dtype=np.float32)
        mod.write_st(skip_src, [f32("tiny.weight", tiny)])
        conv = run_cli(str(skip_src), str(skip_dst))
        if conv.returncode != 0:
            raise SystemExit(conv.stderr)
        if "nothing to pack" not in conv.stderr:
            raise SystemExit(f"expected nothing to pack: {conv.stderr}")
        if skip_dst.exists():
            raise SystemExit("nothing-to-pack wrote dest")

    print("OK: convert-q8 pack / already-packed / padded-k / nothing-to-pack")
    return 0


if __name__ == "__main__":
    sys.exit(main())
