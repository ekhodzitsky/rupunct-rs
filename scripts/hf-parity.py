#!/usr/bin/env python3
"""Compare rupunct CLI restore to HuggingFace RUPunct_small pipeline.

Needs: transformers, torch, and a built `rupunct` on PATH (or RUPUNCT_BIN).
Weights: $RUPUNCT_MODEL or ~/.rupunct/punct.

Exit 0 if every sentence matches; 1 on mismatch; 2 if deps/weights missing.
This is a local oracle, not CI — CI has no PyTorch.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SENTENCES = [
    "привет как дела",
    "привет меня зовут анна сколько будет стоить шестьдесят тысяч тенге",
    "ты пойдешь со мной в кино",
]


def model_dir() -> Path:
    env = os.environ.get("RUPUNCT_MODEL", "").strip()
    if env:
        return Path(env)
    home = Path(os.environ.get("HOME") or os.environ.get("USERPROFILE") or ".")
    return home / ".rupunct" / "punct"


def hf_restore(text: str, clf) -> str:
    preds = clf(text)
    out = []
    for item in preds:
        word = item["word"].strip()
        label = item["entity_group"]
        out.append(process_token(word, label))
    return " ".join(out).strip()


def process_token(token: str, label: str) -> str:
    if label.startswith("UPPER_TOTAL_"):
        cased, rest = token.upper(), label[len("UPPER_TOTAL_") :]
        is_upper = True
    elif label.startswith("UPPER_"):
        cased, rest = token.capitalize(), label[len("UPPER_") :]
        is_upper = True
    elif label.startswith("LOWER_"):
        cased, rest = token, label[len("LOWER_") :]
        is_upper = False
    else:
        return token
    suffix = {
        "O": "",
        "PERIOD": ".",
        "COMMA": ",",
        "QUESTION": "?",
        "VOSKL": "!",
        "DVOETOCHIE": ":",
        "PERIODCOMMA": ";",
        "DEFIS": "-",
        "MNOGOTOCHIE": "...",
        "QUESTIONVOSKL": "?!",
        "TIRE": " —" if is_upper else "—",
    }.get(rest, "")
    return cased + suffix


def rupunct_restore(bin_path: str, model: Path, text: str) -> str:
    p = subprocess.run(
        [bin_path, "--model-dir", str(model), "--text", text],
        check=True,
        capture_output=True,
        text=True,
    )
    return p.stdout.rstrip("\n")


def main() -> int:
    try:
        from transformers import AutoTokenizer, pipeline
    except ImportError:
        print("hf-parity: install transformers and torch", file=sys.stderr)
        return 2
    md = model_dir()
    if not (md / "config.json").is_file():
        print(f"hf-parity: missing {md}", file=sys.stderr)
        return 2
    bin_path = os.environ.get("RUPUNCT_BIN") or shutil.which("rupunct")
    if not bin_path:
        local = Path("target/release/rupunct")
        bin_path = str(local) if local.is_file() else None
    if not bin_path:
        print("hf-parity: rupunct not on PATH (set RUPUNCT_BIN)", file=sys.stderr)
        return 2
    tk = AutoTokenizer.from_pretrained(md, strip_accents=False, add_prefix_space=True)
    clf = pipeline("ner", model=str(md), tokenizer=tk, aggregation_strategy="first")
    failed = 0
    for s in SENTENCES:
        hf = hf_restore(s, clf)
        ru = rupunct_restore(bin_path, md, s)
        ok = hf == ru
        print(json.dumps({"input": s, "hf": hf, "rupunct": ru, "match": ok}, ensure_ascii=False))
        if not ok:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
