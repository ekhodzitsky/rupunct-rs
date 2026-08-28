#!/usr/bin/env python3
"""Build a local F1 gold cache from public corpora. Not CI. Not shipped in git.

Written: UD Russian-SynTagRus *test* (CC BY-NC-SA 4.0) — human punctuation.
Spoken-domain: Common Voice ru sentence-collector (CC0) — read-speech prompts.

Golos transcripts are unpunctuated, so they cannot be F1 gold (do not use
Balalaika/RuPunct-restored Golos as gold either — it is circular).

  scripts/fetch-eval.py
  scripts/f1.py --gold ~/.rupunct/eval/f1_gold.json --quiet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

HOME = Path(os.environ.get("HOME") or os.environ.get("USERPROFILE") or ".")
DEFAULT_DEST = HOME / ".rupunct" / "eval"

UD_URL = (
    "https://raw.githubusercontent.com/UniversalDependencies/"
    "UD_Russian-SynTagRus/master/ru_syntagrus-ud-test.conllu"
)
UD_SHA256 = "97f473474d1aa8c007572cd122f90030bc7dc1707579d56b0b354c19b64af258"
CV_URL = (
    "https://raw.githubusercontent.com/common-voice/common-voice/"
    "main/server/data/ru/sentence-collector.txt"
)
CV_SHA256 = "e0869a008310c452f63fd02e4d52f8abc58df2de73e558c486719161e3ad1b87"

PUNCT = set(".,?!:;")
QUOTES = set('«»"“”„‟')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, expect: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and sha256_file(dest) == expect:
        return
    print(f"fetch-eval: downloading {url}", file=sys.stderr)
    urllib.request.urlretrieve(url, dest)
    got = sha256_file(dest)
    if got != expect:
        dest.unlink(missing_ok=True)
        raise SystemExit(f"fetch-eval: {dest} sha256 {got} != {expect}")


def keep(text: str, min_words: int, max_words: int) -> bool:
    t = text.strip()
    if not t or t.startswith("- "):
        return False
    if any(c in t for c in QUOTES):
        return False
    if any("A" <= c <= "Z" or "a" <= c <= "z" for c in t):
        return False
    words = t.split()
    if not (min_words <= len(words) <= max_words):
        return False
    if not any(c in t for c in PUNCT):
        return False
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return False
    cyr = sum(1 for c in letters if "а" <= c.lower() <= "я" or c.lower() == "ё")
    return cyr == len(letters)


def syntagrus_texts(path: Path) -> list[str]:
    out = []
    prefix = "# text = "
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            out.append(line[len(prefix) :].strip())
    return out


def take(texts: list[str], n: int, min_words: int, max_words: int) -> list[str]:
    picked: list[str] = []
    seen: set[str] = set()
    for t in texts:
        if not keep(t, min_words, max_words):
            continue
        key = t.casefold()
        if key in seen:
            continue
        seen.add(key)
        picked.append(t)
        if len(picked) >= n:
            break
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--n-written", type=int, default=400)
    ap.add_argument("--n-asr", type=int, default=400)
    ap.add_argument("--min-words", type=int, default=5)
    ap.add_argument("--max-words", type=int, default=40)
    args = ap.parse_args()

    raw = args.dest / "raw"
    ud_path = raw / "ru_syntagrus-ud-test.conllu"
    cv_path = raw / "cv-ru-sentence-collector.txt"
    download(UD_URL, ud_path, UD_SHA256)
    download(CV_URL, cv_path, CV_SHA256)

    written = take(syntagrus_texts(ud_path), args.n_written, args.min_words, args.max_words)
    asr = take(
        cv_path.read_text(encoding="utf-8").splitlines(),
        args.n_asr,
        args.min_words,
        args.max_words,
    )
    if len(written) < args.n_written or len(asr) < args.n_asr:
        raise SystemExit(
            f"fetch-eval: not enough sentences (written {len(written)}/{args.n_written}, "
            f"asr {len(asr)}/{args.n_asr})"
        )

    cases = []
    for i, t in enumerate(written, 1):
        cases.append({"id": f"ud.{i:04d}", "kind": "written", "gold": t, "source": "syntagrus-test"})
    for i, t in enumerate(asr, 1):
        cases.append({"id": f"cv.{i:04d}", "kind": "asr", "gold": t, "source": "common-voice-ru"})

    gold = args.dest / "f1_gold.json"
    payload = {
        "comment": (
            "Local cache. Written: UD Russian-SynTagRus test (CC BY-NC-SA 4.0). "
            "ASR-style: Common Voice ru sentence-collector (CC0). "
            "Golos transcripts have no punctuation and are not included."
        ),
        "n_written": len(written),
        "n_asr": len(asr),
        "sources": {
            "syntagrus": {"url": UD_URL, "sha256": UD_SHA256, "license": "CC BY-NC-SA 4.0"},
            "common_voice_ru": {"url": CV_URL, "sha256": CV_SHA256, "license": "CC0-1.0"},
        },
        "cases": cases,
    }
    args.dest.mkdir(parents=True, exist_ok=True)
    gold.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(gold), "n_written": len(written), "n_asr": len(asr)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
