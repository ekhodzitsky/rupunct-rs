#!/usr/bin/env python3
"""Word-level punctuation/casing F1 of rupunct vs in-tree gold.

Not CI. Needs a built `rupunct` and weights ($RUPUNCT_MODEL or ~/.rupunct/RUPunct_big).
Gold stays punctuated; the harness lowercases and drops isolated marks, restores,
then scores trailing punctuation and casing on whitespace words.

  scripts/f1.py
  RUPUNCT_MODEL=~/.rupunct/RUPunct_big scripts/f1.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD = ROOT / "tests/fixtures/f1_gold.json"

# Longest-first trailing marks (RUPunct suffix names).
TRAIL = (
    ("?!", "QUESTIONVOSKL"),
    ("...", "MNOGOTOCHIE"),
    (".", "PERIOD"),
    (",", "COMMA"),
    ("?", "QUESTION"),
    ("!", "VOSKL"),
    (":", "DVOETOCHIE"),
    (";", "PERIODCOMMA"),
    ("—", "TIRE"),
    ("-", "DEFIS"),
)

DROP = set('.,?!:;«»"()[]—…')


def model_dir() -> Path:
    env = os.environ.get("RUPUNCT_MODEL", "").strip()
    if env:
        return Path(env)
    home = Path(os.environ.get("HOME") or os.environ.get("USERPROFILE") or ".")
    return home / ".rupunct" / "RUPunct_big"


def find_bin() -> str | None:
    env = os.environ.get("RUPUNCT_BIN")
    if env:
        return env
    which = shutil.which("rupunct")
    if which:
        return which
    local = ROOT / "target/release/rupunct"
    return str(local) if local.is_file() else None


def to_asr(text: str) -> str:
    """Lowercase gold and drop isolated punctuation; keep hyphen and decimals."""
    out: list[str] = []
    n = len(text)
    for i, c in enumerate(text):
        if c == ".":
            left = text[i - 1] if i else ""
            right = text[i + 1] if i + 1 < n else ""
            if left.isdigit() and right.isdigit():
                out.append(c)
            continue
        if c == "-":
            out.append(c)
            continue
        if c in DROP:
            continue
        out.append(c.lower() if c.isalpha() else c)
    return " ".join("".join(out).split())


def parse_word(w: str) -> tuple[str, str, str]:
    punct = "O"
    rest = w
    for suf, name in TRAIL:
        if len(rest) > len(suf) and rest.endswith(suf):
            rest = rest[: -len(suf)]
            punct = name
            break
    letters = [c for c in rest if c.isalpha()]
    if not letters:
        case = "none"
    elif rest == rest.upper() and len(letters) > 1:
        case = "upper"
    elif rest[0].isupper():
        case = "title"
    else:
        case = "lower"
    return rest.lower(), punct, case


def feats(text: str) -> list[tuple[str, str, str]]:
    out = []
    for w in text.split():
        core, punct, case = parse_word(w)
        if core == "":
            continue
        out.append((core, punct, case))
    return out


class JsonlWorker:
    def __init__(self, bin_path: str, md: Path, flags: list[str]) -> None:
        cmd = [bin_path, "--jsonl", "--model-dir", str(md), *flags]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert self.proc.stdout is not None
        rec = json.loads(self.proc.stdout.readline())
        if rec.get("ready") is not True:
            raise SystemExit(f"f1: worker not ready: {rec}")

    def restore(self, text: str, cid: str) -> str:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps({"id": cid, "text": text}, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        rec = json.loads(self.proc.stdout.readline())
        if rec.get("error"):
            raise RuntimeError(f"{cid}: {rec['error']}")
        return rec["text"]

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score(pairs: list[tuple[str, str]]) -> dict:
    """pairs of (gold_class, pred_class)."""
    labels = sorted({g for g, _ in pairs} | {p for _, p in pairs})
    per: dict[str, dict[str, int]] = {lab: {"tp": 0, "fp": 0, "fn": 0} for lab in labels}
    for g, p in pairs:
        if g == p:
            per[g]["tp"] += 1
        else:
            per[g]["fn"] += 1
            per[p]["fp"] += 1
    rows = {}
    tp = fp = fn = 0
    ntp = nfp = nfn = 0
    for lab, c in per.items():
        rows[lab] = {
            "support": c["tp"] + c["fn"],
            "p": round(prf(c["tp"], c["fp"], c["fn"])[0], 4),
            "r": round(prf(c["tp"], c["fp"], c["fn"])[1], 4),
            "f1": round(prf(c["tp"], c["fp"], c["fn"])[2], 4),
        }
        tp += c["tp"]
        fp += c["fp"]
        fn += c["fn"]
        if lab != "O":
            ntp += c["tp"]
            nfp += c["fp"]
            nfn += c["fn"]
    return {
        "per_class": rows,
        "micro_all": round(prf(tp, fp, fn)[2], 4),
        "micro_punct": round(prf(ntp, nfp, nfn)[2], 4),
        "n": len(pairs),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--kind", choices=["written", "asr", "all"], default="all")
    ap.add_argument("--quiet", action="store_true", help="print only the summary JSON")
    args = ap.parse_args()

    md = model_dir()
    if not (md / "config.json").is_file():
        print(f"f1: missing {md}", file=sys.stderr)
        return 2
    bin_path = find_bin()
    if not bin_path:
        print("f1: rupunct not on PATH (set RUPUNCT_BIN)", file=sys.stderr)
        return 2

    spec = json.loads(args.gold.read_text(encoding="utf-8"))
    cases = spec["cases"]
    if args.kind != "all":
        cases = [c for c in cases if c.get("kind") == args.kind]

    worker = JsonlWorker(bin_path, md, [])
    punct_pairs: list[tuple[str, str]] = []
    case_pairs: list[tuple[str, str]] = []
    exact = 0
    aligned = 0
    skipped = 0
    by_kind: dict[str, list[bool]] = defaultdict(list)
    try:
        for case in cases:
            gold = case["gold"]
            src = to_asr(gold)
            pred = worker.restore(src, case["id"])
            g = feats(gold)
            p = feats(pred)
            ok = pred == gold
            if ok:
                exact += 1
            if [x[0] for x in g] != [x[0] for x in p]:
                skipped += 1
                if not args.quiet:
                    print(
                        json.dumps(
                            {
                                "id": case["id"],
                                "align": False,
                                "gold": gold,
                                "input": src,
                                "pred": pred,
                            },
                            ensure_ascii=False,
                        )
                    )
                continue
            aligned += 1
            by_kind[case.get("kind", "written")].append(ok)
            for (gc, gp, gcase), (_, pp, pcase) in zip(g, p):
                punct_pairs.append((gp, pp))
                if gcase != "none" and pcase != "none":
                    case_pairs.append((gcase, pcase))
            if not args.quiet:
                print(
                    json.dumps(
                        {
                            "id": case["id"],
                            "exact": ok,
                            "gold": gold,
                            "pred": pred,
                        },
                        ensure_ascii=False,
                    )
                )
    finally:
        worker.close()

    punct = score(punct_pairs)
    casing = score(case_pairs) if case_pairs else {}
    case_acc = (
        sum(1 for g, p in case_pairs if g == p) / len(case_pairs) if case_pairs else 0.0
    )
    summary = {
        "n": len(cases),
        "exact": exact,
        "exact_rate": round(exact / len(cases), 4) if cases else 0.0,
        "aligned": aligned,
        "skipped_align": skipped,
        "punct": punct,
        "case_accuracy": round(case_acc, 4),
        "casing": casing,
        "by_kind": {
            k: {"n": len(v), "exact": sum(v), "exact_rate": round(sum(v) / len(v), 4)}
            for k, v in sorted(by_kind.items())
        },
        "model": str(md),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
