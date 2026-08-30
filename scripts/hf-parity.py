#!/usr/bin/env python3
"""Compare rupunct to HuggingFace RUPunct_small (local oracle, not CI).

Needs transformers, torch, a built `rupunct`, and RUPunct_small at
$RUPUNCT_MODEL or ~/.rupunct/punct (not the product default RUPunct_big).

Per case `expect`:
  text   — splice/CLI string equals per-word `process_token` join (not NER grouping)
  labels — first-subtoken id2label equals the HF encoder (hyphen/newline/long)

Flags (--strip-punct / --lowercase) run HF on `hf_input` (encoder-side text).

  scripts/hf-parity.py
  scripts/hf-parity.py --write-golden tests/fixtures/hf_parity_golden.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "tests/fixtures/hf_parity_cases.json"
DEFAULT_GOLDEN = ROOT / "tests/fixtures/hf_parity_golden.json"


def model_dir() -> Path:
    env = os.environ.get("RUPUNCT_MODEL", "").strip()
    if env:
        return Path(env)
    home = Path(os.environ.get("HOME") or os.environ.get("USERPROFILE") or ".")
    return home / ".rupunct" / "punct"


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


def hf_join(words: list[str], labels: list[str]) -> str:
    """Per-word card `process_token`, not HuggingFace entity grouping."""
    return " ".join(process_token(w, lab) for w, lab in zip(words, labels)).strip()


def hf_word_preds(text: str, tokenizer, model, device) -> tuple[list[str], list[str]]:
    """First-subtoken label per pre-tokenizer word (same rule as rupunct)."""
    import torch

    enc = tokenizer(text, return_tensors="pt", truncation=False)
    word_ids = enc.word_ids(0)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
    tens = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        logits = model(**tens).logits[0]
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    first_idx: dict[int, int] = {}
    by_wid: dict[int, list[str]] = {}
    for i, (tok, wid) in enumerate(zip(tokens, word_ids)):
        if wid is None:
            continue
        if wid not in first_idx:
            first_idx[wid] = i
        by_wid.setdefault(wid, []).append(tok)
    labels: list[str] = []
    words: list[str] = []
    for wid in sorted(first_idx):
        i = first_idx[wid]
        labels.append(id2label[int(logits[i].argmax())])
        words.append(tokenizer.convert_tokens_to_string(by_wid[wid]).strip())
    return words, labels


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
        ready = self.proc.stdout.readline()
        try:
            rec = json.loads(ready)
        except json.JSONDecodeError as e:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise SystemExit(f"hf-parity: worker failed: {ready!r} {err}") from e
        if rec.get("ready") is not True:
            raise SystemExit(f"hf-parity: worker not ready: {rec}")

    def restore(self, text: str, cid: str) -> dict:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps({"id": cid, "text": text}, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        rec = json.loads(line)
        if rec.get("error"):
            raise RuntimeError(f"{cid}: {rec['error']}")
        return rec

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def find_bin() -> str | None:
    env = os.environ.get("RUPUNCT_BIN")
    if env:
        return env
    which = shutil.which("rupunct")
    if which:
        return which
    local = ROOT / "target/release/rupunct"
    return str(local) if local.is_file() else None


def load_cases(path: Path, tokenizer) -> list[dict]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for raw in spec["cases"]:
        case = dict(raw)
        case.setdefault("flags", [])
        gen = case.pop("generate", None)
        if gen:
            seed = gen["seed"].strip()
            text = seed
            min_tokens = int(gen["min_tokens"])
            n = 0
            while True:
                n = len(tokenizer.encode(text, add_special_tokens=True))
                if n >= min_tokens:
                    break
                text = f"{text} {seed}"
            case["input"] = text
            case["n_tokens"] = n
        out.append(case)
    return out


def opts_ok(case: dict, hf_text: str, ru_text: str, hf_labs: list[str], ru_labs: list[str]) -> bool:
    if hf_labs != ru_labs:
        return False
    if case.get("expect", "text") == "text":
        return hf_text == ru_text
    return True


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    ap.add_argument("--write-golden", type=Path, nargs="?", const=DEFAULT_GOLDEN)
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError:
        print("hf-parity: install transformers and torch", file=sys.stderr)
        return 2

    md = model_dir()
    if not (md / "config.json").is_file():
        print(f"hf-parity: missing {md}", file=sys.stderr)
        return 2
    bin_path = find_bin()
    if not bin_path:
        print("hf-parity: rupunct not on PATH (set RUPUNCT_BIN)", file=sys.stderr)
        return 2

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tk = AutoTokenizer.from_pretrained(str(md), strip_accents=False, add_prefix_space=True)
    hf_model = AutoModelForTokenClassification.from_pretrained(str(md))
    hf_model.to(device)
    hf_model.eval()

    cases = load_cases(args.cases, tk)
    golden_prev = None
    if args.golden.is_file() and not args.write_golden:
        golden_prev = json.loads(args.golden.read_text(encoding="utf-8"))
        golden_by_id = {c["id"]: c for c in golden_prev.get("cases", [])}
    else:
        golden_by_id = {}

    workers: dict[tuple[str, ...], JsonlWorker] = {}
    failed = 0
    rows = []
    try:
        for case in cases:
            flags = list(case.get("flags") or [])
            key = tuple(flags)
            if key not in workers:
                workers[key] = JsonlWorker(bin_path, md, flags)
            ru = workers[key].restore(case["input"], case["id"])
            hf_src = case.get("hf_input") or case["input"]
            hf_words, hf_labs = hf_word_preds(hf_src, tk, hf_model, device)
            hf_text = hf_join(hf_words, hf_labs)
            ru_text = ru["text"]
            ru_labs = [w["label"] for w in ru.get("words") or []]
            match_labs = hf_labs == ru_labs
            match_text = hf_text == ru_text
            ok = opts_ok(case, hf_text, ru_text, hf_labs, ru_labs)
            rec = {
                "id": case["id"],
                "input": case["input"],
                "flags": flags,
                "expect": case.get("expect", "text"),
                "hf_input": case.get("hf_input"),
                "hf_text": hf_text,
                "rupunct_text": ru_text,
                "hf_labels": hf_labs,
                "rupunct_labels": ru_labs,
                "match_text": match_text,
                "match_labels": match_labs,
                "n_tokens": case.get("n_tokens"),
            }
            if case["id"] in golden_by_id:
                g = golden_by_id[case["id"]]
                if ru_text != g.get("rupunct_text") or ru_labs != g.get("rupunct_labels"):
                    rec["golden_drift"] = True
                    ok = False
            rec["ok"] = ok
            print(
                json.dumps(
                    {
                        "id": rec["id"],
                        "ok": ok,
                        "match_text": match_text,
                        "match_labels": match_labs,
                        "expect": rec["expect"],
                        "n_tokens": rec["n_tokens"],
                    },
                    ensure_ascii=False,
                )
            )
            if not ok:
                failed += 1
                print(
                    json.dumps(
                        {
                            "id": rec["id"],
                            "hf_text": hf_text,
                            "rupunct_text": ru_text,
                            "hf_labels": hf_labs,
                            "rupunct_labels": ru_labs,
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                )
            rows.append(rec)
    finally:
        for w in workers.values():
            w.close()

    if args.write_golden:
        payload = {
            "checkpoint": "RUPunct/RUPunct_small",
            "n_ok": len(rows) - failed,
            "n_failed": failed,
            "cases": rows,
        }
        args.write_golden.parent.mkdir(parents=True, exist_ok=True)
        args.write_golden.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.write_golden}", file=sys.stderr)

    print(json.dumps({"failed": failed, "total": len(rows)}, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
