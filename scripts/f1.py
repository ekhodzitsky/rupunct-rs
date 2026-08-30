#!/usr/bin/env python3
"""Word-level punctuation/casing F1 vs gold.

Not CI. Gold stays punctuated; the harness lowercases and drops isolated marks,
restores, then scores trailing punctuation and casing on whitespace words.

  scripts/f1.py
  RUPUNCT_MODEL=~/.rupunct/RUPunct_big scripts/f1.py
  scripts/f1.py --gold ~/.rupunct/eval/f1_gold.json --quiet
  scripts/f1.py --system kontur --gold ~/.rupunct/eval/f1_gold.json --quiet
  scripts/f1.py --system silero --gold ~/.rupunct/eval/f1_gold.json --quiet

`--system rupunct` needs a built binary and weights.
`--system kontur` needs torch+transformers (`kontur-ai/sbert_punc_case_ru`).
`--system silero` needs torch (`snakers4/silero-models` TE, cached under ~/.rupunct).
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


def kontur_render(token: str, label: str) -> str:
    if label.startswith("UPPER_TOTAL_"):
        cased = token.upper()
        rest = label[len("UPPER_TOTAL_") :]
    elif label.startswith("UPPER_"):
        cased = token[:1].upper() + token[1:] if token else token
        rest = label[len("UPPER_") :]
    elif label.startswith("LOWER_"):
        cased = token
        rest = label[len("LOWER_") :]
    else:
        return token
    suf = {"O": "", "PERIOD": ".", "COMMA": ",", "QUESTION": "?"}.get(rest, "")
    return cased + suf


class KonturWorker:
    """Official first-subtoken restore for kontur-ai/sbert_punc_case_ru."""

    def __init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as e:
            raise SystemExit(
                "f1 --system kontur needs torch and transformers "
                "(Python 3.12 venv; not CI)"
            ) from e
        repo = "kontur-ai/sbert_punc_case_ru"
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(repo, strip_accents=False)
        self.model = AutoModelForTokenClassification.from_pretrained(repo)
        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.id2 = {}
        for k, v in self.model.config.id2label.items():
            try:
                self.id2[int(k)] = v
            except ValueError:
                continue

    def restore(self, text: str, _cid: str) -> str:
        return self._punctuate(text)

    def close(self) -> None:
        return None

    def _punctuate(self, text: str) -> str:
        text = text.strip().lower()
        words = text.split()
        if not words:
            return text
        enc = self.tokenizer(words, is_split_into_words=True)
        ids = enc["input_ids"]
        if len(ids) > 512:
            mid = max(1, len(words) // 2)
            left = self._punctuate(" ".join(words[:mid]))
            right = self._punctuate(" ".join(words[mid:]))
            return f"{left} {right}".strip()
        mask = enc["attention_mask"]
        with self.torch.no_grad():
            logits = self.model(
                input_ids=self.torch.tensor([ids], device=self.device),
                attention_mask=self.torch.tensor([mask], device=self.device),
            ).logits
        pred = logits.argmax(dim=-1)[0].tolist()
        word_ids = enc.word_ids()
        out: list[str] = []
        for i, word in enumerate(words):
            try:
                pos = word_ids.index(i)
            except ValueError:
                out.append(word)
                continue
            label = self.id2.get(int(pred[pos]), "LOWER_O")
            out.append(kontur_render(word, label))
        return " ".join(out)


SILERO_TE_URL = "https://models.silero.ai/te_models/v2_4lang_q.pt"
SILERO_TE_NAME = "v2_4lang_q.pt"


class SileroWorker:
    """snakers4/silero-models text enhancement (`enhance_text`, lan=ru)."""

    def __init__(self) -> None:
        try:
            import torch
            from torch import package
        except ImportError as e:
            raise SystemExit(
                "f1 --system silero needs torch (Python 3.12 venv; not CI)"
            ) from e
        engines = torch.backends.quantized.supported_engines
        if "fbgemm" in engines:
            torch.backends.quantized.engine = "fbgemm"
        elif "qnnpack" in engines:
            torch.backends.quantized.engine = "qnnpack"
        home = Path(os.environ.get("HOME") or os.environ.get("USERPROFILE") or ".")
        dest = home / ".rupunct" / SILERO_TE_NAME
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.is_file() or dest.stat().st_size < 10_000:
            torch.hub.download_url_to_file(SILERO_TE_URL, str(dest), progress=True)
        imp = package.PackageImporter(str(dest))
        self.model = imp.load_pickle("te_model", "model")

    def restore(self, text: str, _cid: str) -> str:
        import contextlib
        import io

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                out = self.model.enhance_text(text, "ru")
        except (IndexError, RuntimeError, ValueError):
            return text
        if not isinstance(out, str):
            out = str(out)
        return out

    def close(self) -> None:
        return None


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
    ap.add_argument(
        "--system", choices=["rupunct", "kontur", "silero"], default="rupunct"
    )
    ap.add_argument("--quiet", action="store_true", help="print only the summary JSON")
    args = ap.parse_args()

    md = model_dir()
    spec = json.loads(args.gold.read_text(encoding="utf-8"))
    cases = spec["cases"]
    if args.kind != "all":
        cases = [c for c in cases if c.get("kind") == args.kind]

    if args.system == "kontur":
        worker = KonturWorker()
        model_name = "kontur-ai/sbert_punc_case_ru"
    elif args.system == "silero":
        worker = SileroWorker()
        model_name = "snakers4/silero-models te"
    else:
        if not (md / "config.json").is_file():
            print(f"f1: missing {md}", file=sys.stderr)
            return 2
        bin_path = find_bin()
        if not bin_path:
            print("f1: rupunct not on PATH (set RUPUNCT_BIN)", file=sys.stderr)
            return 2
        worker = JsonlWorker(bin_path, md, [])
        model_name = str(md)
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
        "model": model_name,
        "system": args.system,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
