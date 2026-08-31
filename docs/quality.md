# Quality

Local numbers. Not CI. The engine is HuggingFace-faithful; the ceiling is the checkpoint.

## Examples (pinned `RUPunct_big`)

| Input | Output |
|-------|--------|
| `привет как дела` | `Привет, как дела?` |
| `ты пойдешь со мной в кино` | `Ты пойдешь со мной в кино?` |
| already-punctuated `привет, как дела` | double marks unless `--strip-punct` |

## HuggingFace parity

`scripts/hf-parity.py` (transformers required, not CI) is **39/39** on
`tests/fixtures/hf_parity_golden.json`: first-subtoken labels match the encoder.
Space-separated inputs also match a per-word `process_token` join.

Splice keeps hyphens, newlines, and tabs; a space-join does not. Consecutive
identical tags are **not** merged the way `pipeline(..., aggregation_strategy="first")`
groups entities. Live goldens (`cargo test --lib -- --ignored`) lock **small**
strings at `~/.rupunct/punct`; they are not a linguistic oracle.

## F1

`scripts/f1.py`. Gold stays punctuated; the harness lowercases, drops isolated
marks, restores, then scores trailing punctuation and casing on whitespace words.

The 31-sentence in-tree set (`tests/fixtures/f1_gold.json`) is a **smoke** test
and ranked small above big; that ranking does **not** hold on public corpora.

External gold (local cache, not git): 400 sentences from
[UD Russian-SynTagRus](https://github.com/UniversalDependencies/UD_Russian-SynTagRus)
test (CC BY-NC-SA 4.0) and 400 [Common Voice](https://github.com/common-voice/common-voice)
ru prompts (CC0). Golos transcripts have **no** punctuation, so they cannot be F1 gold.

| checkpoint | split | n | exact | punct micro-F1 (excl. `O`) | case acc. |
|------------|-------|---|------:|---------------------------:|----------:|
| small | SynTagRus test | 400 | 40% | 0.78 | 97% |
| small | Common Voice ru | 400 | 53% | 0.76 | 98% |
| small | all | 800 | 46% | 0.77 | 98% |
| big | SynTagRus test | 400 | 54% | 0.86 | 98% |
| big | Common Voice ru | 400 | 62% | 0.83 | 98% |
| big | all | 800 | **58%** | **0.85** | 98% |

Same 800 sentences, `scripts/f1.py --system` (not CI):

| system | exact | punct micro-F1 | case acc. |
|--------|------:|---------------:|----------:|
| rupunct `RUPunct_big` | **58%** | **0.85** | 98% |
| [kontur-ai/sbert_punc_case_ru](https://huggingface.co/kontur-ai/sbert_punc_case_ru) | 49% | 0.79 | 98% |
| [Silero TE](https://github.com/snakers4/silero-models) | 37% | 0.75 | 97% |

Kontur predicts only `. , ?`. Silero TE predicts `.,-!?—` (quantized package; some
hyphen inputs crash and score as identity). Training domains differ; this gold is
written UD + read-speech prompts. Not a claim of SOTA on every Russian ASR domain.

```sh
scripts/fetch-eval.py
scripts/f1.py --gold ~/.rupunct/eval/f1_gold.json --quiet
scripts/f1.py --system kontur --gold ~/.rupunct/eval/f1_gold.json --quiet
scripts/f1.py --system silero --gold ~/.rupunct/eval/f1_gold.json --quiet
scripts/hf-parity.py
```
