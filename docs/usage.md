# Usage

CLI flags, JSONL protocol, library API, windows.

## CLI

```sh
rupunct --text "привет меня зовут анна"
rupunct "привет как дела"
rupunct --strip-punct --lowercase --text "привет, как дела"
rupunct --complete bash >> ~/.bash_completion
rupunct --help
```

| Mode | How |
|------|-----|
| One-shot | `--text`, `[TEXT]`, or `--file` (`-` = stdin) |
| Worker | non-TTY stdin, or `--jsonl` |
| TTY, no input | usage error **2**, not help |

| Flag | Behavior |
|------|----------|
| `--strip-punct` | drop isolated punctuation tokens (off) |
| `--lowercase` | lowercase words before the encoder (off) |
| `--lossy` | on restore failure, emit the original text (exit **0**) |
| `--max-seq` | BERT window, ≥ 3 and ≤ checkpoint (`512` on big, `2048` on small) |
| `--overlap` | must be `< --max-seq` (default **64**) |
| `--threads` | parallel windows / JSONL items, ≥ 1 (default: available parallelism) |

```
0  success (--lossy restore failure is success)
1  load or restore failure
2  usage error
```

One-shot bodies are capped at **8 MiB**. JSONL frames at **1 MiB**. Non-UTF-8 argv is usage (exit **2**).

`--strip-punct` avoids `привет, как` → double marks. `--lowercase` avoids `ООО` → `Ооо`. Both default **off** so the HuggingFace card path is unchanged.

## JSONL

Stdin that is not a TTY, without `--text` / `--file` / `[TEXT]`, is the worker. `--jsonl` forces it. `--text -` is oneshot from stdin.

```text
{"ready":true}
{"id":1,"text":"привет как дела"}
{"id":1,"text":"Привет, как дела?","words":[...]}
{"text":""}
{"error":"text is empty"}
```

Request-level failures (bad JSON, empty `text`, restore error, oversized line, invalid UTF-8) are `{"error":"…"}` frames; the loop continues. Optional `"id"` is echoed. Independent requests run in parallel (`--threads`); responses stay in input order. Only a broken stdin/stdout aborts the worker.

JSONL `--lossy` emits `{"text":…}` instead of `{"error":…}`. Empty JSONL `"text"` is an error frame; oneshot empty input is identity.

Each `words[]` entry: `word`, `text`, `label`, `score`, `start`, `end` (byte offsets into the original string). `score` is softmax, not calibrated.

## Library

```rust
use rupunct::{Punctuator, RestoreOpts, default_model_dir};

let p = Punctuator::load(&default_model_dir())?;
let ru = p.restore("привет как дела")?;
let ru = p.restore_with("привет как дела", RestoreOpts {
    max_window_tokens: 256,
    overlap_tokens: 32,
    ..RestoreOpts::default()
})?;
let words = p.restore_words("привет как дела")?;
let batch = p.restore_batch(&["привет", "как дела"])?;
```

`restore` returns `Err` on inference failure. `restore_lossy` returns the original string. Output is spliced into the original string (newlines, tabs, and hyphens stay; `--strip-punct` deletes those spans).

`words[i].start` / `end` are byte offsets into the original string.

## Windows

Inputs longer than 512 WordPiece tokens (or `max_position_embeddings` if smaller) split on word boundaries with a 64-token overlap. Independent windows run in parallel (`std::thread`; `--threads` / `RestoreOpts.threads`, `0` = auto). JSONL in-flight items use one window thread each so batch × windows does not oversubscribe.

Default **big** allows **512** positions (matches the default window). Small allows **2048**; `--max-seq 2048` is small-only. An explicit `--max-seq` larger than the checkpoint is a CLI error (the library still clamps `RestoreOpts` for tiny test models).
