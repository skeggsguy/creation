# Data pipeline

Builds a ~7B-token (6B train + val + slack) clean-licensed English corpus in the
domain ratios from `DataConfig.mix` in `src/config.py`. The output is a set of
jsonl shards plus `data/clean/manifest.json`, which the tokenizer stage consumes.
**Nothing here concatenates or shuffles the corpus** — the manifest is the
hand-off.

Run everything from the repo root:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run python -m src.data.<module> ...
```

## Layout

```
data/raw/hf_cache/            HF_HOME (set automatically by src/data/common.py)
data/raw/<source>/            *.jsonl.gz raw shards + _complete.json marker
data/clean/<domain>/          *.jsonl clean shards, _val_<domain>.jsonl, markers
data/clean/psych_classifier/  fastText (or sklearn) topic model + meta.json
data/clean/manifest.json      ordered [{file, domain, split, epochs, est_tokens, docs}]
data/clean/stats.md           the same, as a table
```

Every clean line is `{"text": str, "meta": {...}}`. Shards roll at ~200 MB.
Token counts everywhere are estimates: `len(text) / 4`. Real counts come from the
tokenizer stage.

## Pipeline order

| # | Module | Produces |
|---|---|---|
| 1 | `download.py` | `data/raw/<source>/` |
| 2 | `gutenberg_filter.py` | `clean/philosophy`, `clean/comedy`, `clean/scifi` |
| 3 | `clean_general.py` | `clean/general`, `clean/haiku` |
| 4 | `topic_filter.py train` / `apply` | `clean/psych_classifier`, `clean/psych` |
| 4 | `topic_filter.py tilt-cosmopedia` | `clean/textbooks` |
| 5 | `mix.py` | `clean/manifest.json`, `clean/stats.md` |

**Stage 3 must run before stage 4 `apply`.** `clean_general.py` records how many
FineWeb-Edu rows it consumed in `data/clean/general/_complete.json`
(`fineweb_docs_scanned`); `topic_filter apply` starts its scan at that offset,
which is the only thing keeping the `psych` slice disjoint from `general`.
Stages 2, 3 and `tilt-cosmopedia` are independent of each other.

## Full-size run

Sizes below are what lands in `data/raw/`. Every source is read in streaming
mode, which does **not** populate `data/raw/hf_cache` (measured at ~100 KB after
a full smoke run), so there is no second copy to budget for.

Note that `cosmopedia` is deliberately **not** downloaded: only
`tilt-cosmopedia` reads it, and it keeps ~5% of what it scans, so staging the
raw scan would cost ~35 GB to produce ~3 GB of output. `iter_source` falls back
to Hub streaming when a source has no local shards, so the tilt just streams it.

```bash
# 1. downloads — ~27 GB.  Resumable: re-running skips sources that already
#    have a _complete.json marker.
uv run python -m src.data.download --sources gutenberg,scifi,haiku   # ~7 GB
uv run python -m src.data.download --sources fineweb_edu             # ~20 GB (9.5B token cap)

# 2. Gutenberg domains (~20 min over the full dump)
uv run python -m src.data.gutenberg_filter --domains philosophy,comedy,scifi

# 3. general + haiku.  Writes the FineWeb offset that stage 4 needs.
uv run python -m src.data.clean_general --domains general,haiku

# 4. psych + textbooks
uv run python -m src.data.topic_filter train --scan-docs 300000
uv run python -m src.data.topic_filter apply                  # reads the offset from stage 3
uv run python -m src.data.topic_filter tilt-cosmopedia         # streams from the Hub

# 5. val carve + mix plan
uv run python -m src.data.mix
cat data/clean/stats.md
```

Per-source download estimates:

| source | repo | on-disk (raw shards) |
|---|---|---|
| `fineweb_edu` | HuggingFaceFW/fineweb-edu `sample-10BT` | ~20 GB (capped at 9.5B tokens) |
| `cosmopedia` | HuggingFaceTB/smollm-corpus `cosmopedia-v2` | not staged — streamed by `tilt-cosmopedia` |
| `gutenberg` | sedthh/gutenberg_english | ~6 GB (~46k books) |
| `scifi` | stevez80/Sci-Fi-Books-gutenberg | ~0.5 GB |
| `haiku_*` | statworx / dugward / huanggab | < 50 MB total |

## Flags

Shared conventions: every stage takes `--limit N` (cap raw docs scanned — this is
the smoke-test knob) and `--force` (redo a stage that is already marked complete).
Stages are idempotent: a finished output directory carries `_complete.json` and is
skipped otherwise. Partial shards are written as `*.part` and ignored by readers,
so an interrupted run never leaves half a shard behind.

### `download.py`
- `--sources all | haiku | a,b,c` — `--list` shows names and completion state.
- `--max-tokens N` — per-source scan budget (defaults in `DEFAULT_MAX_TOKENS`).

### `gutenberg_filter.py`
- `--domains philosophy,comedy,scifi`
- `--max-chunk-chars 50000` — books longer than this are chunked on paragraph
  boundaries, one doc per chunk.
- Allowlists (`PHILOSOPHY_AUTHORS`, `PHILOSOPHY_SUBJECTS`, `COMEDY_AUTHORS`,
  `COMEDY_SUBJECTS`, LoC class prefixes, `VETO_SUBJECTS`, `FICTION_SUBJECTS`) are
  plain module constants at the top of the file — edit them directly.

### `clean_general.py`
- `--domains general,haiku`
- `--target-tokens` — defaults to `mix_ratio x total_tokens x 1.08 + val_tokens`.
- `--min-chars 200` — FineWeb-Edu is already quality-filtered, so this plus a
  first-1k-chars dedup is the whole "cleaning" step.

### `topic_filter.py`
- `train --scan-docs --n-pos --n-neg --min-distinct` — bootstraps seeds by
  keyword-scoring FineWeb-Edu, then trains fastText supervised; falls back to
  TF-IDF + LogisticRegression if `import fasttext` fails. Holdout precision /
  recall / F1 land in `psych_classifier/meta.json`.
- `apply --threshold 0.70 --target-tokens --skip-docs` — `--skip-docs` defaults
  to the general slice's offset; override only if you know what you're doing.
- `tilt-cosmopedia --tilt-fraction 0.6 --filler-every 10` — topical docs fill
  60% of the slice, then 1-in-N general Cosmopedia docs fill the rest.
  `COSMO_KEYWORDS` is the topic list.

### `mix.py`
- `--total-tokens`, `--val-tokens`, `--max-domain-epochs` override config.
- `--no-val` re-plans without touching the val split; `--force-val` re-carves it.

## Source choices and gotchas

**Gutenberg: `sedthh/gutenberg_english`, not `common-pile/project_gutenberg`.**
Both were inspected by streaming rows. common-pile exposes only
`metadata.{license, language, url, title}` — a title and nothing else, so
selecting philosophy or comedy from it would be title-string guessing. sedthh
exposes `METADATA` as a JSON string with `authors`, `subjects` (real LoC subject
headings), `locc` (LoC class letters — `B*` is philosophy/psychology/ethics,
`PN6*` is wit and humour) and `bookshelves`. That is exactly what the allowlists
need.

Gotchas that the code works around, each found by inspecting real rows:

- **`stevez80` string columns are Python `repr`s.** `text` arrives as the literal
  string `"b'\xef\xbb\xbfThe Project Gutenberg eBook of...'"`, title and author as
  `"'Shelley, Mary Wollstonecraft'"`. `common.maybe_literal` unwraps them.
- **`sedthh` puts a blank line between every hard-wrapped line.** Left alone, a
  70-column novel becomes one paragraph per line.
  `common.undouble_blank_lines` detects it (blank-line breaks followed by
  lowercase) and `common.unwrap_hard_wraps` re-flows the prose, joining only
  same-indentation, non-terminated, long lines so verse and dialogue survive.
- **Substring matching on subject headings is a trap.** `"logic"` matches inside
  `"psychoLOGICal fiction"`, which dropped every novel catalogued that way into
  the philosophy pile (Moby-Dick, Persuasion, The Scarlet Letter). Subject
  matching is word-bounded, and philosophy additionally vetoes any fiction
  heading unless the LoC class is a B-class.
- **Cosmopedia-v2 has no topic column.** The config exposes only
  `{prompt, text, token_length, audience, format, seed_data}`, and `seed_data` is
  the source corpus ("fineweb"), not a subject. The tilt therefore keyword-scores
  the `prompt` (which quotes the seed extract) plus the head of the generated
  text.
- **The three haiku sets use three different line separators**: `' / '`
  (statworx `text`), `' \ '` (dugward `content`), `'/'` (huanggab
  `processed_title`). Some dugward rows have no separator at all and are dropped.
  Output is one haiku per doc, lines joined by `\n`.
- **Curly apostrophes.** Transcriber-note stripping matches `'`, `’`, `` ` ``.

## Determinism

Seed 1337 (`common.SEED`) everywhere. Every stage reads its input in shard order
and writes in that same order; the only randomness is negative-sampling in
`topic_filter train` and the val-carve boundary, both seeded. Re-running a stage
on the same input produces byte-identical output.
