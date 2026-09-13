"""Stage 3 — the `general` (FineWeb-Edu) and `haiku` slices.

FineWeb-Edu is already quality- and language-filtered upstream, so "cleaning" it
means: minimal whitespace normalisation, drop docs under ``--min-chars`` (200),
dedup on a hash of the first 1k chars, and stop once the token target is hit.

The number of raw FineWeb-Edu rows this stage *consumed* is recorded in
``data/clean/general/_complete.json`` as ``fineweb_docs_scanned``.
``topic_filter.py apply`` starts its scan at that offset, which is what keeps the
`psych` slice disjoint from the `general` slice.

Haiku sources (all three inspected; each uses a different line separator):
  statworx/haiku       text            lines joined by ' / '
  dugward/english_haiku content        lines joined by ' \\ '   (some rows have no
                                       separator at all and are dropped)
  huanggab/reddit_haiku processed_title lines joined by '/'
Output format is one haiku per doc, lines separated by a single newline.

Usage
-----
    python -m src.data.clean_general --domains general,haiku
    python -m src.data.clean_general --domains general --target-tokens 3.6e9
    python -m src.data.clean_general --limit 500 --force        # smoke test
"""

from __future__ import annotations

import argparse
import re
import sys

from src.config import load_config
from src.data import common as C

MIN_CHARS = 200
# Extra headroom over the ratio target, so mix.py has room to carve val + slack.
SLACK = 1.08

HAIKU_SEPARATORS = {
    "haiku_statworx": (["text", "text_punc"], re.compile(r"\s*/\s*")),
    "haiku_dugward": (["content"], re.compile(r"\s*\\+\s*")),
    "haiku_reddit": (["processed_title"], re.compile(r"\s*/\s*")),
}
HAIKU_MIN_CHARS = 12
HAIKU_MAX_CHARS = 220
HAIKU_MAX_LINE_CHARS = 90


def default_target(domain: str, cfg_path: str | None = None) -> int:
    cfg = load_config(cfg_path)
    ratio = cfg.data.mix.get(domain, 0.0)
    return int(cfg.train.total_tokens * ratio * SLACK) + cfg.data.val_tokens_per_domain


# --------------------------------------------------------------------------- #

def clean_general(
    target_tokens: int, limit: int | None, min_chars: int, force: bool
) -> dict:
    outdir = C.CLEAN_DIR / "general"
    if C.is_complete(outdir) and not force:
        print("[general] already complete — skipping (--force to redo)")
        return C.read_marker(outdir)
    C.clear_dir(outdir)

    writer = C.ShardWriter(outdir, "general")
    seen: set[str] = set()
    scanned = kept = dropped_short = dropped_dup = 0
    tokens = 0

    for row in C.progress(C.iter_source("fineweb_edu", limit=limit), desc="general"):
        scanned += 1
        text = C.norm_ws(row.get("text") or "")
        if len(text) < min_chars:
            dropped_short += 1
            continue
        key = C.doc_hash(text)
        if key in seen:
            dropped_dup += 1
            continue
        seen.add(key)
        writer.write({
            "text": text,
            "meta": {
                "source": "HuggingFaceFW/fineweb-edu:sample-10BT",
                "id": row.get("id"),
                "url": row.get("url"),
                "edu_score": row.get("score"),
            },
        })
        kept += 1
        tokens += C.est_tokens(len(text))
        if tokens >= target_tokens:
            break

    stats = writer.close()
    stats.update(
        domain="general",
        source="HuggingFaceFW/fineweb-edu:sample-10BT",
        fineweb_docs_scanned=scanned,      # <-- psych starts here
        target_tokens=target_tokens,
        dropped_short=dropped_short,
        dropped_dup=dropped_dup,
    )
    C.write_marker(outdir, stats)
    print(
        f"[general] kept {kept:,}/{scanned:,} docs (~{C.human(stats['est_tokens'])} tokens); "
        f"dropped {dropped_short:,} short, {dropped_dup:,} dup. "
        f"psych scan offset = {scanned:,}"
    )
    if tokens < target_tokens:
        print(
            f"[general] WARNING: only ~{C.human(tokens)} of {C.human(target_tokens)} "
            "target tokens — the FineWeb-Edu raw slice ran out.",
            file=sys.stderr,
        )
    return stats


# --------------------------------------------------------------------------- #

def split_haiku(raw: str, splitter: re.Pattern) -> list[str] | None:
    if not raw:
        return None
    text = C.norm_ws(raw.replace("\\n", "\n"))
    parts = [p.strip() for p in splitter.split(text) if p.strip()] if splitter.search(text) else None
    if parts is None and "\n" in text:
        parts = [p.strip() for p in text.split("\n") if p.strip()]
    if not parts or len(parts) != 3:
        return None
    if any(len(p) > HAIKU_MAX_LINE_CHARS or len(p) < 2 for p in parts):
        return None
    return parts


def clean_haiku(limit: int | None, force: bool) -> dict:
    outdir = C.CLEAN_DIR / "haiku"
    if C.is_complete(outdir) and not force:
        print("[haiku] already complete — skipping (--force to redo)")
        return C.read_marker(outdir)
    C.clear_dir(outdir)

    writer = C.ShardWriter(outdir, "haiku")
    seen: set[str] = set()
    per_source: dict[str, dict[str, int]] = {}

    for name, (fields, splitter) in HAIKU_SEPARATORS.items():
        counts = {"scanned": 0, "kept": 0, "unsplittable": 0, "not_english": 0, "dup": 0}
        try:
            rows = C.iter_source(name, limit=limit)
            for row in C.progress(rows, desc=name):
                counts["scanned"] += 1
                raw = next((row[f] for f in fields if row.get(f)), None)
                lines = split_haiku(raw or "", splitter)
                if lines is None:
                    counts["unsplittable"] += 1
                    continue
                text = "\n".join(lines)
                if not (HAIKU_MIN_CHARS <= len(text) <= HAIKU_MAX_CHARS):
                    counts["unsplittable"] += 1
                    continue
                if not C.looks_english(text, min_ascii_ratio=0.95, require_stopword=False):
                    counts["not_english"] += 1
                    continue
                key = C.doc_hash(text, n=200)
                if key in seen:
                    counts["dup"] += 1
                    continue
                seen.add(key)
                writer.write({
                    "text": text,
                    "meta": {"source": C.SOURCES[name]["repo"], "dataset": name},
                })
                counts["kept"] += 1
        except Exception as exc:
            counts["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[haiku:{name}] FAILED: {counts['error']}", file=sys.stderr)
        per_source[name] = counts
        print(f"[haiku:{name}] kept {counts['kept']:,}/{counts['scanned']:,} "
              f"(unsplittable {counts['unsplittable']:,}, dup {counts['dup']:,})")

    stats = writer.close()
    stats.update(domain="haiku", per_source=per_source)
    C.write_marker(outdir, stats)
    print(f"[haiku] total {stats['docs']:,} haikus, ~{C.human(stats['est_tokens'])} est tokens")
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", default="general,haiku")
    ap.add_argument("--target-tokens", type=float, default=None,
                    help="general slice size (default: mix ratio x total_tokens x slack)")
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS)
    ap.add_argument("--limit", type=int, default=None, help="max raw docs to scan (smoke testing)")
    ap.add_argument("--config", default=None, help="configs/*.toml to read the mix from")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    unknown = set(domains) - {"general", "haiku"}
    if unknown:
        ap.error(f"unknown domain(s): {sorted(unknown)}; this stage owns general+haiku")

    if "general" in domains:
        target = int(args.target_tokens) if args.target_tokens else default_target("general", args.config)
        print(f"[general] target ~{C.human(target)} est tokens")
        clean_general(target, args.limit, args.min_chars, args.force)
    if "haiku" in domains:
        clean_haiku(args.limit, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
