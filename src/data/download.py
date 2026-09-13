"""Stage 1 — download raw corpora into ``data/raw/<source>/``.

Every source is streamed from the Hub (all of them are bigger than what we need,
except the tiny haiku sets) and written out as gzipped jsonl shards.  Downstream
stages read those shards, so the network is hit exactly once per source.

Resumable / idempotent
----------------------
A source directory that already has a ``_complete.json`` marker is skipped.
Re-run with ``--force`` to redo one.  Interrupted runs leave a ``*.part`` file
which is ignored by every reader and overwritten on the next attempt.

Usage
-----
    python -m src.data.download --list
    python -m src.data.download --sources all --limit 200      # smoke test
    python -m src.data.download --sources fineweb_edu --max-tokens 8e9
    python -m src.data.download --sources gutenberg,scifi,haiku
"""

from __future__ import annotations

import argparse
import sys
import time

from src.data import common as C

HAIKU_SOURCES = ["haiku_statworx", "haiku_dugward", "haiku_reddit"]
ALL_SOURCES = list(C.SOURCES)

# Defaults sized for the 6B-token target in TrainConfig, with slack.  These are
# *scan* budgets: the cleaning stages select from within them.
DEFAULT_MAX_TOKENS = {
    # Worth a local copy: `general` and `psych` each make a full pass over it.
    # general needs 0.55 * 6B = 3.3B; psych filters the remainder of the stream.
    "fineweb_edu": 9_500_000_000,
    # NOT worth a local copy: only `tilt-cosmopedia` reads it, and it keeps ~5%
    # of what it scans, so staging the raw scan on disk costs ~35 GB to produce
    # ~3 GB of output.  Skip this source and let `topic_filter tilt-cosmopedia`
    # stream it directly (iter_source falls back to the Hub automatically).
    # This budget only applies if you download it anyway.
    "cosmopedia": 9_000_000_000,
}


def _row_tokens(row: dict) -> int:
    """Token estimate.  Uses the dataset's own count when it has one, else the
    longest string field (column names differ per source: text / TEXT / content /
    processed_title)."""
    for key in ("token_count", "token_length"):
        if isinstance(row.get(key), int):
            return row[key]
    return C.est_tokens(_longest_str(row))


def _longest_str(row: dict) -> int:
    return max((len(v) for v in row.values() if isinstance(v, str)), default=0)


def download_source(
    name: str,
    limit: int | None = None,
    max_tokens: float | None = None,
    force: bool = False,
) -> dict:
    spec = C.SOURCES[name]
    outdir = C.RAW_DIR / name

    if C.is_complete(outdir) and not force:
        stats = C.read_marker(outdir)
        print(
            f"[{name}] already complete: {stats.get('docs', 0):,} docs, "
            f"~{C.human(stats.get('est_tokens', 0))} tokens — skipping (--force to redo)"
        )
        return stats
    if force:
        C.clear_dir(outdir)

    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS.get(name)
    keep = spec["keep"]

    print(f"[{name}] {spec['repo']} config={spec['config']} — {spec['note']}")
    print(f"[{name}] limit={limit} max_tokens={max_tokens and C.human(max_tokens)}")

    writer = C.ShardWriter(outdir, name, gzip_out=True)
    tokens = 0
    docs = 0
    t0 = time.time()
    try:
        for row in C.progress(C.hf_stream(name, limit=limit), desc=f"dl {name}"):
            rec = {k: row.get(k) for k in keep} if keep else dict(row)
            # stevez80 stores repr(bytes) in its string columns
            if name == "scifi":
                rec = {k: C.maybe_literal(v) for k, v in rec.items()}
            if _longest_str(rec) == 0:
                continue
            writer.write(rec)
            docs += 1
            tokens += _row_tokens(rec)
            if max_tokens is not None and tokens >= max_tokens:
                print(f"\n[{name}] hit token budget at {docs:,} docs")
                break
    except KeyboardInterrupt:
        writer.close()
        print(f"\n[{name}] interrupted after {docs:,} docs — re-run to resume", file=sys.stderr)
        raise

    stats = writer.close()
    stats.update(
        source=name, repo=spec["repo"], config=spec["config"],
        est_tokens=tokens, limit=limit, seconds=round(time.time() - t0, 1),
    )
    C.write_marker(outdir, stats)
    print(
        f"[{name}] done: {stats['docs']:,} docs, ~{C.human(stats['est_tokens'])} est tokens, "
        f"{len(stats['files'])} shard(s) in {stats['seconds']}s"
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--sources", default="all",
        help="comma-separated source names, 'all', or 'haiku' for the three haiku sets",
    )
    ap.add_argument("--limit", type=int, default=None, help="max docs per source (smoke testing)")
    ap.add_argument(
        "--max-tokens", type=float, default=None,
        help="per-source token budget (default: see DEFAULT_MAX_TOKENS)",
    )
    ap.add_argument("--force", action="store_true", help="re-download even if already complete")
    ap.add_argument("--list", action="store_true", help="list sources and exit")
    args = ap.parse_args(argv)

    if args.list:
        for name, spec in C.SOURCES.items():
            done = "complete" if C.is_complete(C.RAW_DIR / name) else "-"
            print(f"{name:16s} {spec['repo']:40s} {spec['config'] or '':14s} [{done}]")
        return 0

    if args.sources == "all":
        names = ALL_SOURCES
    elif args.sources == "haiku":
        names = HAIKU_SOURCES
    else:
        names = [s.strip() for s in args.sources.split(",") if s.strip()]
        names = HAIKU_SOURCES if names == ["haiku"] else names

    unknown = [n for n in names if n not in C.SOURCES]
    if unknown:
        ap.error(f"unknown source(s): {unknown}; known: {ALL_SOURCES}")

    failures: list[tuple[str, str]] = []
    for name in names:
        try:
            download_source(name, limit=args.limit, max_tokens=args.max_tokens, force=args.force)
        except KeyboardInterrupt:
            return 130
        except Exception as exc:  # keep going; report at the end
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)

    if failures:
        print("\nfailed sources:", file=sys.stderr)
        for name, err in failures:
            print(f"  {name}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
