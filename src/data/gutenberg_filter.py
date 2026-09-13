"""Stage 2 — carve the Gutenberg domains out of the raw dumps.

Produces:
  data/clean/philosophy/*.jsonl
  data/clean/comedy/*.jsonl
  data/clean/scifi/*.jsonl        (from stevez80/Sci-Fi-Books-gutenberg)

Each line is ``{"text": ..., "meta": {title, author, source, ...}}`` — one doc per
book, or per ~50k-char chunk for books longer than that.

Source choice
-------------
We use **sedthh/gutenberg_english**, not common-pile/project_gutenberg.  Both were
inspected with 5 streamed rows:

  common-pile/project_gutenberg  -> {id, text, source, added, metadata:{license,
      language, url, title}}.  Only a *title*; no author, no subject headings.
      Selecting philosophy/comedy from it would mean title-string guessing.

  sedthh/gutenberg_english       -> {TEXT, SOURCE, METADATA} where METADATA is a
      JSON string carrying {text_id, title, issued, authors, subjects, locc,
      bookshelves}.  ``subjects`` holds the real LoC subject headings
      ("Philosophy; Ethics; Stoics") and ``locc`` the LoC class letters (B =
      Philosophy/Psychology/Religion, PN6231 = humour).  That is exactly the
      metadata the allowlists below need, so sedthh wins.

Its quirks, handled in common.py: CRLF line endings, every hard-wrapped line
separated by a blank line (``undouble_blank_lines``), and headers that are
sometimes already stripped and sometimes not (``strip_gutenberg`` is a no-op when
the markers are absent).

Usage
-----
    python -m src.data.gutenberg_filter --domains philosophy,comedy,scifi
    python -m src.data.gutenberg_filter --limit 2000 --force
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from src.data import common as C

# --------------------------------------------------------------------------- #
# allowlists — edit freely, they are plain data
# --------------------------------------------------------------------------- #

# Matched (case-insensitively) as substrings against the `authors` field, which is
# formatted "Surname, Forename, dates".  Keep them surname-first and specific
# enough not to catch unrelated people.
PHILOSOPHY_AUTHORS = [
    "plato", "aristotle", "xenophon", "socrates",
    "aurelius, marcus", "marcus aurelius", "antoninus, marcus aurelius",
    "epictetus", "seneca, lucius annaeus", "cicero", "plutarch", "lucretius",
    "epicurus", "diogenes laertius", "boethius", "augustine",
    "aquinas", "erasmus", "montaigne", "descartes", "spinoza", "leibniz",
    "locke, john", "berkeley, george", "hume, david", "kant, immanuel",
    "rousseau, jean-jacques", "voltaire", "smith, adam", "mill, john stuart",
    "bentham, jeremy", "hegel", "schopenhauer", "kierkegaard", "nietzsche",
    "emerson, ralph waldo", "thoreau, henry david", "james, william",
    "russell, bertrand", "santayana, george", "dewey, john", "bergson, henri",
    "spencer, herbert", "carlyle, thomas", "pascal, blaise", "hobbes, thomas",
    "machiavelli", "confucius", "lao", "sun tzu", "bacon, francis",
    "schiller, f. c. s.", "royce, josiah", "sidgwick, henry", "green, t. h.",
]

# Substring match against `subjects` + `bookshelves`.
PHILOSOPHY_SUBJECTS = [
    "philosophy", "philosophers", "ethics", "stoic", "stoicism", "metaphysics",
    "epistemology", "logic", "ontology", "aesthetics", "moral", "virtue",
    "free will", "consciousness", "conduct of life", "wisdom",
    "political science -- philosophy", "knowledge, theory of",
    "life -- philosophy", "meaning (philosophy)", "mind and body",
    "psychology", "self-actualization",
]

# LoC classes: B (philosophy), BC (logic), BD (speculative), BF (psychology),
# BJ (ethics).  Excluded: BL/BM/BP/BQ/BR/BS/BT/BV/BX (religion/scripture).
PHILOSOPHY_LOCC_PREFIXES = ["BC", "BD", "BF", "BJ", "BH"]
PHILOSOPHY_LOCC_EXACT = ["B"]

COMEDY_AUTHORS = [
    "twain, mark", "clemens, samuel", "jerome, jerome k.", "wodehouse, p. g.",
    "munro, h. h.", "saki", "leacock, stephen", "bierce, ambrose",
    "ade, george", "nash, ogden", "lear, edward", "carroll, lewis",
    "jacobs, w. w.", "milne, a. a.", "benchley, robert", "marquis, don",
    "field, eugene", "nye, bill", "ward, artemus", "browne, charles farrar",
    "shaw, henry wheeler", "billings, josh", "harte, bret",
    "grossmith, george", "grossmith, weedon", "gilbert, w. s.",
    "thurber, james", "runyon, damon", "butler, ellis parker",
    "hood, thomas", "lucas, e. v.", "beerbohm, max", "chesterton, g. k.",
    "irving, washington", "rabelais", "sterne, laurence", "smollett, tobias",
]

COMEDY_SUBJECTS = [
    "humor", "humour", "wit and humor", "satire", "comedy", "parody",
    "burlesque", "limericks", "nonsense literature", "comic",
    "humorous stories", "humorous poetry", "farce",
]

# PN6231 = wit & humour collections, PN6110-6120 = humorous verse/anthologies,
# PR/PS are literature classes we only reach via subject/author.
COMEDY_LOCC_PREFIXES = ["PN6"]

# Authors on the list who are prolific but mostly off-domain; require a subject
# hit as well for these.  (Chesterton wrote theology, Irving wrote history.)
AUTHOR_NEEDS_SUBJECT = {"chesterton, g. k.", "irving, washington", "cicero", "plutarch"}

# Subject headings that veto a book for *either* domain.
VETO_SUBJECTS = [
    "bible", "sermons", "hymns", "prayers", "theology", "catechisms",
    "readers", "dictionaries", "encyclopedias", "bibliography",
    "periodicals", "indexes", "genealogy", "cookbooks", "almanacs",
    # Pre-modern English is bad pretraining data for a modern-English model.
    # ("Christian ethics -- Poetry; Love poetry, English (Middle)" was pulling
    # Gower's Confessio Amantis into philosophy on the word "ethics".)
    # NB: these must end on a word character — _subject_re appends \b, which
    # would never match after a closing paren.
    "english (middle", "middle english", "old english", "anglo-saxon",
]

# Philosophy is non-fiction.  Without this, every novel catalogued as
# "Psychological fiction" lands in the philosophy pile (and note that the
# subject match MUST be word-bounded, or "logic" matches "psychoLOGICal").
FICTION_SUBJECTS = [
    "fiction", "stories", "novel", "juvenile", "romances", "tales",
    "legends", "drama", "plays", "fairy", "adventure",
]

# Wodehouse: only pre-1930 texts are reliably public domain in the US dumps;
# the dataset's `issued` field is the PG release date, so gate on copyright-safe
# author-year instead, using the title year when present, else keep (the dataset
# itself is PD-only, this is belt-and-braces).
WODEHOUSE_MAX_YEAR = 1929

MIN_DOC_CHARS = 1_000
MAX_CHUNK_CHARS = 50_000


# --------------------------------------------------------------------------- #

def parse_meta(row: dict) -> dict:
    """sedthh rows: {TEXT, SOURCE, METADATA(json string)}."""
    raw = row.get("METADATA") or row.get("metadata") or "{}"
    if isinstance(raw, str):
        try:
            meta = json.loads(raw)
        except json.JSONDecodeError:
            meta = {}
    else:
        meta = dict(raw)
    return meta


def _subject_re(terms: list[str]) -> re.Pattern:
    """Word-bounded alternation.  Substring matching is a trap here: 'logic'
    inside 'psychological', 'moral' inside 'morale', 'comic' inside 'comics'."""
    parts = [
        re.escape(t).replace(r"\ ", r"\s+").replace(r"\-", r"[-\s]")
        for t in sorted(set(terms), key=len, reverse=True)
    ]
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.I)


PHILOSOPHY_SUBJECT_RE = _subject_re(PHILOSOPHY_SUBJECTS)
COMEDY_SUBJECT_RE = _subject_re(COMEDY_SUBJECTS)
VETO_SUBJECT_RE = _subject_re(VETO_SUBJECTS)
FICTION_SUBJECT_RE = _subject_re(FICTION_SUBJECTS)


def _hay(meta: dict) -> tuple[str, str, str]:
    authors = (meta.get("authors") or "").lower()
    subjects = " ; ".join(
        str(meta.get(k) or "") for k in ("subjects", "bookshelves")
    ).lower()
    locc = (meta.get("locc") or "").upper()
    return authors, subjects, locc


def _locc_hit(locc: str, prefixes: list[str], exact: list[str] | None = None) -> bool:
    for token in re.split(r"[;,\s]+", locc):
        if not token:
            continue
        if exact and token in exact:
            return True
        if any(token.startswith(p) for p in prefixes):
            return True
    return False


def classify(meta: dict) -> str | None:
    """Return 'philosophy', 'comedy' or None.  Comedy wins ties (a humorous essay
    on ethics is more useful to us as comedy than as philosophy)."""
    if (meta.get("language") or "en") not in ("en", "english", None):
        return None
    authors, subjects, locc = _hay(meta)

    if VETO_SUBJECT_RE.search(subjects):
        return None

    com_author = next((a for a in COMEDY_AUTHORS if a in authors), None)
    com_subject = bool(COMEDY_SUBJECT_RE.search(subjects))
    com_locc = _locc_hit(locc, COMEDY_LOCC_PREFIXES)
    if com_author and "wodehouse" in com_author:
        year = _issued_year(meta)
        if year and year > WODEHOUSE_MAX_YEAR:
            com_author = None
    if com_author in AUTHOR_NEEDS_SUBJECT and not com_subject:
        com_author = None
    if com_author or com_subject or com_locc:
        return "comedy"

    # Philosophy: non-fiction only.  A fiction subject heading vetoes the book
    # unless the Library of Congress class is a B-class (real philosophy), which
    # outranks a stray "Tales" heading.
    phi_locc = _locc_hit(locc, PHILOSOPHY_LOCC_PREFIXES, PHILOSOPHY_LOCC_EXACT)
    if FICTION_SUBJECT_RE.search(subjects) and not phi_locc:
        return None
    phi_author = next((a for a in PHILOSOPHY_AUTHORS if a in authors), None)
    phi_subject = bool(PHILOSOPHY_SUBJECT_RE.search(subjects))
    if phi_author in AUTHOR_NEEDS_SUBJECT and not (phi_subject or phi_locc):
        phi_author = None
    if phi_author or phi_subject or phi_locc:
        return "philosophy"
    return None


_YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")


def _issued_year(meta: dict) -> int | None:
    m = _YEAR.search(str(meta.get("issued") or ""))
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #

def run_gutenberg(
    domains: list[str], limit: int | None, max_chunk: int, force: bool
) -> dict[str, dict]:
    """One pass over the raw Gutenberg dump, splitting into philosophy/comedy."""
    wanted = [d for d in domains if d in ("philosophy", "comedy")]
    if not wanted:
        return {}

    outdirs = {d: C.CLEAN_DIR / d for d in wanted}
    skip = [d for d in wanted if C.is_complete(outdirs[d]) and not force]
    if skip and len(skip) == len(wanted):
        print(f"[gutenberg] {', '.join(skip)} already complete — skipping (--force to redo)")
        return {d: C.read_marker(outdirs[d]) for d in wanted}
    for d in wanted:
        C.clear_dir(outdirs[d])

    writers = {d: C.ShardWriter(outdirs[d], d) for d in wanted}
    seen: set[str] = set()
    counts = {d: {"books": 0, "chunks": 0} for d in wanted}
    scanned = 0

    for row in C.progress(C.iter_source("gutenberg", limit=limit), desc="gutenberg"):
        scanned += 1
        meta = parse_meta(row)
        domain = classify(meta)
        if domain not in wanted:
            continue
        text = C.prose_clean(row.get("TEXT") or row.get("text") or "")
        if len(text) < MIN_DOC_CHARS:
            continue
        key = C.doc_hash(text)
        if key in seen:
            continue
        seen.add(key)
        title = str(meta.get("title") or "").replace("\n", " ").strip()
        author = str(meta.get("authors") or "").strip()
        chunks = C.chunk_text(text, max_chars=max_chunk)
        counts[domain]["books"] += 1
        for i, chunk in enumerate(chunks):
            writers[domain].write({
                "text": chunk,
                "meta": {
                    "title": title,
                    "author": author,
                    "source": "sedthh/gutenberg_english",
                    "gutenberg_id": meta.get("text_id"),
                    "subjects": meta.get("subjects"),
                    "locc": meta.get("locc"),
                    "chunk": i,
                    "n_chunks": len(chunks),
                },
            })
            counts[domain]["chunks"] += 1

    out = {}
    for d, w in writers.items():
        stats = w.close()
        stats.update(domain=d, books=counts[d]["books"], scanned=scanned,
                     source="sedthh/gutenberg_english")
        C.write_marker(outdirs[d], stats)
        out[d] = stats
        print(f"[{d}] {stats['books']:,} books -> {stats['docs']:,} docs, "
              f"~{C.human(stats['est_tokens'])} est tokens (scanned {scanned:,})")
    return out


def run_scifi(limit: int | None, max_chunk: int, force: bool) -> dict:
    """stevez80/Sci-Fi-Books-gutenberg: {id, title, author, text}, where every
    string column is a Python *repr* (``"b'\\xef\\xbb\\xbfThe Project ...'"`` /
    ``"'Shelley, Mary Wollstonecraft'"``) — see common.maybe_literal."""
    outdir = C.CLEAN_DIR / "scifi"
    if C.is_complete(outdir) and not force:
        print("[scifi] already complete — skipping (--force to redo)")
        return C.read_marker(outdir)
    C.clear_dir(outdir)

    writer = C.ShardWriter(outdir, "scifi")
    seen: set[str] = set()
    books = 0
    for row in C.progress(C.iter_source("scifi", limit=limit), desc="scifi"):
        text = C.prose_clean(C.maybe_literal(row.get("text") or ""))
        if len(text) < MIN_DOC_CHARS or not C.looks_english(text[:5000]):
            continue
        key = C.doc_hash(text)
        if key in seen:
            continue
        seen.add(key)
        title = str(C.maybe_literal(row.get("title") or "")).strip()
        author = str(C.maybe_literal(row.get("author") or "")).strip()
        chunks = C.chunk_text(text, max_chars=max_chunk)
        books += 1
        for i, chunk in enumerate(chunks):
            writer.write({
                "text": chunk,
                "meta": {
                    "title": title, "author": author,
                    "source": "stevez80/Sci-Fi-Books-gutenberg",
                    "gutenberg_id": row.get("id"),
                    "chunk": i, "n_chunks": len(chunks),
                },
            })
    stats = writer.close()
    stats.update(domain="scifi", books=books, source="stevez80/Sci-Fi-Books-gutenberg")
    C.write_marker(outdir, stats)
    print(f"[scifi] {books:,} books -> {stats['docs']:,} docs, "
          f"~{C.human(stats['est_tokens'])} est tokens")
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", default="philosophy,comedy,scifi")
    ap.add_argument("--limit", type=int, default=None, help="max raw books to scan per source")
    ap.add_argument("--max-chunk-chars", type=int, default=MAX_CHUNK_CHARS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    unknown = set(domains) - {"philosophy", "comedy", "scifi"}
    if unknown:
        ap.error(f"unknown domain(s): {sorted(unknown)}")

    run_gutenberg(domains, args.limit, args.max_chunk_chars, args.force)
    if "scifi" in domains:
        run_scifi(args.limit, args.max_chunk_chars, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
