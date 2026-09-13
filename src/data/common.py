"""Shared helpers for the data pipeline.

Every other module in ``src/data`` imports this FIRST, because importing it sets
``HF_HOME`` so that the HuggingFace cache lands under ``data/raw/hf_cache``
instead of ``~/.cache/huggingface``.  ``datasets`` is imported lazily inside
functions for the same reason.

Conventions used across the pipeline
------------------------------------
* Raw data:    ``data/raw/<source>/<source>_####.jsonl.gz``  (gzip: these are big)
* Clean data:  ``data/clean/<domain>/<domain>_####.jsonl``    (plain: the tokenizer
  stage streams them and the manifest points at them)
* Every jsonl line is ``{"text": str, "meta": {...}}`` for clean data.
* Shards are rolled at ~200 MB (uncompressed bytes written).
* A finished output directory gets a ``_complete.json`` marker so re-running a
  stage is a no-op unless ``--force`` is passed.  Partial shards are written as
  ``*.part`` and only renamed when complete, so a crash never leaves a
  half-written shard that a later stage would silently consume.
* Token estimates are ``len(text) / 4`` everywhere (chars-per-token for English
  BPE at our vocab size).  Real counts come from the tokenizer stage.
"""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
HF_CACHE = RAW_DIR / "hf_cache"

# Must happen before `datasets` / `huggingface_hub` are imported anywhere.
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

SEED = 1337
SHARD_MAX_BYTES = 200 * 1024 * 1024
CHARS_PER_TOKEN = 4


# --------------------------------------------------------------------------- #
# source registry
# --------------------------------------------------------------------------- #

# `keep` = fields copied into the raw shard (None = keep everything).
SOURCES: dict[str, dict[str, Any]] = {
    "fineweb_edu": {
        "repo": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "keep": ("text", "id", "url", "score", "token_count"),
        "stream": True,
        "note": "~10B GPT-2 tokens; feeds both `general` and (disjointly) `psych`.",
    },
    "cosmopedia": {
        "repo": "HuggingFaceTB/smollm-corpus",
        "config": "cosmopedia-v2",
        "split": "train",
        "keep": ("text", "prompt", "audience", "format", "seed_data", "token_length"),
        "stream": True,
        "note": "~28M synthetic textbook docs; we only need a topic-tilted slice.",
    },
    "gutenberg": {
        "repo": "sedthh/gutenberg_english",
        "config": None,
        "split": "train",
        "keep": None,
        "stream": True,
        "note": "chosen over common-pile/project_gutenberg: carries author+subject metadata.",
    },
    "scifi": {
        "repo": "stevez80/Sci-Fi-Books-gutenberg",
        "config": None,
        "split": "train",
        "keep": None,
        "stream": True,
        "note": "~482MB public-domain sci-fi; `text` is a Python bytes-repr string.",
    },
    "haiku_statworx": {
        "repo": "statworx/haiku",
        "config": None,
        "split": "train",
        "keep": ("text", "source", "text_punc"),
        "stream": False,
        "note": "lines joined by ' / '.",
    },
    "haiku_dugward": {
        "repo": "dugward/english_haiku",
        "config": None,
        "split": "train",
        "keep": ("content",),
        "stream": False,
        "note": "lines joined by ' \\\\ '; some rows are unsplittable and get dropped.",
    },
    "haiku_reddit": {
        "repo": "huanggab/reddit_haiku",
        "config": None,
        "split": "train",
        "keep": ("processed_title", "ups", "id"),
        "stream": False,
        "note": "lines joined by '/'.",
    },
}


# --------------------------------------------------------------------------- #
# text normalisation
# --------------------------------------------------------------------------- #

_ZERO_WIDTH = dict.fromkeys(
    map(ord, "​‌‍⁠﻿­"), None
)
_MULTI_NL = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+$", re.M)


def norm_ws(text: str) -> str:
    """Minimal whitespace normalisation.  No lowercasing, no unicode mangling
    beyond NFC and zero-width removal, no collapsing of runs of spaces (that
    would destroy verse indentation)."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_ZERO_WIDTH)
    text = unicodedata.normalize("NFC", text)
    text = _TRAILING_WS.sub("", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


_LOWER_AFTER_BREAK = re.compile(r"\n\n[ \t]*(?=[a-z,;])")


def undouble_blank_lines(text: str) -> str:
    """Some Gutenberg dumps (notably sedthh/gutenberg_english) separate *every*
    hard-wrapped line with a blank line, so naive paragraph detection sees one
    paragraph per wrapped line.  Detect that and undo it."""
    breaks = text.count("\n\n")
    if breaks < 10:
        return text
    lowercase_continuations = len(_LOWER_AFTER_BREAK.findall(text))
    if lowercase_continuations / breaks < 0.25:
        return text
    text = _MULTI_NL.sub("\x00", text)          # protect real paragraph breaks
    text = text.replace("\n\n", "\n")           # blank line was a soft wrap
    return text.replace("\x00", "\n\n")


_SENTENCE_END = re.compile(r"[.!?:;\"'”’)\]]\s*$")


def unwrap_hard_wraps(text: str, min_len: int = 55) -> str:
    """Join lines that were hard-wrapped mid-sentence (classic Gutenberg 70-col
    wrapping).  Only joins when the line is long enough to look like a wrap, has
    no terminal punctuation, and the next line continues in lowercase — so verse,
    headings and dialogue survive."""
    out: list[str] = []
    for line in text.split("\n"):
        if (
            out
            and out[-1].strip()
            and line.strip()
            and len(out[-1].rstrip()) >= min_len
            and not _SENTENCE_END.search(out[-1])
            and not line.lstrip()[:1].isupper()
            # same indentation => same block; a different indent means a new
            # quote/verse/heading block, which we leave alone
            and abs(_indent(out[-1]) - _indent(line)) <= 1
        ):
            out[-1] = out[-1].rstrip() + " " + line.lstrip()
        else:
            out.append(line)
    return "\n".join(out)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def est_tokens(chars: int) -> int:
    return chars // CHARS_PER_TOKEN


def doc_hash(text: str, n: int = 1000) -> str:
    """Dedup key: first ``n`` chars, whitespace-squashed and casefolded."""
    head = re.sub(r"\s+", " ", text[:n]).strip().casefold()
    return hashlib.blake2b(head.encode("utf-8"), digest_size=16).hexdigest()


def maybe_literal(value: Any) -> Any:
    """stevez80/Sci-Fi-Books-gutenberg stores ``repr(bytes)`` / ``repr(str)`` in
    its string columns (``"b'\\xef\\xbb\\xbfThe Project Gutenberg...'"``).
    Undo that when we see it."""
    if not isinstance(value, str) or len(value) < 3:
        return value
    if value[:2] in ("b'", 'b"') or value[0] in "'\"":
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return value
        if isinstance(parsed, bytes):
            return parsed.decode("utf-8", "replace")
        if isinstance(parsed, str):
            return parsed
    return value


# --------------------------------------------------------------------------- #
# Project Gutenberg boilerplate
# --------------------------------------------------------------------------- #

_PG_START = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS)\s+PROJECT GUTENBERG\s+(?:EBOOK|ETEXT)\b[^\n]*?\*\*\*",
    re.I,
)
_PG_START_LOOSE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS)\s+PROJECT GUTENBERG\s+(?:EBOOK|ETEXT)\b[^\n]*", re.I
)
_PG_END = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS)\s+PROJECT GUTENBERG\s+(?:EBOOK|ETEXT)\b", re.I
)
_PG_END_LOOSE = re.compile(
    r"^\s*End of (?:the |this )?Project Gutenberg(?:'s)?\b", re.I | re.M
)
_SMALL_PRINT_END = re.compile(r"\*\s*END\s*\*\s*THE SMALL PRINT[^\n]*\n", re.I)
_PRODUCED_BY = re.compile(
    r"^[ \t]*(?:E-?text |Etext )?(?:Produced|Prepared|Transcribed|Scanned|Updated editions)"
    r"\b[^\n]*(?:\n(?!\s*\n)[^\n]*)*",
    re.I | re.M,
)
# NB: Gutenberg texts use curly apostrophes as often as straight ones.
_APOS = r"['’`´]?s?"
_NOTE_WORD = r"(?:Note|Annotation|Convention|Comment|Remark)s?"
_TRANSCRIBER_BRACKET = re.compile(
    rf"\[\s*(?:Transcriber{_APOS}|Editor{_APOS}|Publisher{_APOS})\s+{_NOTE_WORD}\b.*?\]",
    re.I | re.S,
)
# The `[|+]?` prefix catches the ASCII-art boxed notes common in the sci-fi dump:
#   +------------------------------------+
#   | Transcriber's note:                 |
_TRANSCRIBER_BLOCK = re.compile(
    rf"^[ \t]*[|+]?[ \t]*(?:Transcriber{_APOS}|Editor{_APOS})[\s&]+(?:{_NOTE_WORD}|Errata)\b.*?(?:\n\s*\n)",
    re.I | re.S | re.M,
)
# Leftover box borders / rules.
_ASCII_BOX_LINE = re.compile(r"^[ \t]*[|+][-=+| \t]{8,}[|+]?[ \t]*$", re.M)
# Catch-all: a *short* line mentioning Project Gutenberg is always boilerplate
# ("Welcome to Project Gutenberg's presentation of...", "PROJECT GUTENBERG
# EDITOR'S BOOKMARKS", "End of Project Gutenberg's X, by Y").  Real prose in
# these books never mentions it.
_PG_MENTION_LINE = re.compile(
    r"^.{0,200}?(?:project\s+gutenberg|gutenberg\.org|gutenberg[-\s]?tm|pglaf)\b.{0,200}$",
    re.I | re.M,
)
_PG_TRAILER = re.compile(
    r"(?:End of (?:the )?Project Gutenberg|This file should be named|"
    r"Updated editions will be named|Creating the works from|"
    r"Section \d\.\s+(?:General )?Information about|"
    r"START: FULL LICENSE|www\.gutenberg\.org)",
    re.I,
)
_ILLUSTRATION = re.compile(r"\[\s*Illustrations?\b[^\]]{0,400}\]", re.I | re.S)
_FOOTNOTE_MARK = re.compile(r"\[\s*(?:Footnote|FN)[^\]]{0,600}\]", re.I | re.S)


def strip_gutenberg(text: str) -> str:
    """Remove PG header/footer/licence, 'Produced by' credits, transcriber notes
    and [Illustration] markers.  Safe to call on text that has none of them."""
    if not text:
        return ""
    text = text.lstrip("﻿")

    m = _PG_START.search(text) or _PG_START_LOOSE.search(text)
    if m:
        text = text[m.end():]
    else:
        m = _SMALL_PRINT_END.search(text)
        if m:
            text = text[m.end():]

    # Take the EARLIEST end marker.  Some dumps carry both forms — a plain
    # "End of Project Gutenberg's X, by Y" line and, further down, the
    # "*** END OF ... ***" banner — and cutting at the later one leaves the
    # earlier line sitting in the text.  The loose form is only trusted in the
    # back half of the document so an incidental mention can't truncate a book.
    half = len(text) // 2
    ends = [m.start() for m in (_PG_END.search(text), _PG_END_LOOSE.search(text, half)) if m]
    if ends:
        text = text[: min(ends)]
    else:
        # No explicit end marker: chop at the first licence-ish trailer that
        # appears in the last 15% of the document.
        m = _PG_TRAILER.search(text, int(len(text) * 0.85))
        if m:
            text = text[: m.start()]

    text = _TRANSCRIBER_BRACKET.sub("", text)
    text = _TRANSCRIBER_BLOCK.sub("", text)
    text = _ILLUSTRATION.sub("", text)
    text = _FOOTNOTE_MARK.sub("", text)
    text = _PRODUCED_BY.sub("", text, count=2)
    text = _PG_MENTION_LINE.sub("", text)
    text = _ASCII_BOX_LINE.sub("", text)
    return text


def prose_clean(text: str) -> str:
    """Full clean for Gutenberg-style prose books."""
    text = strip_gutenberg(text)
    text = norm_ws(text)
    text = undouble_blank_lines(text)
    text = unwrap_hard_wraps(text)
    return norm_ws(text)


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #

def chunk_text(text: str, max_chars: int = 50_000, min_chars: int = 2_000) -> list[str]:
    """Split a book into ~``max_chars`` pieces on paragraph boundaries.
    Deterministic; no randomness."""
    if len(text) <= max_chars:
        return [text] if len(text) >= min_chars else ([text] if text else [])
    paras = text.split("\n\n")
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for para in paras:
        plen = len(para) + 2
        if size + plen > max_chars and size >= min_chars:
            chunks.append("\n\n".join(buf))
            buf, size = [], 0
        if plen > max_chars:  # a single monster paragraph: hard-split it
            if buf:
                chunks.append("\n\n".join(buf))
                buf, size = [], 0
            for i in range(0, len(para), max_chars):
                chunks.append(para[i : i + max_chars])
            continue
        buf.append(para)
        size += plen
    if buf:
        tail = "\n\n".join(buf)
        if len(tail) < min_chars and chunks:
            chunks[-1] = chunks[-1] + "\n\n" + tail
        elif tail:
            chunks.append(tail)
    return chunks


# --------------------------------------------------------------------------- #
# shard IO
# --------------------------------------------------------------------------- #

class ShardWriter:
    """Writes jsonl shards, rolling at ``max_bytes``.  Shards are built as
    ``*.part`` and renamed on completion."""

    def __init__(
        self,
        outdir: str | Path,
        prefix: str,
        max_bytes: int = SHARD_MAX_BYTES,
        gzip_out: bool = False,
    ):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.max_bytes = max_bytes
        self.gzip_out = gzip_out
        self.ext = ".jsonl.gz" if gzip_out else ".jsonl"
        self.index = 0
        self.fh = None
        self.cur_bytes = 0
        self.docs = 0
        self.chars = 0
        self.files: list[Path] = []

    def _path(self, idx: int) -> Path:
        return self.outdir / f"{self.prefix}_{idx:04d}{self.ext}"

    def _open(self) -> None:
        while self._path(self.index).exists():
            self.index += 1
        self.part = self._path(self.index).with_suffix(self._path(self.index).suffix + ".part")
        self.fh = (
            gzip.open(self.part, "wt", encoding="utf-8", compresslevel=5)
            if self.gzip_out
            else open(self.part, "w", encoding="utf-8")
        )
        self.cur_bytes = 0

    def _close_shard(self) -> None:
        if self.fh is None:
            return
        self.fh.close()
        final = self._path(self.index)
        self.part.rename(final)
        self.files.append(final)
        self.fh = None
        self.index += 1

    def write(self, obj: dict) -> None:
        if self.fh is None:
            self._open()
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        self.fh.write(line)
        n = len(line.encode("utf-8"))
        self.cur_bytes += n
        self.docs += 1
        self.chars += len(obj.get("text", ""))
        if self.cur_bytes >= self.max_bytes:
            self._close_shard()

    def close(self) -> dict:
        self._close_shard()
        return {
            "docs": self.docs,
            "chars": self.chars,
            "est_tokens": est_tokens(self.chars),
            "files": [str(p.relative_to(REPO_ROOT)) for p in self.files],
        }


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def shard_paths(directory: str | Path) -> list[Path]:
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(
        p for p in d.iterdir()
        if p.name.endswith((".jsonl", ".jsonl.gz")) and not p.name.startswith("_")
    )


def iter_jsonl_dir(directory: str | Path, skip: int = 0, limit: int | None = None) -> Iterator[dict]:
    n = 0
    for path in shard_paths(directory):
        for rec in read_jsonl(path):
            n += 1
            if n <= skip:
                continue
            yield rec
            if limit is not None and n - skip >= limit:
                return


# --------------------------------------------------------------------------- #
# completion markers
# --------------------------------------------------------------------------- #

def marker_path(directory: str | Path) -> Path:
    return Path(directory) / "_complete.json"


def is_complete(directory: str | Path) -> bool:
    return marker_path(directory).exists()


def read_marker(directory: str | Path) -> dict:
    p = marker_path(directory)
    return json.loads(p.read_text()) if p.exists() else {}


def write_marker(directory: str | Path, stats: dict) -> None:
    Path(directory).mkdir(parents=True, exist_ok=True)
    marker_path(directory).write_text(json.dumps(stats, indent=2) + "\n")


def clear_dir(directory: str | Path) -> None:
    """Remove shards + marker so a stage can be re-run cleanly (--force)."""
    d = Path(directory)
    if not d.is_dir():
        return
    for p in d.iterdir():
        if p.name.endswith((".jsonl", ".jsonl.gz", ".part", ".json")):
            p.unlink()


# --------------------------------------------------------------------------- #
# HF access
# --------------------------------------------------------------------------- #

def hf_stream(name: str, limit: int | None = None, skip: int = 0) -> Iterator[dict]:
    """Stream a registered source straight from the Hub."""
    from datasets import load_dataset

    spec = SOURCES[name]
    ds = load_dataset(
        spec["repo"], spec["config"], split=spec["split"], streaming=True
    )
    if skip:
        ds = ds.skip(skip)
    n = 0
    for row in ds:
        yield row
        n += 1
        if limit is not None and n >= limit:
            return


def iter_source(name: str, limit: int | None = None, skip: int = 0) -> Iterator[dict]:
    """Yield rows for a source, preferring the local raw shards written by
    ``download.py`` and falling back to HF streaming."""
    local = RAW_DIR / name
    if shard_paths(local):
        yield from iter_jsonl_dir(local, skip=skip, limit=limit)
    else:
        print(f"[{name}] no local raw shards, streaming from the Hub", file=sys.stderr)
        yield from hf_stream(name, limit=limit, skip=skip)


def source_doc_count(name: str) -> int:
    m = read_marker(RAW_DIR / name)
    return int(m.get("docs", 0))


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #

_ASCII_LETTERS = re.compile(r"[A-Za-z]")
_COMMON_EN = {
    "the", "and", "of", "to", "a", "in", "is", "it", "you", "that", "he", "was",
    "for", "on", "are", "as", "with", "his", "they", "i", "at", "be", "this",
    "have", "from", "or", "one", "had", "by", "but", "not", "what", "all", "we",
}


def looks_english(text: str, min_ascii_ratio: float = 0.9, require_stopword: bool = True) -> bool:
    if not text:
        return False
    letters = _ASCII_LETTERS.findall(text)
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False
    if len(letters) / len(alpha) < min_ascii_ratio:
        return False
    if require_stopword:
        words = set(re.findall(r"[a-z']+", text.lower()))
        return bool(words & _COMMON_EN)
    return True


def human(n: float) -> str:
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{n:.0f}"
        n /= 1000
    return f"{n:.1f}P"


def progress(iterable: Iterable, desc: str, total: int | None = None):
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc, total=total, unit="doc", smoothing=0.05)
    except ImportError:
        return iterable
