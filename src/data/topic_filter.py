"""Stage 4 — topic filtering.

Three subcommands:

``train``            Bootstrap a psychology/self-help classifier with no labelled
                     data: keyword-score a scan of FineWeb-Edu, take the top
                     hits as positives and keyword-free docs as negatives, then
                     train fastText supervised (falling back to TF-IDF +
                     LogisticRegression if fastText is unavailable).
                     -> data/clean/psych_classifier/

``apply``            Stream FineWeb-Edu *past the offset the general slice
                     consumed* (so the two slices are disjoint), keep docs the
                     classifier scores above ``--threshold`` until
                     ``--target-tokens``.  -> data/clean/psych/

``tilt-cosmopedia``  Keyword-score Cosmopedia-v2 for philosophy/logic/psych/ethics
                     and build the textbooks slice: topical docs first, general
                     cosmopedia docs as filler.  -> data/clean/textbooks/

Note on Cosmopedia metadata: the config exposes only
``{prompt, text, token_length, audience, format, seed_data}`` — there is no topic
or category column (``seed_data`` is just the corpus the seed extract came from,
e.g. "fineweb"/"auto_math_text"), so the tilt scores the *prompt* (which quotes
the seed extract) plus the head of the generated text.

Usage
-----
    python -m src.data.topic_filter train --scan-docs 200000
    python -m src.data.topic_filter apply --target-tokens 4.6e8
    python -m src.data.topic_filter tilt-cosmopedia --target-tokens 7.8e8
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import sys
from pathlib import Path

from src.config import load_config
from src.data import common as C

CLASSIFIER_DIR = C.CLEAN_DIR / "psych_classifier"
SLACK = 1.08

# --------------------------------------------------------------------------- #
# keyword seeds — module constants, easy to edit
# --------------------------------------------------------------------------- #

PSYCH_KEYWORDS = [
    # clinical / CBT register
    "cognitive behavioral", "cognitive behavioural", "cbt", "psychotherapy",
    "therapist", "counseling", "counselling", "talk therapy", "clinical psychology",
    "cognitive distortion", "automatic thoughts", "thought record", "reframing",
    "exposure therapy", "behavioral activation", "dialectical behavior",
    "acceptance and commitment", "psychoanalysis", "psychiatrist",
    # emotion / wellbeing register
    "anxiety", "depression", "stress management", "burnout", "self-esteem",
    "emotional regulation", "emotion regulation", "coping strategies",
    "coping mechanisms", "resilience", "mindfulness", "meditation",
    "mental health", "well-being", "wellbeing", "emotional intelligence",
    "self-compassion", "self-awareness", "trauma", "grief", "loneliness",
    "panic attack", "mood", "rumination", "triggers",
    # self-help / motivation register
    "self-help", "personal growth", "personal development", "habit formation",
    "goal setting", "motivation", "procrastination", "productivity habits",
    "growth mindset", "self-improvement", "boundaries", "assertiveness",
    "active listening", "communication skills", "conflict resolution",
    "relationship advice", "parenting", "empathy", "gratitude journal",
    # academic psychology
    "psychology", "psychological", "behavioral science", "cognitive science",
    "attachment theory", "developmental psychology", "social psychology",
    "personality traits", "big five", "cognitive bias", "neuroscience of",
    "operant conditioning", "classical conditioning", "maslow",
]

# Negatives for the seed set must not merely lack keywords — they should also be
# plausible hard negatives.  We take keyword-free docs at random, which is what
# `apply` will mostly see.
PSYCH_STRONG = {
    "cognitive behavioral", "cognitive behavioural", "psychotherapy", "therapist",
    "self-help", "mental health", "emotional regulation", "emotion regulation",
    "self-compassion", "cognitive distortion", "mindfulness", "self-esteem",
    "coping strategies", "coping mechanisms", "psychology",
}

COSMO_KEYWORDS = [
    # philosophy / ethics
    "philosophy", "philosopher", "philosophical", "ethics", "ethical",
    "moral", "morality", "virtue", "metaphysics", "epistemology", "ontology",
    "existentialism", "stoicism", "stoic", "utilitarian", "deontolog",
    "free will", "consciousness", "aesthetics", "phenomenology",
    "socrates", "plato", "aristotle", "kant", "nietzsche", "hume", "descartes",
    "aurelius", "epictetus", "seneca", "spinoza", "kierkegaard", "sartre",
    "political philosophy", "philosophy of mind", "philosophy of science",
    "meaning of life", "human nature", "the good life",
    # logic / reasoning.  NB: bare "logic" is NOT on this list — it matches
    # "logic systems"/"logic gates" in every electronics textbook.  Same reason
    # bare "behavior", "reasoning" and "debate" are out: too generic to carry a
    # topic signal on their own.
    "logical fallacy", "logical fallacies", "logical reasoning", "syllogism",
    "deductive reasoning", "inductive reasoning", "critical thinking",
    "argumentation", "propositional logic", "formal logic", "symbolic logic",
    "philosophical logic", "modal logic", "informal fallacy", "rhetoric",
    "valid argument", "sound argument", "premise and conclusion",
    # psychology (shared with the psych slice on purpose — textbooks tilt too)
    "psychology", "psychological", "cognitive science", "cognitive bias",
    "mental health", "emotional intelligence", "self-awareness",
    "mindfulness", "empathy", "personality psychology", "developmental psychology",
    "social psychology", "attachment theory", "behavioral science",
    "human behavior", "human behaviour", "emotional regulation",
]

MIN_CHARS = 400


def _compile(keywords: list[str]) -> re.Pattern:
    """Leading word boundary only — deliberately, so prefixes like "deontolog"
    match deontology/deontological and "psycholog" catches the family.  The
    trailing end is left open, so keep generic single words OFF these lists."""
    parts = []
    for kw in sorted(set(keywords), key=len, reverse=True):
        esc = re.escape(kw).replace(r"\ ", r"\s+").replace(r"\-", r"[-\s]")
        parts.append(rf"\b{esc}")
    return re.compile("|".join(parts), re.I)


PSYCH_RE = _compile(PSYCH_KEYWORDS)
PSYCH_STRONG_RE = _compile(sorted(PSYCH_STRONG))
COSMO_RE = _compile(COSMO_KEYWORDS)


def keyword_score(text: str, pattern: re.Pattern, head: int = 8000) -> tuple[int, int]:
    """(distinct keyword phrases matched, total matches) over the head of a doc."""
    hits = pattern.findall(text[:head])
    norm = {re.sub(r"\s+", " ", h.lower()) for h in hits}
    return len(norm), len(hits)


# --------------------------------------------------------------------------- #
# classifier
# --------------------------------------------------------------------------- #

def _ft_line(text: str, max_chars: int = 3000) -> str:
    """fastText wants one doc per line, no newlines.  Lowercasing here affects the
    classifier only — the corpus itself is never lowercased."""
    t = re.sub(r"\s+", " ", text[:max_chars]).strip().lower()
    return re.sub(r"([.,!?;:()\"'])", r" \1 ", t)


def _ft_psych_prob(model, line: str) -> float:
    """P(__label__psych) for one line.

    fastText 0.9.3's own ``model.predict()`` ends in
    ``np.array(probs, copy=False)``, which NumPy 2 rejects outright
    ("Unable to avoid copy while creating an array as requested").  The
    underlying pybind call returns plain ``(prob, label)`` tuples and skips
    numpy entirely, so we use that and keep ``predict()`` only as a fallback
    for a future fixed release.
    """
    try:
        for prob, label in model.f.predict(line + "\n", -1, 0.0, "strict"):
            if label == "__label__psych":
                return float(prob)
        return 0.0
    except AttributeError:
        labels, probs = model.predict(line, k=-1)
        for label, prob in zip(labels, probs):
            if label == "__label__psych":
                return float(prob)
        return 0.0


class TopicClassifier:
    """Thin wrapper so `apply` doesn't care which backend trained the model."""

    def __init__(self, backend: str, model, meta: dict):
        self.backend = backend
        self.model = model
        self.meta = meta

    # -- training ---------------------------------------------------------- #
    @classmethod
    def train(cls, pos: list[str], neg: list[str], outdir: Path, seed: int = C.SEED) -> "TopicClassifier":
        outdir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed)
        rows = [(1, t) for t in pos] + [(0, t) for t in neg]
        rng.shuffle(rows)
        split = int(len(rows) * 0.9)
        train_rows, test_rows = rows[:split], rows[split:]

        backend = "fasttext"
        try:
            import fasttext
        except ImportError as exc:
            print(f"[train] fasttext unavailable ({exc}); falling back to sklearn", file=sys.stderr)
            backend = "sklearn"

        if backend == "fasttext":
            train_path = outdir / "train.ft.txt"
            with open(train_path, "w", encoding="utf-8") as fh:
                for label, text in train_rows:
                    fh.write(f"__label__{'psych' if label else 'other'} {_ft_line(text)}\n")
            try:
                model = fasttext.train_supervised(
                    input=str(train_path), epoch=25, lr=0.5, wordNgrams=2,
                    dim=100, minCount=2, loss="softmax", bucket=200_000, seed=seed,
                    thread=4, verbose=1,
                )
                model.save_model(str(outdir / "model.bin"))
            except Exception as exc:
                print(f"[train] fastText training failed ({exc}); falling back to sklearn", file=sys.stderr)
                backend = "sklearn"

        if backend == "sklearn":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline

            model = Pipeline([
                ("tfidf", TfidfVectorizer(
                    max_features=200_000, ngram_range=(1, 2), sublinear_tf=True,
                    min_df=2, strip_accents="unicode")),
                ("clf", LogisticRegression(max_iter=2000, C=4.0, random_state=seed)),
            ])
            model.fit([_ft_line(t) for _, t in train_rows], [l for l, _ in train_rows])
            with open(outdir / "model.pkl", "wb") as fh:
                pickle.dump(model, fh)

        self = cls(backend, model, {"backend": backend, "seed": seed})
        self.meta.update(
            n_pos=len(pos), n_neg=len(neg),
            **self.evaluate(test_rows),
        )
        (outdir / "meta.json").write_text(json.dumps(self.meta, indent=2) + "\n")
        return self

    def evaluate(self, rows: list[tuple[int, str]], threshold: float = 0.5) -> dict:
        if not rows:
            return {}
        tp = fp = fn = tn = 0
        for label, text in rows:
            pred = 1 if self.score(text) >= threshold else 0
            if label and pred:
                tp += 1
            elif label and not pred:
                fn += 1
            elif not label and pred:
                fp += 1
            else:
                tn += 1
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        return {
            "holdout_n": len(rows),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        }

    # -- inference --------------------------------------------------------- #
    @classmethod
    def load(cls, outdir: Path = CLASSIFIER_DIR) -> "TopicClassifier":
        meta_path = outdir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no classifier at {outdir} — run `python -m src.data.topic_filter train` first"
            )
        meta = json.loads(meta_path.read_text())
        if meta["backend"] == "fasttext":
            import fasttext
            model = fasttext.load_model(str(outdir / "model.bin"))
        else:
            with open(outdir / "model.pkl", "rb") as fh:
                model = pickle.load(fh)
        return cls(meta["backend"], model, meta)

    def score(self, text: str) -> float:
        line = _ft_line(text)
        if not line:
            return 0.0
        if self.backend == "fasttext":
            return _ft_psych_prob(self.model, line)
        return float(self.model.predict_proba([line])[0][1])


# --------------------------------------------------------------------------- #
# subcommand: train
# --------------------------------------------------------------------------- #

def cmd_train(args) -> int:
    outdir = Path(args.outdir)
    if (outdir / "meta.json").exists() and not args.force:
        print(f"[train] classifier already at {outdir} — skipping (--force to retrain)")
        print(json.dumps(json.loads((outdir / "meta.json").read_text()), indent=2))
        return 0

    rng = random.Random(C.SEED)
    pos: list[str] = []
    neg: list[str] = []
    scanned = 0
    min_distinct = args.min_distinct

    for row in C.progress(C.iter_source("fineweb_edu", limit=args.scan_docs), desc="seed scan"):
        scanned += 1
        text = C.norm_ws(row.get("text") or "")
        if len(text) < MIN_CHARS:
            continue
        distinct, _ = keyword_score(text, PSYCH_RE)
        strong, _ = keyword_score(text, PSYCH_STRONG_RE)
        if distinct >= min_distinct and strong >= 1 and len(pos) < args.n_pos:
            pos.append(text)
        elif distinct == 0 and len(neg) < args.n_neg and rng.random() < args.neg_rate:
            neg.append(text)
        if len(pos) >= args.n_pos and len(neg) >= args.n_neg:
            break

    print(f"[train] scanned {scanned:,} docs -> {len(pos):,} positives, {len(neg):,} negatives")
    if len(pos) < 200:
        print(
            f"[train] ERROR: only {len(pos)} positives; raise --scan-docs or lower "
            "--min-distinct", file=sys.stderr,
        )
        return 1
    if len(neg) < 200:
        print(f"[train] ERROR: only {len(neg)} negatives; raise --scan-docs", file=sys.stderr)
        return 1

    clf = TopicClassifier.train(pos, neg, outdir)
    clf.meta["scanned"] = scanned
    (outdir / "meta.json").write_text(json.dumps(clf.meta, indent=2) + "\n")
    print(f"[train] backend={clf.backend} " + json.dumps(
        {k: v for k, v in clf.meta.items() if k in ("precision", "recall", "f1", "holdout_n")}))
    return 0


# --------------------------------------------------------------------------- #
# subcommand: apply
# --------------------------------------------------------------------------- #

def general_offset() -> int:
    """Rows of FineWeb-Edu the `general` slice already consumed."""
    marker = C.read_marker(C.CLEAN_DIR / "general")
    return int(marker.get("fineweb_docs_scanned", 0))


def cmd_apply(args) -> int:
    outdir = C.CLEAN_DIR / "psych"
    if C.is_complete(outdir) and not args.force:
        print("[psych] already complete — skipping (--force to redo)")
        return 0
    C.clear_dir(outdir)

    clf = TopicClassifier.load(Path(args.classifier))
    skip = args.skip_docs if args.skip_docs is not None else general_offset()
    target = int(args.target_tokens) if args.target_tokens else _default_target("psych", args.config)
    print(f"[psych] backend={clf.backend} threshold={args.threshold} "
          f"skip={skip:,} target=~{C.human(target)} tokens")
    if skip == 0:
        print("[psych] WARNING: general slice offset is 0 — run clean_general.py first "
              "or the psych slice will overlap the general slice.", file=sys.stderr)

    writer = C.ShardWriter(outdir, "psych")
    seen: set[str] = set()
    scanned = kept = 0
    tokens = 0
    for row in C.progress(C.iter_source("fineweb_edu", limit=args.limit, skip=skip), desc="psych"):
        scanned += 1
        text = C.norm_ws(row.get("text") or "")
        if len(text) < MIN_CHARS:
            continue
        prob = clf.score(text)
        if prob < args.threshold:
            continue
        key = C.doc_hash(text)
        if key in seen:
            continue
        seen.add(key)
        writer.write({
            "text": text,
            "meta": {
                "source": "HuggingFaceFW/fineweb-edu:sample-10BT",
                "id": row.get("id"), "url": row.get("url"),
                "edu_score": row.get("score"), "psych_score": round(prob, 4),
            },
        })
        kept += 1
        tokens += C.est_tokens(len(text))
        if tokens >= target:
            break

    stats = writer.close()
    stats.update(domain="psych", scanned=scanned, skip_docs=skip,
                 threshold=args.threshold, keep_rate=round(kept / max(scanned, 1), 4),
                 classifier=clf.meta, target_tokens=target,
                 source="HuggingFaceFW/fineweb-edu:sample-10BT")
    C.write_marker(outdir, stats)
    print(f"[psych] kept {kept:,}/{scanned:,} ({stats['keep_rate']:.1%}), "
          f"~{C.human(stats['est_tokens'])} est tokens")
    if tokens < target:
        print(f"[psych] NOTE: ~{C.human(tokens)} of {C.human(target)} target — "
              "mix.py will repeat this domain up to max_domain_epochs.", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- #
# subcommand: tilt-cosmopedia
# --------------------------------------------------------------------------- #

def cmd_tilt(args) -> int:
    outdir = C.CLEAN_DIR / "textbooks"
    if C.is_complete(outdir) and not args.force:
        print("[textbooks] already complete — skipping (--force to redo)")
        return 0
    C.clear_dir(outdir)

    target = int(args.target_tokens) if args.target_tokens else _default_target("textbooks", args.config)
    topical_target = int(target * args.tilt_fraction)
    print(f"[textbooks] target ~{C.human(target)} tokens, "
          f"{args.tilt_fraction:.0%} topical (~{C.human(topical_target)}), "
          f"filler every {args.filler_every} docs")

    writer = C.ShardWriter(outdir, "textbooks")
    seen: set[str] = set()
    scanned = topical = filler = 0
    tok_topical = tok_filler = 0
    since_filler = 0

    for row in C.progress(C.iter_source("cosmopedia", limit=args.limit), desc="textbooks"):
        scanned += 1
        text = C.norm_ws(row.get("text") or "")
        if len(text) < MIN_CHARS:
            continue
        prompt = row.get("prompt") or ""
        distinct, total = keyword_score(prompt + "\n" + text[:4000], COSMO_RE)
        is_topical = distinct >= args.min_distinct and total >= args.min_hits
        since_filler += 1
        if is_topical and tok_topical >= topical_target:
            is_topical = False  # topical bucket full; treat as a filler candidate
        if not is_topical:
            if since_filler < args.filler_every or tok_filler >= target - topical_target:
                continue
            since_filler = 0
        key = C.doc_hash(text)
        if key in seen:
            continue
        seen.add(key)
        writer.write({
            "text": text,
            "meta": {
                "source": "HuggingFaceTB/smollm-corpus:cosmopedia-v2",
                "audience": row.get("audience"), "format": row.get("format"),
                "seed_data": row.get("seed_data"),
                "topical": is_topical, "topic_hits": distinct,
            },
        })
        n = C.est_tokens(len(text))
        if is_topical:
            topical += 1
            tok_topical += n
        else:
            filler += 1
            tok_filler += n
        if tok_topical + tok_filler >= target:
            break

    stats = writer.close()
    stats.update(domain="textbooks", scanned=scanned, topical_docs=topical,
                 filler_docs=filler, topical_tokens=tok_topical, filler_tokens=tok_filler,
                 target_tokens=target, source="HuggingFaceTB/smollm-corpus:cosmopedia-v2")
    C.write_marker(outdir, stats)
    share = tok_topical / max(tok_topical + tok_filler, 1)
    print(f"[textbooks] {topical:,} topical + {filler:,} filler docs "
          f"({share:.0%} topical tokens), ~{C.human(stats['est_tokens'])} est tokens "
          f"from {scanned:,} scanned")
    return 0


def _default_target(domain: str, cfg_path: str | None) -> int:
    cfg = load_config(cfg_path)
    ratio = cfg.data.mix.get(domain, 0.0)
    return int(cfg.train.total_tokens * ratio * SLACK) + cfg.data.val_tokens_per_domain


# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="bootstrap the psych classifier")
    t.add_argument("--scan-docs", type=int, default=200_000)
    t.add_argument("--n-pos", type=int, default=2_000)
    t.add_argument("--n-neg", type=int, default=4_000)
    t.add_argument("--neg-rate", type=float, default=0.05, help="sampling rate for negatives")
    t.add_argument("--min-distinct", type=int, default=4, help="distinct keywords for a seed positive")
    t.add_argument("--outdir", default=str(CLASSIFIER_DIR))
    t.add_argument("--force", action="store_true")
    t.set_defaults(func=cmd_train)

    a = sub.add_parser("apply", help="filter FineWeb-Edu into data/clean/psych")
    a.add_argument("--target-tokens", type=float, default=None)
    a.add_argument("--threshold", type=float, default=0.70)
    a.add_argument("--skip-docs", type=int, default=None,
                   help="default: however many FineWeb-Edu rows the general slice used")
    a.add_argument("--limit", type=int, default=None, help="max docs to scan (smoke testing)")
    a.add_argument("--classifier", default=str(CLASSIFIER_DIR))
    a.add_argument("--config", default=None)
    a.add_argument("--force", action="store_true")
    a.set_defaults(func=cmd_apply)

    c = sub.add_parser("tilt-cosmopedia", help="build the topic-tilted textbooks slice")
    c.add_argument("--target-tokens", type=float, default=None)
    c.add_argument("--tilt-fraction", type=float, default=0.6,
                   help="share of the slice that must be topical")
    c.add_argument("--min-distinct", type=int, default=3)
    c.add_argument("--min-hits", type=int, default=5)
    c.add_argument("--filler-every", type=int, default=10,
                   help="keep 1 in N non-topical docs as filler")
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--config", default=None)
    c.add_argument("--force", action="store_true")
    c.set_defaults(func=cmd_tilt)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
