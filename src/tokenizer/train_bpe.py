"""Train our own byte-level BPE tokenizer (24,576 vocab) on the actual corpus mix.

Why train our own instead of reusing GPT-2's 50k vocab: at ~300M params a 50k
vocab puts ~50M params (with untied head, double that) into embedding tables.
A 24k vocab trained on OUR text keeps the embedding share modest AND allocates
merges to the diction that actually appears in the corpus (Stoic philosophy,
pulp sci-fi, early-20th-century humor) instead of Reddit-2019 English.

Byte-level means no unknown tokens ever: any UTF-8 string decomposes into the
256 base byte tokens, and BPE merges build frequent words/subwords on top.

Usage:
    uv run python src/tokenizer/train_bpe.py [--manifest data/clean/manifest.json]
        [--sample-tokens 300000000] [--out data/tokenized/tokenizer.json]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, load_config  # noqa: E402

ENDOFTEXT = "<|endoftext|>"  # document separator, id 0


def iter_manifest_texts(manifest_path: Path, sample_chars: int, seed: int):
    """Yield doc texts from the mix manifest, proportionally to the mix weights
    (the manifest lists each domain's files; we interleave by domain weight so
    the tokenizer sees the same distribution the model will)."""
    manifest = json.loads(manifest_path.read_text())
    entries = [e for e in manifest["entries"] if e["split"] == "train"]
    rng = random.Random(seed)
    by_domain: dict[str, list] = {}
    for e in entries:
        by_domain.setdefault(e["domain"], []).append(e)

    cfg = load_config()
    domains = list(by_domain)
    weights = [cfg.data.mix.get(d, 0.01) for d in domains]
    budget_per_domain = {
        d: int(sample_chars * w / sum(weights)) for d, w in zip(domains, weights)
    }

    for domain, files in by_domain.items():
        remaining = budget_per_domain[domain]
        files = files.copy()
        rng.shuffle(files)
        for entry in files:
            if remaining <= 0:
                break
            with open(entry["file"], encoding="utf-8") as f:
                for line in f:
                    if remaining <= 0:
                        break
                    text = json.loads(line)["text"]
                    remaining -= len(text)
                    yield text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA_DIR / "clean" / "manifest.json"))
    ap.add_argument("--sample-tokens", type=int, default=300_000_000,
                    help="approx tokens of training sample (chars/4 heuristic)")
    ap.add_argument("--out", default=str(DATA_DIR / "tokenized" / "tokenizer.json"))
    ap.add_argument("--vocab-size", type=int, default=None, help="default: config vocab_size")
    args = ap.parse_args()

    cfg = load_config()
    vocab_size = args.vocab_size or cfg.model.vocab_size

    tokenizer = Tokenizer(models.BPE(byte_fallback=False))
    # ByteLevel pre-tokenizer: split on whitespace/punctuation boundaries and map
    # raw bytes to printable stand-ins (same scheme GPT-2 uses).
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[ENDOFTEXT],           # id 0 by construction
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    sample_chars = args.sample_tokens * 4
    print(f"training {vocab_size}-vocab byte-level BPE on ~{sample_chars/1e9:.1f}G chars")
    tokenizer.train_from_iterator(
        iter_manifest_texts(Path(args.manifest), sample_chars, cfg.train.seed),
        trainer=trainer,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out))
    print(f"saved {out}")

    # Report compression on a held-back sample (target: >= 3.8 chars/token).
    probe = []
    for i, text in enumerate(iter_manifest_texts(Path(args.manifest), 2_000_000, cfg.train.seed + 1)):
        probe.append(text)
        if i > 500:
            break
    joined = "\n".join(probe)
    n_tok = len(tokenizer.encode(joined).ids)
    print(f"compression: {len(joined)/n_tok:.2f} chars/token on {len(probe)} held-back docs")


if __name__ == "__main__":
    main()
