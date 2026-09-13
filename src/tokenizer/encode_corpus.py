"""Encode the mixed corpus to memmap-able uint16 .bin shards (nanoGPT-style).

Output layout (see docs/contracts.md):
    data/tokenized/<domain>_train_0000.bin   contiguous uint16 token ids,
    data/tokenized/<domain>_val.bin          docs separated by <|endoftext|> (id 0)
    data/tokenized/meta.json                 token counts, tokenizer path, dtype

Why pre-tokenize to flat binary: during training the GPU must never wait on
CPU tokenization; a memmapped uint16 array gives zero-copy random windows and
the whole 6B-token corpus is only ~12GB on disk.

Usage:
    uv run python src/tokenizer/encode_corpus.py [--manifest ...] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR  # noqa: E402

SHARD_TOKENS = 250_000_000  # ~500MB per uint16 shard
EOT_ID = 0


def encode_domain(tok: Tokenizer, entries: list[dict], out_dir: Path, domain: str,
                  split: str) -> int:
    """Stream-encode one domain's jsonl files into shard(s). Honors per-entry
    'epochs' (float): full repeats + a fractional prefix pass."""
    shard_idx, buf, total = 0, [], 0
    buf_len = 0

    def flush(final: bool) -> None:
        nonlocal shard_idx, buf, buf_len
        if not buf or (not final and buf_len < SHARD_TOKENS):
            return
        name = f"{domain}_val.bin" if split == "val" else f"{domain}_train_{shard_idx:04d}.bin"
        arr = np.concatenate(buf)
        path = out_dir / name
        if split == "val" or not path.exists():
            arr.tofile(path)
        else:  # multiple flushes into numbered train shards
            arr.tofile(path)
        buf, buf_len = [], 0
        shard_idx += 1

    for entry in entries:
        epochs = float(entry.get("epochs", 1.0))
        full, frac = int(epochs), epochs - int(epochs)
        lines = Path(entry["file"]).read_text(encoding="utf-8").splitlines()
        passes = [1.0] * full + ([frac] if frac > 0.005 else [])
        for take in passes:
            n_lines = max(1, int(len(lines) * take))
            for start in range(0, n_lines, 1000):  # chunked to bound memory
                batch_texts = [json.loads(ln)["text"] for ln in lines[start:start + 1000]]
                for enc in tok.encode_batch(batch_texts):
                    ids = np.asarray(enc.ids + [EOT_ID], dtype=np.uint16)
                    buf.append(ids)
                    buf_len += len(ids)
                    total += len(ids)
                    if buf_len >= SHARD_TOKENS:
                        flush(final=False)
    flush(final=True)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA_DIR / "clean" / "manifest.json"))
    ap.add_argument("--tokenizer", default=str(DATA_DIR / "tokenized" / "tokenizer.json"))
    ap.add_argument("--out", default=str(DATA_DIR / "tokenized"))
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    assert tok.get_vocab_size() <= 65536, "uint16 requires vocab <= 65536"
    manifest = json.loads(Path(args.manifest).read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {"dtype": "uint16", "eot_id": EOT_ID, "tokenizer": args.tokenizer,
            "vocab_size": tok.get_vocab_size(), "tokens": {"train": {}, "val": {}}}

    for split in ("train", "val"):
        by_domain: dict[str, list] = {}
        for e in manifest["entries"]:
            if e["split"] == split:
                by_domain.setdefault(e["domain"], []).append(e)
        for domain, entries in tqdm(by_domain.items(), desc=split):
            n = encode_domain(tok, entries, out_dir, domain, split)
            meta["tokens"][split][domain] = n
            print(f"  {split}/{domain}: {n/1e6:.1f}M tokens")

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    grand = sum(meta["tokens"]["train"].values())
    print(f"total train tokens: {grand/1e9:.2f}B -> {out_dir}/meta.json")


if __name__ == "__main__":
    main()
