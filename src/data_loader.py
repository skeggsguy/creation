"""Memmapped, domain-weighted batch sampling over pre-tokenized .bin shards.

WHY a memmap of raw uint16 instead of a "real" dataset library?
--------------------------------------------------------------
Pretraining data access is the dumbest possible workload: read
`seq_len + 1` contiguous token ids from a random place, a few dozen times per
step.  A tokenized corpus is just a flat array of token ids, so we store it as
exactly that — a headerless file of little-endian uint16 (our vocab is 24576,
which fits in 16 bits; uint16 halves the file size and the page-cache pressure
versus int32).  `np.memmap` then lets the OS page cache do all the work: the
hot parts of the corpus live in RAM, we never write a loader thread, and
startup is instant no matter how big the corpus is.  This is the nanoGPT
approach and it is genuinely hard to beat.

WHY sample random windows instead of streaming sequentially?
------------------------------------------------------------
Random windows give every batch an i.i.d. sample of the corpus, which is what
SGD assumes.  Sequential reading would correlate the examples inside a batch
(all from the same book/domain), which raises gradient variance in exactly the
wrong way.  The cost is that documents get cut at arbitrary points and we do
not see every token exactly once per "epoch" — for a corpus we are only going
to pass over a handful of times, that is a fine trade.

WHY domain-weighted mixing per *row* rather than per *batch*?
-------------------------------------------------------------
Our corpus is a deliberate blend (55% general web-edu, 12% textbooks, 9%
philosophy, ...).  If a whole batch came from one domain, the gradient for that
step would point at that domain only, and the model would lurch between
registers.  Choosing the domain independently for each of the B rows makes
every single step an unbiased sample of the *mixture*, so the blend ratio is
enforced continuously instead of on average over many steps.

THE RESUME INVARIANT
--------------------
`TrainStream.batch(step)` is a pure function of (seed, step, batch_size,
seq_len) — there is no iterator cursor, no shuffle buffer, no RNG carried in
the checkpoint.  A resumed run that sets `step` back to where it left off
replays exactly the same token windows, which is what lets
tests/test_resume.py compare an interrupted run against an uninterrupted one
loss-for-loss.  If you ever change this file's sampling maths you break that
test on purpose — that is the alarm working.

The one thing that legitimately changes the stream is the corpus itself: adding
a shard between sessions changes which window each (seed, step) maps to from
that point on.  That is fine — the guarantee we need is that a *resumed*
session continues the stream the killed one was on, not that history stays
replayable forever.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, DataConfig  # noqa: E402

# Token id dtype on disk, per docs/contracts.md.  Also the reason the tokenizer
# vocab is capped at 65535.
TOKEN_DTYPE = np.uint16


# --------------------------------------------------------------------------
# Corpus discovery
# --------------------------------------------------------------------------


@dataclass
class DomainData:
    """All shards belonging to one domain, plus a sampling index."""

    name: str
    shards: list[np.memmap]
    paths: list[Path]
    val: np.memmap | None
    val_path: Path | None

    @property
    def n_tokens(self) -> int:
        return int(sum(len(s) for s in self.shards))

    @property
    def n_val_tokens(self) -> int:
        return 0 if self.val is None else int(len(self.val))


class Corpus:
    """Opens `data/tokenized/` and exposes per-domain memmaps.

    Layout (docs/contracts.md):
        <domain>_train_0000.bin, <domain>_train_0001.bin, ...   uint16
        <domain>_val.bin                                        uint16
        meta.json      {"tokenizer": ..., token counts per domain}

    We treat meta.json as advisory: the files on disk are the truth, because
    the tokenizer pipeline is owned by another component and may add shards
    between sessions.  meta.json is used for sanity-printing and to surface the
    tokenizer path.
    """

    def __init__(self, data_cfg: DataConfig, require_val: bool = True):
        self.dir = Path(data_cfg.tokenized_dir)
        self.mix = dict(data_cfg.mix)
        self.max_domain_epochs = data_cfg.max_domain_epochs
        if not self.dir.is_dir():
            raise FileNotFoundError(
                f"tokenized dir {self.dir} does not exist — run the tokenizer pipeline first"
            )

        self.meta: dict = {}
        meta_path = self.dir / "meta.json"
        if meta_path.exists():
            try:
                self.meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError as e:  # advisory only; never fatal
                print(f"[data] warning: could not parse {meta_path}: {e}")

        # Normalise the mixture so rounding in a hand-edited TOML can't make
        # np.random.choice complain about probabilities not summing to 1.
        total_w = sum(self.mix.values())
        if total_w <= 0:
            raise ValueError("data.mix weights must be positive")
        self.domains: list[DomainData] = []
        self.weights: list[float] = []
        for name, w in self.mix.items():
            paths = sorted(self.dir.glob(f"{name}_train_*.bin"))
            if not paths:
                raise FileNotFoundError(
                    f"no shards matching {name}_train_*.bin in {self.dir} "
                    f"(domain '{name}' is in data.mix with weight {w})"
                )
            shards = [np.memmap(p, dtype=TOKEN_DTYPE, mode="r") for p in paths]
            vp = self.dir / f"{name}_val.bin"
            if not vp.exists():
                if require_val:
                    raise FileNotFoundError(f"missing validation shard {vp}")
                vp = None
            val = np.memmap(vp, dtype=TOKEN_DTYPE, mode="r") if vp else None
            self.domains.append(DomainData(name, shards, paths, val, vp))
            self.weights.append(w / total_w)

        self.names = [d.name for d in self.domains]
        # Per-domain shard-choice probabilities, proportional to shard length so
        # that a random token position is uniform over the whole domain rather
        # than uniform over shards (shards are rarely equal-sized).
        self.shard_p = [
            np.array([len(s) for s in d.shards], dtype=np.float64) for d in self.domains
        ]
        self.shard_p = [p / p.sum() for p in self.shard_p]

    def summary(self) -> str:
        lines = [f"corpus @ {self.dir}"]
        tot = sum(d.n_tokens for d in self.domains)
        for d, w in zip(self.domains, self.weights):
            lines.append(
                f"  {d.name:<11} {d.n_tokens/1e6:9.2f}M train  "
                f"{d.n_val_tokens/1e6:6.2f}M val  mix={w:.3f}  shards={len(d.shards)}"
            )
        lines.append(f"  {'TOTAL':<11} {tot/1e6:9.2f}M train tokens")
        return "\n".join(lines)

    def epoch_warnings(self, total_tokens: float) -> list[str]:
        """Which domains will be repeated more than `max_domain_epochs` times?

        Repeating a small themed slice a few times is fine (decisions.md allows
        2–4 epochs); repeating it 40 times would just memorise it.  We surface
        this at startup rather than discovering it in the loss curves.
        """
        out = []
        for d, w in zip(self.domains, self.weights):
            if d.n_tokens == 0:
                continue
            epochs = total_tokens * w / d.n_tokens
            if epochs > self.max_domain_epochs:
                out.append(
                    f"domain '{d.name}' would be seen {epochs:.1f}x over "
                    f"{total_tokens/1e9:.2f}B tokens (max_domain_epochs="
                    f"{self.max_domain_epochs}) — consider more data or less weight"
                )
        return out


# --------------------------------------------------------------------------
# Training stream
# --------------------------------------------------------------------------


class TrainStream:
    """Deterministic, domain-weighted batch sampler.

    Yields `(x, y)` int32 arrays of shape (batch_size, seq_len) where
    `y` is `x` shifted left by one — the next-token-prediction target.  We read
    `seq_len + 1` tokens and split, so every position in the window has a real
    target and we never waste a token.
    """

    def __init__(self, corpus: Corpus, batch_size: int, seq_len: int, seed: int):
        self.corpus = corpus
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.seed = seed
        self.tokens_per_batch = batch_size * seq_len
        self._weights = np.array(corpus.weights, dtype=np.float64)

    def _rng(self, step: int) -> np.random.Generator:
        # SeedSequence mixes (seed, step) with a hash, so consecutive steps are
        # statistically independent — unlike `default_rng(seed + step)`, whose
        # streams for adjacent seeds are only *nominally* independent.
        return np.random.default_rng(np.random.SeedSequence([self.seed, step]))

    def batch(self, step: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Return (x, y, domain_names) for the given global step.

        Pure function of (seed, step, batch_size, seq_len): this is the resume
        guarantee.  The third return value is only used for logging/debugging.
        """
        rng = self._rng(step)
        need = self.seq_len + 1
        buf = np.empty((self.batch_size, need), dtype=np.int32)
        picks: list[str] = []

        # Vectorised domain draw first (one call, B samples) — then a tiny
        # Python loop for the per-row window, which costs microseconds against
        # a multi-second training step.
        dom_idx = rng.choice(len(self.corpus.domains), size=self.batch_size, p=self._weights)
        for row, di in enumerate(dom_idx):
            dom = self.corpus.domains[di]
            si = int(rng.choice(len(dom.shards), p=self.corpus.shard_p[di]))
            shard = dom.shards[si]
            hi = len(shard) - need
            if hi <= 0:
                # Shard shorter than one window: pad by wrapping around.  Only
                # happens with toy/synthetic corpora.
                idx = np.arange(need) % len(shard)
                buf[row] = shard[idx].astype(np.int32)
            else:
                off = int(rng.integers(0, hi + 1))
                buf[row] = shard[off : off + need].astype(np.int32)
            picks.append(dom.name)

        return buf[:, :-1], buf[:, 1:], picks


def val_batches(
    corpus: Corpus, domain: str, batch_size: int, seq_len: int, max_batches: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fixed, non-overlapping validation windows for one domain.

    Validation must be *the same tokens every time* or the val curve measures
    noise rather than progress — hence no RNG here at all: we simply walk
    `<domain>_val.bin` from the front in contiguous windows.  The number of
    batches is capped so validation costs a fixed, small slice of wall-clock.
    """
    dom = next((d for d in corpus.domains if d.name == domain), None)
    if dom is None or dom.val is None:
        return []
    need = seq_len + 1
    per_batch = batch_size * need
    n = min(max_batches, len(dom.val) // per_batch)
    out = []
    for b in range(n):
        chunk = np.asarray(dom.val[b * per_batch : (b + 1) * per_batch], dtype=np.int32)
        chunk = chunk.reshape(batch_size, need)
        out.append((chunk[:, :-1], chunk[:, 1:]))
    return out


# --------------------------------------------------------------------------
# Synthetic corpus — so tests never depend on the real (multi-GB) data
# --------------------------------------------------------------------------


def make_synthetic_corpus(
    tmpdir: str | Path,
    vocab: int = 512,
    domains: dict[str, float] | None = None,
    tokens_per_domain: int = 200_000,
    val_tokens: int = 40_000,
    shards_per_domain: int = 2,
    seed: int = 0,
    learnable: bool = True,
) -> DataConfig:
    """Write a tiny fake tokenized corpus and return a DataConfig pointing at it.

    `learnable=True` generates a *structured* stream (a small random Markov
    chain over the vocab) rather than uniform noise.  That matters for
    tests/test_overfit.py: uniform-random tokens have an irreducible loss of
    log(vocab) on unseen data, and while a model can still memorise one fixed
    batch, a structured stream makes the loss curve look like real training and
    catches bugs (e.g. a broken causal mask) that noise would hide.
    """
    tmpdir = Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    domains = domains or {"general": 0.7, "haiku": 0.3}
    rng = np.random.default_rng(seed)

    # A sparse transition table: each token id has ~4 plausible successors.
    succ = rng.integers(0, vocab, size=(vocab, 4), dtype=np.int64)

    def gen(n: int, s: int) -> np.ndarray:
        r = np.random.default_rng(s)
        out = np.empty(n, dtype=TOKEN_DTYPE)
        cur = int(r.integers(0, vocab))
        if learnable:
            choices = r.integers(0, 4, size=n)
            for i in range(n):
                out[i] = cur
                cur = int(succ[cur, choices[i]])
        else:
            out[:] = r.integers(0, vocab, size=n, dtype=np.int64).astype(TOKEN_DTYPE)
        return out

    meta: dict = {"tokenizer": str(tmpdir / "tokenizer.json"), "vocab_size": vocab, "domains": {}}
    for di, name in enumerate(domains):
        per_shard = max(tokens_per_domain // shards_per_domain, 1024)
        for s in range(shards_per_domain):
            gen(per_shard, seed * 1000 + di * 10 + s).tofile(tmpdir / f"{name}_train_{s:04d}.bin")
        gen(val_tokens, seed * 1000 + di * 10 + 99).tofile(tmpdir / f"{name}_val.bin")
        meta["domains"][name] = {
            "train_tokens": per_shard * shards_per_domain,
            "val_tokens": val_tokens,
            "shards": shards_per_domain,
        }
    (tmpdir / "meta.json").write_text(json.dumps(meta, indent=2))

    return DataConfig(
        tokenized_dir=str(tmpdir),
        mix=dict(domains),
        val_tokens_per_domain=val_tokens,
    )


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import tempfile

    from src.config import load_config

    ap = argparse.ArgumentParser(description="tokenized-corpus loader")
    ap.add_argument("--config", default=None)
    ap.add_argument("--real", action="store_true", help="open the real corpus instead of a synthetic one")
    args = ap.parse_args()

    cfg: Config = load_config(args.config)
    failures = []

    if args.real:
        corpus = Corpus(cfg.data)
        seq_len, bs = cfg.model.seq_len, 8
    else:
        with tempfile.TemporaryDirectory() as td:
            dc = make_synthetic_corpus(td, vocab=512, tokens_per_domain=60_000)
            corpus = Corpus(dc)
            seq_len, bs = 64, 8
            print(corpus.summary())

            stream = TrainStream(corpus, batch_size=bs, seq_len=seq_len, seed=1337)
            x0, y0, doms = stream.batch(7)
            x0b, y0b, domsb = stream.batch(7)
            if not (np.array_equal(x0, x0b) and doms == domsb):
                failures.append("batch(step) is not deterministic")
            x1, _, _ = stream.batch(8)
            if np.array_equal(x0, x1):
                failures.append("consecutive steps produced identical batches")
            if not np.array_equal(x0[:, 1:], y0[:, :-1]):
                failures.append("y is not x shifted by one")
            if x0.shape != (bs, seq_len):
                failures.append(f"bad shape {x0.shape}")

            # Mixture check: over many steps the domain draw should match `mix`.
            counts = {n: 0 for n in corpus.names}
            for s in range(400):
                for d in TrainStream(corpus, 8, seq_len, 1337).batch(s)[2]:
                    counts[d] += 1
            tot = sum(counts.values())
            for name, w in zip(corpus.names, corpus.weights):
                got = counts[name] / tot
                print(f"  mix {name:<10} target={w:.3f} observed={got:.3f}")
                if abs(got - w) > 0.03:
                    failures.append(f"mix for {name}: {got:.3f} vs {w:.3f}")

            vb = val_batches(corpus, corpus.names[0], batch_size=4, seq_len=seq_len, max_batches=3)
            vb2 = val_batches(corpus, corpus.names[0], batch_size=4, seq_len=seq_len, max_batches=3)
            if not vb or not all(np.array_equal(a[0], b[0]) for a, b in zip(vb, vb2)):
                failures.append("val batches not fixed/deterministic")
            print(f"  val batches: {len(vb)} x {vb[0][0].shape}")

    for f in failures:
        print(f"FAIL  {f}")
    print(f"{'FAILED' if failures else 'OK'}  data_loader self-test ({len(failures)} failures)")
    sys.exit(1 if failures else 0)
