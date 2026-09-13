#!/usr/bin/env python3
"""Can the model learn at all?  Overfit ONE batch, on both backends.

WHY THIS TEST EXISTS
====================
This is the single highest-value test in a from-scratch LLM project.  A
transformer with a subtly broken component — a causal mask that leaks the
future, a residual connection dropped, RoPE applied to the wrong axis, an
optimizer that never updates a parameter group — will still produce a loss
curve that *slowly goes down*, because predicting token frequencies alone gets
you a long way.  You can lose a week to that.

Memorising a single fixed batch is the diagnostic that separates the two cases.
A correct model has far more parameters than the ~500 tokens in the batch, so
it can drive the loss to ~0 by pure memorisation within a few hundred steps.  A
model with a broken gradient path *cannot*, no matter how long you run it.  So:

    loss -> ~0 on one batch  =>  the forward pass, the backward pass, the
                                 optimizer, and the data plumbing all work
    loss stalls              =>  something is structurally wrong; find it now,
                                 not 12 hours into an overnight run

We run it on both backends to catch a divergence between `model.py` (MLX) and
`model_torch.py` — the two must stay architecturally identical, and the easiest
way for them to silently drift is for one of them to be subtly broken.

Run: `uv run python tests/test_overfit.py`   (exits non-zero on failure)
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, ModelConfig, TrainConfig  # noqa: E402
from src.data_loader import Corpus, TrainStream, make_synthetic_corpus  # noqa: E402
from src.train import make_trainer  # noqa: E402

MAX_STEPS = 300
TARGET_LOSS = 0.5
LR = 3e-3            # aggressive: we *want* to overfit, fast
BATCH, SEQ, VOCAB = 8, 64, 512


def tiny_config(tokenized_dir: str) -> Config:
    """A model small enough to train in seconds, big enough to be a transformer."""
    cfg = Config()
    cfg.model = ModelConfig(
        d_model=128, n_layers=2, n_heads=4, ffn_hidden=352,
        vocab_size=VOCAB, seq_len=SEQ,
    )
    cfg.train = TrainConfig(
        batch_size=BATCH, peak_lr=LR, warmup_tokens=0, grad_clip=1.0, seed=1234,
    )
    cfg.data.tokenized_dir = tokenized_dir
    cfg.data.mix = {"general": 1.0}
    cfg.run_name = "test_overfit"
    return cfg


def run_backend(backend: str, cfg: Config, x, y) -> tuple[float, list[float]]:
    trainer = make_trainer(cfg, backend, compile_step=True)
    trainer.set_lr(LR)
    curve = []
    for step in range(MAX_STEPS):
        loss, _gnorm = trainer.train_step(x, y)
        curve.append(loss)
        if loss < TARGET_LOSS:
            break
    return curve[-1], curve


def check_causal(backend: str, cfg: Config) -> str | None:
    """The other half of 'does the model work': does it cheat?

    Overfitting one batch proves the gradients flow, but it cannot detect a
    *leaky* causal mask — seeing the future only makes memorisation easier.  A
    leak is the most expensive bug possible here (the loss curve looks
    wonderful and the model is worthless at generation), so we test it
    directly: change the tokens *after* position t and the logits at position t
    must not move at all.
    """
    import numpy as np
    trainer = make_trainer(cfg, backend, compile_step=False)
    rng = np.random.default_rng(0)
    T = 16
    a = rng.integers(0, VOCAB, size=(1, T)).astype(np.int32)
    b = a.copy()
    b[0, T // 2 :] = rng.integers(0, VOCAB, size=T - T // 2)  # rewrite the future

    model = getattr(trainer, "raw_model", None) or trainer.model
    if backend == "mlx":
        import mlx.core as mx
        la = np.array(model(mx.array(a)))
        lb = np.array(model(mx.array(b)))
    else:
        import torch
        with torch.no_grad():
            dev = trainer.device
            la = model(torch.from_numpy(a).to(dev, dtype=torch.long)).float().cpu().numpy()
            lb = model(torch.from_numpy(b).to(dev, dtype=torch.long)).float().cpu().numpy()

    past = np.abs(la[:, : T // 2] - lb[:, : T // 2]).max()
    future = np.abs(la[:, T // 2 :] - lb[:, T // 2 :]).max()
    print(f"  {backend:<6} causal check: max|Δlogit| past={past:.2e} future={future:.2e}")
    if past > 1e-5:
        return f"{backend}: causal mask leaks — future tokens changed past logits by {past:.2e}"
    if future < 1e-6:
        return f"{backend}: changing the future changed nothing — model ignores its input?"
    return None


def main() -> int:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        # `learnable=False` => uniform-random token ids.  That removes the
        # bigram shortcut a structured stream would offer, so the only way to
        # drive the loss down is genuine memorisation of this exact batch.
        dc = make_synthetic_corpus(Path(td) / "tok", vocab=VOCAB, learnable=False,
                                   domains={"general": 1.0}, tokens_per_domain=50_000)
        cfg = tiny_config(dc.tokenized_dir)
        cfg.data.mix = dict(dc.mix)

        corpus = Corpus(cfg.data)
        stream = TrainStream(corpus, BATCH, SEQ, cfg.train.seed)
        # THE fixed batch — the same tokens every single step.
        x, y, _ = stream.batch(0)
        print(f"overfitting one batch of {x.shape} ({x.size} tokens), "
              f"target loss < {TARGET_LOSS} within {MAX_STEPS} steps")

        for backend in ("mlx", "torch"):
            t0 = time.time()
            try:
                final, curve = run_backend(backend, cfg, x, y)
            except Exception as e:
                failures.append(f"{backend}: raised {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                continue
            n = len(curve)
            marks = ", ".join(
                f"step {i}: {curve[i]:.3f}" for i in (0, min(24, n - 1), min(99, n - 1), n - 1)
            )
            print(f"  {backend:<6} {n:>3} steps in {time.time()-t0:5.1f}s  "
                  f"final loss {final:.4f}   [{marks}]")
            if final >= TARGET_LOSS:
                failures.append(
                    f"{backend}: loss {final:.4f} did not reach {TARGET_LOSS} in {MAX_STEPS} steps"
                )
            if curve[0] < 5.0:
                # ln(512) = 6.24; a much lower starting loss means the init or
                # the loss reduction is wrong.
                failures.append(f"{backend}: suspicious initial loss {curve[0]:.4f} (expect ~6.2)")
            leak = check_causal(backend, cfg)
            if leak:
                failures.append(leak)

    for f in failures:
        print(f"FAIL  {f}")
    print(f"{'FAILED' if failures else 'PASS'}  test_overfit ({len(failures)} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
