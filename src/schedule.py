"""WSD (Warmup–Stable–Decay) learning-rate schedule.

WHY this schedule instead of the usual cosine?
--------------------------------------------
A cosine schedule has to know the *total* number of training tokens up front:
the LR at step t depends on t/T_total.  If you later decide to train longer,
you cannot — the cosine has already annealed and re-warming a decayed model
costs you loss.  We are training across ~6–12 overnight sessions with no firm
end date, so we use WSD (a.k.a. "trapezoidal", Hu et al. 2024 / MiniCPM):

    lr
    ^        ______________________________
    |       /                              \\_
    |      /  (stable plateau — resumable)   \\_
    |     /                                    \\__  -> 0.1 * peak
    +----+--------------------------------+------+--> tokens
      warmup                          decay_start  total

  * **warmup**  — linear 0 → peak over `warmup_tokens`.  Early steps have
    badly-scaled Adam second moments and a random-init model; a big LR here
    produces the classic loss spike / divergence.
  * **stable**  — flat at `peak_lr`.  This is the crucial property: the LR is
    the *same* every session, so stopping and resuming is a no-op for the
    optimizer.  You can train for as long as you like and the schedule never
    "runs out".
  * **decay**   — linear peak → 0.1*peak over the last `decay_fraction` of the
    horizon.  Nearly all of the final-loss benefit of an annealed schedule
    shows up in this short phase, so we can *mint* a finished model at any
    moment by running a decay from the current checkpoint (`train.py
    --decay-now`) and still keep the un-decayed checkpoint for further
    pretraining.

Why decay to 0.1*peak rather than exactly 0?  Loss is essentially identical,
and a non-zero floor keeps the model trainable if we choose to continue.

THE KEY INVARIANT: `lr_at` is a *pure function of tokens seen*.  Not of steps,
not of wall-clock, not of any mutable state carried in the checkpoint.  That is
what makes the plateau resumable across sessions and what makes the resume test
(tests/test_resume.py) able to assert bit-comparable LR values.  Tokens (rather
than steps) is the right x-axis because the batch size may change between
sessions after re-calibration — the model only cares how much data it has seen.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `import src.*` work whether this file is run as `python -m src.schedule`
# from the repo root or as `python src/schedule.py`.  Every module of ours does
# this so there is exactly one import style to remember.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, TrainConfig  # noqa: E402


def lr_at(tokens_seen: float, cfg: Config | TrainConfig, decay_start: float | None = None) -> float:
    """Return the learning rate after `tokens_seen` tokens.

    Args:
        tokens_seen: total tokens consumed so far, summed over all sessions.
        cfg: a Config or a TrainConfig (we accept either so callers don't have
            to unwrap; everything we need lives in the [train] section).
        decay_start: optional override for where the decay phase begins, in
            tokens.  `train.py --decay-now` passes the current token count here
            so that the anneal starts immediately instead of at the (possibly
            far-away) nominal end of the run.  The decay *length* stays
            `decay_fraction * total_tokens` so the anneal has the same shape.

    Returns:
        The LR for this step, in the same units as `cfg.train.peak_lr`.
    """
    t = cfg.train if isinstance(cfg, Config) else cfg

    peak = t.peak_lr
    floor = 0.1 * peak
    warmup = float(t.warmup_tokens)
    total = float(t.total_tokens)
    decay_len = t.decay_fraction * total
    start = float(decay_start) if decay_start is not None else total - decay_len

    # Guard: a --decay-now issued while still inside warmup (or a pathological
    # config) must not produce a negative-length or backwards schedule.
    start = max(start, 0.0)

    if tokens_seen < warmup and warmup > 0:
        # Linear warmup.  Deliberately starts *above* zero at the first step
        # (tokens_seen is the count *before* this step) so step 0 still moves.
        return peak * (tokens_seen / warmup)

    if tokens_seen < start:
        return peak  # the stable plateau

    if decay_len <= 0:
        return floor

    frac = (tokens_seen - start) / decay_len
    frac = min(max(frac, 0.0), 1.0)
    return peak + (floor - peak) * frac  # linear peak -> 0.1*peak


def decay_end_tokens(cfg: Config | TrainConfig, decay_start: float) -> float:
    """Token count at which a decay phase started at `decay_start` finishes.

    `train.py --decay-now` uses this to know when to stop and exit.
    """
    t = cfg.train if isinstance(cfg, Config) else cfg
    return decay_start + t.decay_fraction * float(t.total_tokens)


def phase_at(tokens_seen: float, cfg: Config | TrainConfig, decay_start: float | None = None) -> str:
    """Human-readable phase name, for log lines and the dashboard."""
    t = cfg.train if isinstance(cfg, Config) else cfg
    total = float(t.total_tokens)
    start = float(decay_start) if decay_start is not None else total - t.decay_fraction * total
    if tokens_seen < t.warmup_tokens:
        return "warmup"
    if tokens_seen < start:
        return "stable"
    return "decay"


# --------------------------------------------------------------------------
# Self-test: `uv run python -m src.schedule --plot-ascii`
# --------------------------------------------------------------------------

def _plot_ascii(cfg: Config, width: int = 78, height: int = 18, decay_start: float | None = None) -> str:
    total = float(cfg.train.total_tokens)
    # Sample slightly past the end so the tail of the decay is visible.
    xs = [total * 1.02 * i / (width - 1) for i in range(width)]
    ys = [lr_at(x, cfg, decay_start) for x in xs]
    top = max(ys) if max(ys) > 0 else 1.0

    rows = []
    for r in range(height):
        # Row r covers LR band [lo, hi); row 0 is the top of the plot.
        hi = top * (height - r) / height
        lo = top * (height - r - 1) / height
        # Fill everything below the curve so the trapezoid is obvious.
        line = "".join("#" if y >= lo else " " for y in ys)
        rows.append(f"{hi:9.2e} |{line}")
    axis = " " * 10 + "+" + "-" * width
    ticks = " " * 11 + f"0{' ' * (width // 2 - 6)}{total / 2e9:.1f}B{' ' * (width // 2 - 8)}{total / 1e9:.1f}B tokens"
    return "\n".join(rows + [axis, ticks])


def _self_test(cfg: Config) -> int:
    """Assert the schedule's contract.  Returns a process exit code."""
    t = cfg.train
    total, warm = float(t.total_tokens), float(t.warmup_tokens)
    start = total - t.decay_fraction * total
    peak, floor = t.peak_lr, 0.1 * t.peak_lr
    failures = []

    def check(name, got, want, tol=1e-9):
        if abs(got - want) > tol * max(1.0, abs(want)):
            failures.append(f"{name}: got {got!r} want {want!r}")

    check("lr(0) == 0", lr_at(0, cfg), 0.0)
    check("lr(half warmup) == peak/2", lr_at(warm / 2, cfg), peak / 2)
    check("lr(warmup) == peak", lr_at(warm, cfg), peak)
    check("lr(mid plateau) == peak", lr_at((warm + start) / 2, cfg), peak)
    check("lr(just before decay) == peak", lr_at(start - 1, cfg), peak)
    check("lr(decay start) == peak", lr_at(start, cfg), peak)
    check("lr(decay mid) == 0.55*peak", lr_at((start + total) / 2, cfg), (peak + floor) / 2)
    check("lr(total) == floor", lr_at(total, cfg), floor)
    check("lr(beyond total) == floor", lr_at(total * 2, cfg), floor)

    # Monotonicity: never increases after warmup, never decreases during it.
    xs = [total * i / 2000 for i in range(2001)]
    ys = [lr_at(x, cfg) for x in xs]
    for i in range(1, len(xs)):
        if xs[i] <= warm and ys[i] < ys[i - 1] - 1e-12:
            failures.append(f"warmup not monotonic at {xs[i]:.3e}")
            break
        if xs[i - 1] >= warm and ys[i] > ys[i - 1] + 1e-12:
            failures.append(f"post-warmup not non-increasing at {xs[i]:.3e}")
            break

    # Resumability: the value must depend only on tokens, so evaluating at the
    # same token count twice (as a resumed session does) must agree exactly.
    if lr_at(1.234e9, cfg) != lr_at(1.234e9, cfg):
        failures.append("not a pure function of tokens_seen")

    # --decay-now: an anneal begun early must reach the floor after
    # decay_fraction * total more tokens, and never exceed peak.
    ds = 1.0e9
    check("decay-now start == peak", lr_at(ds, cfg, decay_start=ds), peak)
    check("decay-now end == floor", lr_at(decay_end_tokens(cfg, ds), cfg, decay_start=ds), floor)

    for f in failures:
        print(f"FAIL  {f}")
    print(f"{'FAILED' if failures else 'OK'}  schedule self-test ({len(failures)} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    import argparse
    import sys

    from src.config import load_config

    ap = argparse.ArgumentParser(description="WSD learning-rate schedule")
    ap.add_argument("--config", default=None, help="configs/session.toml")
    ap.add_argument("--plot-ascii", action="store_true", help="draw the schedule")
    ap.add_argument("--decay-now-at", type=float, default=None,
                    help="token count at which to start an early (mint) decay")
    args = ap.parse_args()

    cfg = load_config(args.config)
    t = cfg.train
    print(f"peak_lr={t.peak_lr}  warmup={t.warmup_tokens/1e6:.0f}M  "
          f"total={t.total_tokens/1e9:.2f}B  decay_fraction={t.decay_fraction}")
    if args.plot_ascii:
        print(_plot_ascii(cfg, decay_start=args.decay_now_at))
        for frac in (0.0, 0.005, 0.01, 0.02, 0.25, 0.5, 0.9, 0.95, 0.99, 1.0):
            x = t.total_tokens * frac
            print(f"  {x/1e9:7.3f}B tokens  lr={lr_at(x, cfg):.3e}  [{phase_at(x, cfg)}]")
    sys.exit(_self_test(cfg))
