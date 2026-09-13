#!/usr/bin/env python3
"""Write plausible fake run output into runs/demo/ so the dashboard can be
developed and tested without the real trainer.

Emits, in real time, records matching docs/contracts.md:
  metrics.jsonl : "train" every ~0.4s, "val" and "event" occasionally
  samples.jsonl : a vibe generation every ~15 train records
  heartbeat     : touched every 5s

    uv run python scripts/gen_fake_metrics.py --seconds 60 [--run demo] [--fresh]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOMAINS = ["general", "textbooks", "philosophy", "scifi", "psych", "comedy", "haiku"]
TOTAL_TOKENS = 6_000_000_000

FAKE_TEXTS = [
    "not a destination but a practice, repeated on the ordinary mornings when no\n"
    "one is watching and nothing in particular depends on it.",
    "you have already survived every day that once looked impossible from here.",
    "autumn wind —\nthe kettle finds its voice\nbefore I do",
    "a stack trace at 3am /\nthe bug was in the test /\nrain on the window",
    "the hull breach sealed itself, which was the first sign that the ship had\n"
    "opinions of its own.",
    "the ladder, he explained, was for the high notes. The bartender did not laugh,\n"
    "which is how you know it was a good joke.",
    "less a thing we have than a thing we do, moment by moment, out of attention.",
]


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_prompts() -> list[str]:
    p = REPO_ROOT / "prompts" / "vibes.txt"
    try:
        lines = [ln.strip() for ln in p.read_text().splitlines()]
    except OSError:
        return ["The meaning of a good life is"]
    out = [ln for ln in lines if ln and not ln.startswith("#")]
    return out or ["The meaning of a good life is"]


def append(path: Path, obj: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(obj) + "\n")
        fh.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--run", default="demo")
    ap.add_argument("--rate", type=float, default=0.4, help="seconds between train records")
    ap.add_argument("--fresh", action="store_true", help="delete existing demo files first")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    run_dir = REPO_ROOT / "runs" / args.run
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    metrics = run_dir / "metrics.jsonl"
    samples = run_dir / "samples.jsonl"
    heartbeat = run_dir / "heartbeat"
    if args.fresh:
        for f in (metrics, samples):
            f.unlink(missing_ok=True)

    prompts = load_prompts()
    session = 1
    step = 0
    tokens = 0
    tokens_per_step = 64 * 1024  # batch_size * seq_len
    last_hb = 0.0
    n_train = 0

    append(metrics, {"type": "event", "t": now(), "session": session,
                     "kind": "session_start", "detail": f"fake session {session}"})

    t_end = time.time() + args.seconds
    while time.time() < t_end:
        step += rng.randint(20, 30)
        tokens = step * tokens_per_step
        # loss decays 10 -> ~3 over the notional full run, plus noise
        frac = min(1.0, n_train / 400.0)
        loss = 3.0 + 7.0 * math.exp(-3.2 * frac) + rng.gauss(0, 0.06)
        tps = 8200 + rng.gauss(0, 260)
        append(metrics, {
            "type": "train", "t": now(), "session": session, "step": step,
            "tokens": tokens, "loss": round(loss, 4),
            "lr": round(3e-4 * min(1.0, 0.15 + frac), 8),
            "tps": round(tps, 1), "mem_gb": round(41.0 + rng.gauss(0, 0.4), 2),
        })
        n_train += 1

        if n_train % 10 == 0:
            base = loss
            append(metrics, {
                "type": "val", "t": now(), "session": session, "step": step, "tokens": tokens,
                "losses": {d: round(base + off + rng.gauss(0, 0.05), 4)
                           for d, off in zip(DOMAINS, [-0.2, -0.35, 0.15, 0.0, 0.05, 0.3, 0.9])},
            })
        if n_train % 15 == 0:
            append(metrics, {"type": "event", "t": now(), "session": session,
                             "kind": "checkpoint_saved", "detail": f"step {step}"})
            (run_dir / "ckpt" / f"step_{step}").mkdir(exist_ok=True)
            link = run_dir / "ckpt" / "latest"
            link.unlink(missing_ok=True)
            link.symlink_to(f"step_{step}")
            append(samples, {
                "t": now(), "step": step, "tokens": tokens,
                "prompt": rng.choice(prompts), "text": " " + rng.choice(FAKE_TEXTS),
            })
        if n_train % 47 == 0:
            append(metrics, {"type": "event", "t": now(), "session": session,
                             "kind": "nan_rollback", "detail": "loss spike 3.1 -> 9.8, halving lr"})

        if time.time() - last_hb > 5:
            heartbeat.touch()
            last_hb = time.time()
        time.sleep(args.rate)

    append(metrics, {"type": "event", "t": now(), "session": session,
                     "kind": "session_end", "detail": f"step {step}, {tokens/1e9:.3f}B tokens"})
    print(f"wrote {n_train} train records to {metrics} ({tokens/TOTAL_TOKENS:.2%} of target)")


if __name__ == "__main__":
    main()
