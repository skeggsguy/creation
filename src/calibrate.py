#!/usr/bin/env python3
"""Measure, don't guess: which backend and which batch size for this machine?

WHY CALIBRATE AT ALL?
=====================
The plan is a multi-day pretraining run on one Apple M5 Max.  Two decisions set
the total wall-clock cost of that run, and neither is answerable from first
principles:

1. **MLX or PyTorch-MPS?**  MLX is Apple's own array framework with lower
   kernel-launch overhead and native unified-memory semantics, which *should*
   win.  But MLX's bf16 matmul on M5 currently regresses ~1.1–1.2x versus
   PyTorch's MPS backend (mlx#3196).  The two are close enough that the answer
   is an empirical question about this specific machine and these specific
   shapes — and a 20% difference is a full extra night of training.

2. **How big a batch?**  On a 128GB unified-memory machine with no gradient
   accumulation, bigger batches mean fewer, larger kernels, which is what a
   launch-bound device wants — until the working set stops fitting.  In practice
   the curve is a *plateau*, not a peak: the GPU is already saturated by batch
   8, throughput stays flat to batch 32, and then activation memory (~1.9GB per
   batch row at seq 1024) runs the machine into swap, where it silently gets 40x
   slower rather than raising an error.  So the sweep's real job is to find the
   biggest batch that is still on the plateau — bigger batch, same speed, less
   gradient noise.

We also measure a smaller model (d768/L12) as a control: if the two backends
rank differently at the two sizes, the difference is about kernel shapes rather
than framework overhead, and that is worth knowing before committing.

HOW TO READ THE NUMBERS
=======================
* **tok/s** is the number that matters: total_tokens / tok_s = wall clock.
* **MFU** (model FLOPs utilisation) = 6*N*tok_s / peak_flops, where N is the
  parameter count and 6*N is the standard forward+backward FLOPs-per-token
  estimate (2 for the forward's matmuls, 4 for the backward's two).  We divide
  by 60 TFLOP/s, the M5 Max's approximate bf16 peak (a large bf16 matmul
  measures ~60 TFLOP/s on this machine, so the assumption is honest).  We
  measure ~38–42% with MLX and ~23–27% with PyTorch-MPS on the 303M model; a
  *falling* MFU as batch size grows means you have passed the sweet spot.
* **thermal drift** (`--soak`): a laptop sustains less than it bursts.  The
  honest planning number is the tok/s after 15+ minutes, not after 90 seconds.
  Measured here: a ~10% drop from the cold burst into the steady state, and
  then flat — the machine holds its clocks once warm.

EVERY MEASUREMENT RUNS IN A SUBPROCESS
======================================
One config per process, so (a) an out-of-memory config kills only its own
worker and the sweep continues, and (b) no measurement inherits a warm
allocator, a fragmented heap, or leftover state from the previous one.  That
costs a few seconds of startup per config and buys numbers you can trust.

USAGE
    uv run python -m src.calibrate --quick          # ~2 min sanity check
    uv run python -m src.calibrate                  # full sweep (~20-30 min)
    uv run python -m src.calibrate --soak 1800      # thermal soak of the winner
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, ModelConfig, RUNS_DIR, load_config, n_params  # noqa: E402
from src.data_loader import Corpus, TrainStream, make_synthetic_corpus  # noqa: E402

PEAK_FLOPS = 60e12          # M5 Max bf16 peak, per the hardware notes
OUT_DIR = RUNS_DIR / "calibration"
# Small batches are in the sweep deliberately.  Activation memory for a 303M
# model at seq 1024 runs ~1.9GB *per batch row*, so batch 64 needs ~130GB and
# the machine starts swapping instead of failing — a 40x slowdown that looks
# like a working run.  The interesting part of the curve is at the low end.
BATCH_SWEEP = (8, 16, 32, 64, 128)
WARMUP_STEPS = 3            # first steps pay for compilation and allocator growth
SOAK_LOG_EVERY_S = 30.0
# Unified memory means an over-large batch does not raise OOM on MLX, it silently
# swaps.  We stop a sweep before that: 96GB of 128GB leaves room for the OS, the
# page cache holding the corpus, and whatever else is running.
MEM_CEILING_GB = 96.0


# --------------------------------------------------------------------------
# Model sizes under test
# --------------------------------------------------------------------------


def estimate_peak_gb(m: ModelConfig, batch: int) -> float:
    """Predict peak memory so we can skip hopeless configs instead of swapping.

    Where the memory actually goes, per token of the batch:

      * **activations kept for the backward pass** — every layer stashes roughly
        8 tensors of width d_model (block input, both norm outputs, q/k/v, the
        attention output, the projection output) and 3 of width ffn_hidden
        (the two SwiGLU branches and their product), all bf16 (2 bytes);
      * **the logits** — width vocab_size, and about three copies of it are
        live at once (the logits, the softmax, its gradient).

    Add the fixed cost of fp32 weights + gradients + Adam's two moments (4
    copies of the parameters), multiply by a fudge factor for allocator
    transients, and the prediction lands within a few percent of measured peak.
    The formula is worth internalising: activation memory is *linear in batch x
    seq x n_layers* and independent of how big your weights are, which is why a
    303M model with 1.2GB of weights can still need 130GB at batch 64.
    """
    per_token = 2 * (m.n_layers * (8 * m.d_model + 3 * m.ffn_hidden) + 3 * m.vocab_size)
    acts = per_token * batch * m.seq_len * 2.0
    state = n_params(m) * 4 * 4
    return (acts + state) / (1 << 30)


def size_variants(cfg: Config) -> dict[str, ModelConfig]:
    """The default (303M) model, plus a smaller control.

    d768/L12 is the GPT-2-small shape; at our 24576-token vocab it comes out
    near ~104M parameters rather than GPT-2's 124M, because we are not spending
    38M parameters on a 50k vocab (decisions.md).
    """
    return {
        "small_d768_L12": replace(cfg.model, d_model=768, n_layers=12, n_heads=12, ffn_hidden=2048),
        "default_d1024_L22": cfg.model,
    }


# --------------------------------------------------------------------------
# Worker: one measurement, one process
# --------------------------------------------------------------------------


def run_worker(spec: dict) -> dict:
    """Time `spec['seconds']` worth of training steps.  Runs in its own process."""
    from src.train import make_trainer

    cfg = Config()
    cfg.model = ModelConfig(**spec["model"])
    cfg.train.batch_size = spec["batch_size"]
    cfg.train.backend = spec["backend"]
    cfg.data.tokenized_dir = spec["data_dir"]
    cfg.data.mix = {"general": 1.0}

    corpus = Corpus(cfg.data, require_val=False)
    stream = TrainStream(corpus, cfg.train.batch_size, cfg.model.seq_len, cfg.train.seed)

    t_build = time.time()
    trainer = make_trainer(cfg, spec["backend"], compile_step=spec["compile"])
    trainer.set_lr(1e-4)
    build_s = time.time() - t_build

    # Warmup: the first steps include kernel compilation (mx.compile /
    # torch.compile) and the allocator growing to its steady-state size.
    # Including them would understate throughput by a lot at short durations.
    t_warm = time.time()
    for s in range(WARMUP_STEPS):
        x, y, _ = stream.batch(s)
        loss, _ = trainer.train_step(x, y)
        # Bail out before spending 90s measuring a config that is swapping.
        if trainer.mem_gb() > MEM_CEILING_GB:
            return {"ok": False, "oom": True, "backend": spec["backend"],
                    "size": spec["size"], "batch_size": spec["batch_size"],
                    "mem_gb": trainer.mem_gb(),
                    "error": f"peak memory {trainer.mem_gb():.1f}GB over the "
                             f"{MEM_CEILING_GB:.0f}GB ceiling"}
    warm_s = time.time() - t_warm

    step_times: list[float] = []
    tokens = 0
    t0 = time.time()
    step = WARMUP_STEPS
    while time.time() - t0 < spec["seconds"]:
        x, y, _ = stream.batch(step)
        ts = time.time()
        loss, _ = trainer.train_step(x, y)
        step_times.append(time.time() - ts)
        tokens += x.size
        step += 1
    elapsed = time.time() - t0

    tok_s = tokens / elapsed
    return {
        "ok": True,
        "backend": spec["backend"],
        "size": spec["size"],
        "batch_size": spec["batch_size"],
        "params": trainer.n_params,
        "compiled": bool(getattr(trainer, "_compiled", False)),
        "steps": len(step_times),
        "tok_s": tok_s,
        "step_ms": 1000 * statistics.median(step_times),
        "step_ms_min": 1000 * min(step_times),
        "mem_gb": trainer.mem_gb(),
        "mfu": 6 * trainer.n_params * tok_s / PEAK_FLOPS,
        "build_s": build_s,
        "warmup_s": warm_s,
        "final_loss": float(loss),
    }


def measure(spec: dict, verbose: bool = True) -> dict:
    """Spawn a worker for one config and parse its result."""
    label = f"{spec['backend']:<5} {spec['size']:<18} batch={spec['batch_size']:<4}"
    if verbose:
        print(f"  {label} ... ", end="", flush=True)
    cmd = [sys.executable, "-m", "src.calibrate", "--worker", json.dumps(spec)]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]),
                       capture_output=True, text=True)
    if r.returncode != 0:
        # Anything from a Metal OOM to a framework assertion lands here.  We
        # record it and keep sweeping — that is the whole point of isolating
        # each measurement in its own process.
        tail = (r.stderr or r.stdout).strip().splitlines()
        err = tail[-1][:200] if tail else f"exit {r.returncode}"
        oom = any(k in (r.stderr + r.stdout).lower()
                  for k in ("out of memory", "oom", "insufficient", "alloc"))
        if verbose:
            print(f"{'OOM' if oom else 'FAILED'} ({err})")
        return {"ok": False, "oom": oom, "error": err, "backend": spec["backend"],
                "size": spec["size"], "batch_size": spec["batch_size"],
                "wall_s": time.time() - t0}
    try:
        res = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        if verbose:
            print("FAILED (unparseable worker output)")
            print(r.stdout[-1000:])
        return {"ok": False, "oom": False, "error": "unparseable worker output",
                "backend": spec["backend"], "size": spec["size"],
                "batch_size": spec["batch_size"]}
    res["wall_s"] = time.time() - t0
    if not res.get("ok"):
        # The worker ran fine but refused the config (memory ceiling).
        if verbose:
            print(f"SKIPPED ({res.get('error', 'unknown')})")
        return res
    if verbose:
        print(f"{res['tok_s']:9.1f} tok/s  {res['step_ms']:7.1f} ms/step  "
              f"{res['mem_gb']:6.2f} GB  MFU {res['mfu']*100:4.1f}%  "
              f"({res['steps']} steps, compiled={res['compiled']})")
    return res


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------


def sweep(cfg: Config, data_dir: Path, backends, sizes: dict[str, ModelConfig],
          batches, seconds: float, compile_step: bool, cooldown: float = 20.0) -> list[dict]:
    """Sweep every (backend, size, batch) combination.

    CAVEAT THIS FUNCTION EXISTS TO MANAGE: measurements are sequential, and a
    laptop that has been at full tilt for ten minutes is slower than one that
    just woke up.  Without care, whichever config we measure *first* wins by a
    few percent for purely thermal reasons.  Two mitigations: an idle cooldown
    between configs, and a regression threshold (5%) wider than the drift we
    expect.  For the number you will actually plan around, use `--soak`.
    """
    results: list[dict] = []
    first = True
    for backend in backends:
        for size_name, mcfg in sizes.items():
            best = 0.0
            for bs in batches:
                est = estimate_peak_gb(mcfg, bs)
                if est > MEM_CEILING_GB:
                    print(f"  {backend:<5} {size_name:<18} batch={bs:<4} ... SKIPPED "
                          f"(predicted {est:.0f}GB > {MEM_CEILING_GB:.0f}GB ceiling)")
                    results.append({"ok": False, "oom": True, "backend": backend,
                                    "size": size_name, "batch_size": bs,
                                    "error": f"predicted {est:.0f}GB peak, over ceiling"})
                    break
                spec = {
                    "backend": backend, "size": size_name, "batch_size": bs,
                    "seconds": seconds, "compile": compile_step,
                    "data_dir": str(data_dir),
                    "model": {k: getattr(mcfg, k) for k in
                              ("d_model", "n_layers", "n_heads", "ffn_hidden",
                               "vocab_size", "seq_len", "rope_theta", "norm_eps",
                               "tied_embeddings")},
                }
                if cooldown and not first:
                    time.sleep(cooldown)   # let the SoC settle between configs
                first = False
                res = measure(spec)
                results.append(res)
                if not res.get("ok"):
                    print(f"    -> stopping this sweep at batch {bs} "
                          f"({'out of memory' if res.get('oom') else 'failure'})")
                    break
                if res["mem_gb"] > MEM_CEILING_GB * 0.75:
                    # The next batch size would roughly double this; stop while
                    # the numbers are still trustworthy.
                    print(f"    -> {res['mem_gb']:.0f}GB is close to the "
                          f"{MEM_CEILING_GB:.0f}GB ceiling; stopping this sweep")
                    best = max(best, res["tok_s"])
                    break
                # Throughput regression: we are past the sweet spot, and larger
                # batches will only be worse (and riskier for memory).  The 5%
                # band keeps thermal drift from ending the sweep early.
                if best and res["tok_s"] < best * 0.95:
                    print(f"    -> throughput regressed ({res['tok_s']:.0f} < "
                          f"{best:.0f} tok/s); stopping this sweep")
                    break
                best = max(best, res["tok_s"])
    return results


TIE_BAND = 0.03   # throughputs within 3% of the best count as a tie


def pick_winner(results: list[dict], size_filter: str | None = None) -> dict | None:
    """Fastest config — but among near-ties, the one with the biggest batch.

    Throughput is flat across a wide range of batch sizes on this machine (the
    GPU saturates by batch 8), so the top of the curve is a plateau, not a peak,
    and the differences between plateau entries are within measurement noise.
    Batch size is not neutral, though: a bigger batch means less gradient noise
    per step, which is free quality.  So when two configs are within 3% of each
    other we take the larger batch, and we only pay for throughput we can
    actually distinguish.
    """
    ok = [r for r in results if r.get("ok") and (size_filter is None or r["size"] == size_filter)]
    if not ok:
        return None
    best = max(r["tok_s"] for r in ok)
    tied = [r for r in ok if r["tok_s"] >= best * (1 - TIE_BAND)]
    return max(tied, key=lambda r: r["batch_size"])


# --------------------------------------------------------------------------
# Soak (thermal drift)
# --------------------------------------------------------------------------


def soak(cfg: Config, data_dir: Path, winner: dict, seconds: float) -> dict:
    """Run one config continuously and watch throughput decay.

    A 90-second burst measures the machine cold.  Over a multi-hour session the
    SoC heats up, clocks drop, and sustained throughput settles somewhere
    lower.  The ratio between the two is the number to multiply your ETA by.
    """
    from src.train import make_trainer

    c = Config()
    c.model = ModelConfig(**winner["model"])
    c.train.batch_size = winner["batch_size"]
    c.data.tokenized_dir = str(data_dir)
    c.data.mix = {"general": 1.0}

    corpus = Corpus(c.data, require_val=False)
    stream = TrainStream(corpus, c.train.batch_size, c.model.seq_len, c.train.seed)
    trainer = make_trainer(c, winner["backend"], compile_step=True)
    trainer.set_lr(1e-4)

    for s in range(WARMUP_STEPS):
        x, y, _ = stream.batch(s)
        trainer.train_step(x, y)

    print(f"soaking {winner['backend']} {winner['size']} batch={winner['batch_size']} "
          f"for {seconds/60:.1f} min — logging tok/s every {SOAK_LOG_EVERY_S:.0f}s")
    samples: list[dict] = []
    t0 = time.time()
    win_tokens, t_win = 0, time.time()
    step = WARMUP_STEPS
    while time.time() - t0 < seconds:
        x, y, _ = stream.batch(step)
        trainer.train_step(x, y)
        win_tokens += x.size
        step += 1
        now = time.time()
        if now - t_win >= SOAK_LOG_EVERY_S:
            tps = win_tokens / (now - t_win)
            rec = {"t_s": round(now - t0, 1), "tok_s": round(tps, 1),
                   "mem_gb": round(trainer.mem_gb(), 2),
                   "mfu": round(6 * trainer.n_params * tps / PEAK_FLOPS, 4)}
            samples.append(rec)
            print(f"  +{rec['t_s']:7.1f}s  {rec['tok_s']:9.1f} tok/s  "
                  f"MFU {rec['mfu']*100:4.1f}%  mem {rec['mem_gb']:.2f} GB")
            win_tokens, t_win = 0, now

    if not samples:
        return {"samples": []}
    first, last = samples[0]["tok_s"], samples[-1]["tok_s"]
    out = {"config": {k: winner[k] for k in ("backend", "size", "batch_size")},
           "seconds": seconds, "samples": samples,
           "first_tok_s": first, "last_tok_s": last,
           "sustained_tok_s": statistics.median(s["tok_s"] for s in samples[len(samples) // 2:]),
           "drift_pct": round(100 * (last - first) / first, 2)}
    print(f"thermal drift: {out['drift_pct']:+.1f}% "
          f"({first:.0f} -> {last:.0f} tok/s), sustained {out['sustained_tok_s']:.0f} tok/s")
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def recommended_block(cfg: Config, winner: dict, sustained_tok_s: float | None = None) -> str:
    tok_s = sustained_tok_s or winner["tok_s"]
    hours = cfg.train.total_tokens / tok_s / 3600
    return "\n".join([
        "# --- recommended [train] block for configs/session.toml ---",
        "# (printed, not written — review before pasting)",
        f"# measured {winner['tok_s']:.0f} tok/s burst"
        + (f", {sustained_tok_s:.0f} tok/s sustained" if sustained_tok_s else "")
        + f" at MFU {winner['mfu']*100:.1f}%",
        f"# {cfg.train.total_tokens/1e9:.1f}B tokens => {hours:.1f} h "
        f"(~{hours/8:.1f} overnight sessions of 8h)",
        "[train]",
        f'backend = "{winner["backend"]}"',
        f"batch_size = {winner['batch_size']}",
    ])


def report_paths(mode: str) -> tuple[Path, Path]:
    """Where a run of this mode writes its report.

    `--quick` gets its own filenames so a 2-minute sanity check can never
    overwrite the 20-minute sweep you actually plan the run from.
    """
    stem = "report_quick" if mode == "quick" else "report"
    return OUT_DIR / f"{stem}.md", OUT_DIR / f"{stem}.json"


def write_report(cfg: Config, results: list[dict], winner: dict | None,
                 soak_result: dict | None, mode: str) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path, json_path = report_paths(mode)
    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": mode,
        "host": {"machine": platform.machine(), "platform": platform.platform(),
                 "python": platform.python_version()},
        "peak_flops": PEAK_FLOPS,
        "results": results,
        "winner": winner,
        "soak": soak_result,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Calibration report",
        "",
        f"- generated: {payload['generated']}  (mode: `{mode}`)",
        f"- host: {platform.platform()} / {platform.machine()}",
        f"- assumed peak: {PEAK_FLOPS/1e12:.0f} TFLOP/s bf16 "
        "(MFU = 6*N*tok_s / peak)",
        "",
        "> Measurements are sequential, so later rows run on a warmer machine. "
        "A cooldown between configs and a 5% regression band absorb most of "
        "that, but differences under ~5% between rows are not significant — "
        "use `--soak` for the number to plan around.",
        "",
    ]
    by_size: dict[str, list[dict]] = {}
    for r in results:
        by_size.setdefault(r["size"], []).append(r)
    for size, rows in by_size.items():
        params = next((r["params"] for r in rows if r.get("ok")), None)
        lines += [f"## {size}" + (f" — {params/1e6:.1f}M params" if params else ""), "",
                  "| backend | batch | tok/s | ms/step | mem GB | MFU | steps | note |",
                  "|---|---:|---:|---:|---:|---:|---:|---|"]
        for r in rows:
            if r.get("ok"):
                lines.append(
                    f"| {r['backend']} | {r['batch_size']} | {r['tok_s']:.0f} | "
                    f"{r['step_ms']:.0f} | {r['mem_gb']:.2f} | {r['mfu']*100:.1f}% | "
                    f"{r['steps']} | compiled={r['compiled']} |")
            else:
                lines.append(
                    f"| {r['backend']} | {r['batch_size']} | — | — | — | — | — | "
                    f"{'OOM' if r.get('oom') else 'failed'}: {r.get('error','')[:80]} |")
        lines.append("")

    if winner:
        hours = cfg.train.total_tokens / winner["tok_s"] / 3600
        lines += [
            "## Winner",
            "",
            f"**{winner['backend']} · {winner['size']} · batch {winner['batch_size']}** — "
            f"{winner['tok_s']:.0f} tok/s, {winner['step_ms']:.0f} ms/step, "
            f"{winner['mem_gb']:.2f} GB, MFU {winner['mfu']*100:.1f}%.",
            "",
            f"At that rate {cfg.train.total_tokens/1e9:.1f}B tokens takes "
            f"**{hours:.1f} h** of GPU time.",
            "",
        ]
    if soak_result and soak_result.get("samples"):
        lines += ["## Thermal soak", "",
                  f"- config: {soak_result['config']}",
                  f"- duration: {soak_result['seconds']/60:.1f} min",
                  f"- first sample: {soak_result['first_tok_s']:.0f} tok/s",
                  f"- last sample: {soak_result['last_tok_s']:.0f} tok/s",
                  f"- **sustained (median of 2nd half): "
                  f"{soak_result['sustained_tok_s']:.0f} tok/s**",
                  f"- drift: {soak_result['drift_pct']:+.1f}%", "",
                  "| t (s) | tok/s | MFU | mem GB |", "|---:|---:|---:|---:|"]
        for s in soak_result["samples"]:
            lines.append(f"| {s['t_s']:.0f} | {s['tok_s']:.0f} | "
                         f"{s['mfu']*100:.1f}% | {s['mem_gb']:.2f} |")
        lines.append("")
    if winner:
        lines += ["## Recommended config", "", "```toml",
                  recommended_block(cfg, winner,
                                    soak_result.get("sustained_tok_s") if soak_result else None),
                  "```", ""]

    md_path.write_text("\n".join(lines))
    return md_path, json_path


# --------------------------------------------------------------------------


def ensure_synthetic_data(vocab: int, seq_len: int, max_batch: int) -> Path:
    """A throwaway corpus big enough that batches never overlap-by-accident.

    Calibration measures *speed*, so the token content is irrelevant — uniform
    random ids are fine and generate instantly.  What matters is that the file
    is comfortably larger than one batch so the memmap reads look like the real
    thing.
    """
    d = OUT_DIR / "synthetic"
    need = max(8_000_000, max_batch * (seq_len + 1) * 8)
    marker = d / "meta.json"
    if marker.exists():
        try:
            if json.loads(marker.read_text())["domains"]["general"]["train_tokens"] >= need:
                return d
        except Exception:
            pass
    print(f"generating synthetic calibration corpus ({need/1e6:.1f}M tokens) in {d}")
    make_synthetic_corpus(d, vocab=vocab, domains={"general": 1.0},
                          tokens_per_domain=need, val_tokens=seq_len * 64,
                          shards_per_domain=1, learnable=False)
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description="benchmark backends and batch sizes")
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="~2 min sanity check: default model, config batch size, both backends")
    ap.add_argument("--soak", type=float, default=None,
                    help="after (or instead of) the sweep, run the winner for N seconds")
    ap.add_argument("--seconds", type=float, default=90.0, help="measured seconds per config")
    ap.add_argument("--cooldown", type=float, default=20.0,
                    help="idle seconds between configs, so thermal drift does not "
                         "favour whichever config runs first")
    ap.add_argument("--backends", default="mlx,torch")
    ap.add_argument("--batches", default=",".join(str(b) for b in BATCH_SWEEP))
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="re-render report.md/json from the stored results, no measuring")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    # --- worker mode: one measurement, JSON on stdout ---------------------
    if args.worker:
        print(json.dumps(run_worker(json.loads(args.worker))))
        return 0

    cfg: Config = load_config(args.config)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    variants = size_variants(cfg)

    if args.quick:
        # ~2 minutes: the real model, both backends, one small batch plus
        # whatever the config asks for (so a config that cannot fit says so).
        seconds = 12.0
        sizes = {"default_d1024_L22": cfg.model}
        batches = sorted({8, cfg.train.batch_size})
        mode = "quick"
    else:
        seconds = args.seconds
        sizes = variants
        mode = "full"

    data_dir = (OUT_DIR / "synthetic" if args.report_only else
                ensure_synthetic_data(cfg.model.vocab_size, cfg.model.seq_len, max(batches)))

    results: list[dict] = []
    prev_soak = None
    if args.report_only:
        # Re-render report.md from the stored measurements — useful after
        # changing the peak-FLOPs assumption or the winner-picking rule, and it
        # costs no GPU time.
        prev = report_paths("full")[1]
        if not prev.exists():
            print(f"--report-only: no {prev} to re-render")
            return 1
        payload = json.loads(prev.read_text())
        results, prev_soak, mode = payload.get("results", []), payload.get("soak"), payload.get("mode", "full")
        print(f"re-rendering report from {len(results)} stored results")
    elif args.soak and not args.quick and args.seconds <= 0:
        # Soak-only run: keep the previous sweep's tables instead of replacing
        # the report with an empty one.
        mode = "soak"
        prev = report_paths("full")[1]
        if prev.exists():
            results = json.loads(prev.read_text()).get("results", [])
            print(f"(soak only — carrying forward {len(results)} results from the last sweep)")
    else:
        print(f"\n=== {mode} sweep: backends={backends} sizes={list(sizes)} "
              f"batches={batches} @ {seconds:.0f}s each ===")
        t0 = time.time()
        results = sweep(cfg, data_dir, backends, sizes, batches, seconds,
                        compile_step=not args.no_compile,
                        cooldown=0.0 if args.quick else args.cooldown)
        print(f"=== sweep done in {(time.time()-t0)/60:.1f} min ===\n")

    winner = pick_winner(results, size_filter="default_d1024_L22") or pick_winner(results)
    soak_result = prev_soak
    if args.soak:
        if winner is None:
            prev = report_paths("full")[1]
            if prev.exists():
                winner = json.loads(prev.read_text()).get("winner")
        if winner is None:
            print("no winner to soak (run a sweep first)")
        elif winner["size"] not in variants:
            print(f"cannot soak: the saved winner's size {winner['size']!r} is not one "
                  f"of {list(variants)} (config changed since the sweep?)")
            winner = None
        else:
            winner = dict(winner)
            winner["model"] = {k: getattr(variants[winner["size"]], k) for k in
                               ("d_model", "n_layers", "n_heads", "ffn_hidden",
                                "vocab_size", "seq_len", "rope_theta", "norm_eps",
                                "tied_embeddings")}
            soak_result = soak(cfg, data_dir, winner, args.soak)

    md, js = write_report(cfg, results, winner, soak_result, mode)
    print(f"\nwrote {md}\nwrote {js}\n")
    if winner:
        print(recommended_block(cfg, winner,
                                soak_result.get("sustained_tok_s") if soak_result else None))
        print()
        ref = n_params(cfg.model)
        if winner.get("params") and abs(winner["params"] - ref) / ref > 0.01 \
                and winner["size"] == "default_d1024_L22":
            print(f"WARNING: measured param count {winner['params']/1e6:.1f}M != "
                  f"config {ref/1e6:.1f}M")
        return 0
    print("FAILED: no configuration completed successfully")
    return 1


if __name__ == "__main__":
    sys.exit(main())
