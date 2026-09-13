#!/usr/bin/env python3
"""Is an interrupted run identical to an uninterrupted one?

WHY THIS TEST EXISTS
====================
This project trains across ~6–12 overnight sessions, so the model spends more
of its life being *resumed* than being started.  Resume is therefore not a
convenience feature, it is the main code path — and it is exactly the kind of
code that fails silently.  The usual bugs:

  * the optimizer's Adam moments are not saved, so every resume secretly
    restarts the optimizer and puts a small dent in the loss curve;
  * the data stream restarts from the beginning, so the model sees the first
    few hundred million tokens over and over;
  * the token counter or the LR schedule drifts, so the "flat" WSD plateau is
    not flat across sessions;
  * a checkpoint is written non-atomically and a crash leaves a model from
    step N beside an optimizer from step N-1.

None of those show up as an error.  They show up as a model that is quietly
worse than it should be, months later.  So we test the strongest property we
can actually demand:

    training 100 steps straight through, and training 50 + resuming for 50
    more, must produce the SAME loss at every step.

That works because every input to a step is deterministic: the batch is a pure
function of (seed, step) (see data_loader.py), the LR is a pure function of
tokens seen (see schedule.py), and the weights/optimizer state are restored
bit-exactly from fp32 safetensors.  Any real divergence means state was lost.

We run the real `src/train.py` in subprocesses — not an in-process imitation of
the loop — because the point is to test the actual resume path, including
process teardown and the `ckpt/latest` symlink.

Run: `uv run python tests/test_resume.py [--backend mlx|torch|both]`
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import make_synthetic_corpus  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
STEPS_TOTAL = 100
STEPS_FIRST = 50
BATCH, SEQ, VOCAB = 8, 64, 512
TOK_PER_STEP = BATCH * SEQ
# bf16 arithmetic is deterministic for identical inputs and shapes, so we
# expect an exact match; the tolerance only guards against a framework
# reordering a reduction between processes.  (metrics.jsonl rounds the loss to
# 4 decimals, so this comparison is at that resolution — which is why we also
# compare the final *weights* element-wise below, where nothing is rounded.)
LOSS_TOL = 1e-4
WEIGHT_TOL = 0.0  # fp32 master weights restored from safetensors: exact or bust


def write_config(path: Path, tokenized_dir: Path, run_name: str) -> Path:
    path.write_text(f"""
run_name = "{run_name}"

[model]
d_model = 128
n_layers = 2
n_heads = 4
ffn_hidden = 352
vocab_size = {VOCAB}
seq_len = {SEQ}

[train]
batch_size = {BATCH}
peak_lr = 1e-3
warmup_tokens = 5000
total_tokens = 100000000
seed = 4242
checkpoint_every_s = 100000
val_every_tokens = 100000000
sample_every_tokens = 100000000

[data]
tokenized_dir = "{tokenized_dir}"
mix = {{ general = 0.7, haiku = 0.3 }}
""")
    return path


def run_train(cfg: Path, runs_dir: Path, backend: str, max_steps: int, extra=()) -> None:
    cmd = [sys.executable, "-m", "src.train", "--config", str(cfg),
           "--runs-dir", str(runs_dir), "--backend", backend,
           "--max-steps", str(max_steps), "--metrics-every-s", "0", *extra]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"train.py exited {r.returncode}")


def train_records(runs_dir: Path, run_name: str) -> list[dict]:
    out = []
    for line in (runs_dir / run_name / "metrics.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec.get("type") == "train":
            out.append(rec)
    return out


def check_backend(backend: str, workdir: Path) -> list[str]:
    failures: list[str] = []
    tok = workdir / "tok"
    make_synthetic_corpus(tok, vocab=VOCAB, domains={"general": 0.7, "haiku": 0.3},
                          tokens_per_domain=200_000, seed=7)

    # --- A: one uninterrupted run of STEPS_TOTAL steps --------------------
    cfg_a = write_config(workdir / "a.toml", tok, f"ref_{backend}")
    runs_a = workdir / "runs_a"
    run_train(cfg_a, runs_a, backend, STEPS_TOTAL)
    ref = train_records(runs_a, f"ref_{backend}")

    # --- B: STEPS_FIRST steps, process exits, then auto-resume ------------
    cfg_b = write_config(workdir / "b.toml", tok, f"res_{backend}")
    runs_b = workdir / "runs_b"
    run_train(cfg_b, runs_b, backend, STEPS_FIRST)
    ckpt = (runs_b / f"res_{backend}" / "ckpt" / "latest").resolve()
    state = json.loads((ckpt / "state.json").read_text())
    print(f"  [{backend}] checkpoint after kill: {ckpt.name} "
          f"step={state['step']} tokens={state['tokens']} session={state['session']}")
    if state["step"] != STEPS_FIRST or state["tokens"] != STEPS_FIRST * TOK_PER_STEP:
        failures.append(f"{backend}: checkpoint state.json says step={state['step']} "
                        f"tokens={state['tokens']}, expected {STEPS_FIRST}/"
                        f"{STEPS_FIRST * TOK_PER_STEP}")
    run_train(cfg_b, runs_b, backend, STEPS_TOTAL, extra=["--resume"])
    got = train_records(runs_b, f"res_{backend}")

    # --- the checks -------------------------------------------------------
    if len(ref) != STEPS_TOTAL:
        failures.append(f"{backend}: reference run logged {len(ref)} records, expected {STEPS_TOTAL}")
    if len(got) != STEPS_TOTAL:
        failures.append(f"{backend}: resumed run logged {len(got)} records, expected {STEPS_TOTAL}")

    # steps/tokens must line up exactly across the seam — no gap, no repeat
    for i, rec in enumerate(got, start=1):
        if rec["step"] != i:
            failures.append(f"{backend}: record {i} has step={rec['step']} (expected {i})")
            break
        if rec["tokens"] != i * TOK_PER_STEP:
            failures.append(f"{backend}: step {i} tokens={rec['tokens']} "
                            f"(expected {i * TOK_PER_STEP})")
            break
    sessions = sorted({r["session"] for r in got})
    if sessions != [1, 2]:
        failures.append(f"{backend}: expected sessions [1, 2] in the resumed run, got {sessions}")

    # the loss stream itself
    worst, worst_i = 0.0, -1
    for i, (a, b) in enumerate(zip(ref, got), start=1):
        d = abs(a["loss"] - b["loss"])
        if d > worst:
            worst, worst_i = d, i
    seam = next((abs(a["loss"] - b["loss"]) for a, b in
                 zip(ref[STEPS_FIRST:], got[STEPS_FIRST:])), float("nan"))
    print(f"  [{backend}] loss@{STEPS_FIRST}: ref={ref[STEPS_FIRST-1]['loss']:.4f} "
          f"resumed={got[STEPS_FIRST-1]['loss']:.4f} | "
          f"first step after seam Δ={seam:.2e} | "
          f"final: ref={ref[-1]['loss']:.4f} resumed={got[-1]['loss']:.4f} "
          f"Δ={abs(ref[-1]['loss']-got[-1]['loss']):.2e}")
    print(f"  [{backend}] max |Δloss| over {STEPS_TOTAL} steps = {worst:.2e} at step {worst_i}")
    if worst > LOSS_TOL:
        failures.append(f"{backend}: loss stream diverged by {worst:.3e} at step {worst_i} "
                        f"(tolerance {LOSS_TOL:.0e}) — resume lost state")

    # LR must be continuous across the seam too (the WSD plateau is only
    # resumable because lr_at() is a pure function of tokens seen).
    for i, (a, b) in enumerate(zip(ref, got), start=1):
        if abs(a["lr"] - b["lr"]) > 1e-12:
            failures.append(f"{backend}: lr diverged at step {i}: {a['lr']} vs {b['lr']}")
            break

    # The un-rounded version of the same claim: after 100 steps the two runs'
    # fp32 master weights must agree element-for-element.
    from safetensors.numpy import load_file
    wa = load_file(str(runs_a / f"ref_{backend}" / "ckpt" / "latest" / "model.safetensors"))
    wb = load_file(str(runs_b / f"res_{backend}" / "ckpt" / "latest" / "model.safetensors"))
    if set(wa) != set(wb):
        failures.append(f"{backend}: checkpoint tensor names differ")
    else:
        import numpy as np
        worst_w, worst_k = 0.0, ""
        for k in wa:
            d = float(np.abs(wa[k].astype(np.float64) - wb[k].astype(np.float64)).max())
            if d > worst_w:
                worst_w, worst_k = d, k
        print(f"  [{backend}] max |Δweight| across {len(wa)} tensors = {worst_w:.3e} ({worst_k})")
        if worst_w > WEIGHT_TOL:
            failures.append(f"{backend}: final weights differ by {worst_w:.3e} in {worst_k}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="both", choices=["mlx", "torch", "both"])
    ap.add_argument("--keep", action="store_true", help="keep the temp run dirs")
    args = ap.parse_args()
    backends = ["mlx", "torch"] if args.backend == "both" else [args.backend]

    failures: list[str] = []
    for backend in backends:
        td = tempfile.mkdtemp(prefix=f"resume_{backend}_")
        print(f"[{backend}] {STEPS_TOTAL} straight vs {STEPS_FIRST}+resume  ({td})")
        try:
            failures += check_backend(backend, Path(td))
        except Exception as e:
            import traceback
            traceback.print_exc()
            failures.append(f"{backend}: {type(e).__name__}: {e}")
        finally:
            if not args.keep:
                import shutil
                shutil.rmtree(td, ignore_errors=True)

    for f in failures:
        print(f"FAIL  {f}")
    print(f"{'FAILED' if failures else 'PASS'}  test_resume ({len(failures)} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
