"""Benchmark the finished pretrain against published models — export, verify, evaluate.

    # smoke (proves the plumbing in ~1 minute)
    uv run --with accelerate python src/eval/run_lm_eval.py \
        --ckpt runs/run01/ckpt/step_366211 --tasks hellaswag --limit 30

    # the real thing (an hour-plus)
    uv run --with accelerate python src/eval/run_lm_eval.py \
        --ckpt runs/run01/ckpt/step_366211

`--with accelerate` is not optional: lm-eval's HF wrapper imports `accelerate`
unconditionally and it is not a project dependency.  The uv overlay supplies it
for the duration of the command without touching the project venv.  Stages 1 and
2 (export + verification) need no such thing and run under plain `uv run`.

Three stages, in order, because each one is worthless without the one before it:

1. **Export** the MLX checkpoint to a HuggingFace `LlamaForCausalLM` directory
   (`src/eval/export_hf.py` — read its header for the weight mapping and the
   RoPE argument).  lm-eval speaks HF, and our architecture is LLaMA's.

2. **Verify** the export.  A mis-mapped weight or a wrong RoPE pairing produces
   a model that still runs, still emits English-ish text, and still scores
   *something* on hellaswag — just a wrong something.  So before spending an
   hour on benchmarks we push identical tokens through both implementations and
   demand they agree: logits within tolerance and next-token argmax agreeing on
   at least 95% of positions.  If that fails the run stops here.  The check runs
   the MLX side twice, once in fp32 (which isolates the *mapping* — any real
   difference is a bug) and once in the bf16 the model actually trains and
   samples in (which shows the honest precision gap).

3. **Evaluate** with lm-eval and write a markdown report, including the
   published anchor scores from `src/eval/anchors.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval import anchors as ANCH  # noqa: E402
from src.eval._common import (  # noqa: E402
    REPO_ROOT,
    ckpt_state,
    load_ckpt_config,
    load_tokenizer,
    read_prompts,
    resolve_ckpt,
    run_dir_for,
)
from src.eval.export_hf import export  # noqa: E402

DEFAULT_TASKS = "hellaswag,arc_easy,piqa,lambada_openai,winogrande,sciq"

VERIFY_PROMPTS = [
    "The meaning of a good life is",
    "In the year 3000, humanity finally discovered that the old machines",
    "Cognitive behavioural therapy teaches that our thoughts, feelings and behaviour",
]

ARGMAX_AGREEMENT_FLOOR = 0.95
# fp32-on-CPU vs fp32: observed 8e-5 on the real checkpoint, so 1e-3 is a wide
# moat around "correct" and still catches any real mapping error, which shows up
# as whole logits of difference, not fractions of a millionth.
LOGIT_DIFF_CEILING_FP32 = 1e-3


# --------------------------------------------------------------------------
# Stage 2 — verification
# --------------------------------------------------------------------------


def _mlx_logits(ckpt: Path, token_rows: list[list[int]], dtype: str, device: str) -> list:
    """Run the MLX model over each row, returning fp32 numpy logits (T, vocab).

    Two knobs, and both matter:

    `dtype="float32"` overrides `src.model.COMPUTE_DTYPE` so the forward pass
    runs in fp32 instead of the bf16 the model trains and samples in.  That
    separates "the weights are in the wrong place" from "bf16 rounds
    differently", which are very different bugs with very different fixes.

    `device="cpu"` pins the ops to MLX's CPU stream, and this one is not
    optional for the reference comparison.  **MLX's Metal fp32 matmul is not
    IEEE fp32**: on a single 64x1024 @ 1024x1024 product it lands ~7.6e-4 off a
    float64 reference, where torch (and MLX's own CPU backend) land at ~1.1e-6.
    Run the check on the GPU and the 22-layer residual stream carries that
    ~2.4e-3 *relative* error all the way to the logits, which — on a model whose
    activations peak above 1000 — is a whole logit of absolute difference.  That
    is indistinguishable at a glance from a genuinely broken export, and it is
    the thing that will make you rewrite a perfectly correct RoPE permutation.
    On the CPU stream the same comparison closes to ~8e-5.
    """
    import mlx.core as mx
    import numpy as np

    import src.model as M
    from src.eval._common import load_mlx_model

    prev = M.COMPUTE_DTYPE
    M.COMPUTE_DTYPE = mx.float32 if dtype == "float32" else mx.bfloat16
    stream = mx.cpu if device == "cpu" else mx.gpu
    try:
        with mx.stream(stream):
            model, _ = load_mlx_model(ckpt)
            out = []
            for row in token_rows:
                logits = model(mx.array([row], dtype=mx.int32))
                mx.eval(logits)
                out.append(np.array(logits[0], dtype=np.float32))
        return out
    finally:
        M.COMPUTE_DTYPE = prev


def _hf_logits(hf_dir: Path, token_rows: list[list[int]], device: str) -> list:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(hf_dir), dtype=torch.float32)
    model = model.to(device).eval()
    out = []
    with torch.no_grad():
        for row in token_rows:
            ids = torch.tensor([row], dtype=torch.long, device=device)
            logits = model(ids).logits[0].float().cpu().numpy()
            out.append(logits)
    del model
    return out


def _sample_tokens(tokenizer, n: int = 200) -> list[int]:
    """A ~n-token slice of real in-distribution text for the argmax check.

    Prefer the held-out validation shard (what the model was scored on during
    training); fall back to concatenating the vibe prompts if the corpus is not
    on this machine.
    """
    import numpy as np

    val = REPO_ROOT / "data" / "tokenized" / "general_val.bin"
    if val.exists():
        arr = np.memmap(val, dtype=np.uint16, mode="r")
        return [int(x) for x in arr[1000:1000 + n]]
    text = " ".join(read_prompts(REPO_ROOT / "prompts" / "vibes.txt"))
    ids = tokenizer.encode(text).ids
    return ids[:n]


def verify(ckpt: Path, hf_dir: Path, tokenizer, device: str = "cpu") -> dict:
    """Compare MLX and exported-HF logits.  Returns a report dict; raises on failure."""
    import numpy as np

    prompt_rows = [tokenizer.encode(p).ids for p in VERIFY_PROMPTS]
    sample_row = _sample_tokens(tokenizer, 200)
    rows = prompt_rows + [sample_row]

    print(f"[verify] 3 prompts + a {len(sample_row)}-token corpus sample, "
          f"HF on {device}/float32")
    t0 = time.time()
    hf = _hf_logits(hf_dir, rows, device)
    # fp32 on MLX's CPU stream is the only apples-to-apples reference — see
    # `_mlx_logits` for why the GPU's fp32 matmul is not one.
    mlx_f32 = _mlx_logits(ckpt, rows, "float32", "cpu")
    # ...and this is how the model really runs: bf16 compute on the GPU.
    mlx_bf16 = _mlx_logits(ckpt, rows, "bfloat16", "gpu")
    print(f"[verify] forward passes done in {time.time()-t0:.1f}s")

    per_prompt = []
    for i, p in enumerate(VERIFY_PROMPTS):
        a, b, c = mlx_f32[i], hf[i], mlx_bf16[i]
        per_prompt.append({
            "prompt": p,
            "n_tokens": len(prompt_rows[i]),
            "max_abs_diff_fp32": float(np.abs(a - b).max()),
            "mean_abs_diff_fp32": float(np.abs(a - b).mean()),
            "max_abs_diff_bf16_path": float(np.abs(c - b).max()),
            "last_token_max_abs_diff_fp32": float(np.abs(a[-1] - b[-1]).max()),
            "argmax_agree": bool((a.argmax(-1) == b.argmax(-1)).all()),
        })

    a, b, c = mlx_f32[-1], hf[-1], mlx_bf16[-1]
    agree_f32 = float((a.argmax(-1) == b.argmax(-1)).mean())
    agree_bf16 = float((c.argmax(-1) == b.argmax(-1)).mean())
    sample = {
        "n_positions": int(a.shape[0]),
        "max_abs_diff_fp32": float(np.abs(a - b).max()),
        "mean_abs_diff_fp32": float(np.abs(a - b).mean()),
        "max_abs_diff_bf16_path": float(np.abs(c - b).max()),
        "argmax_agreement_fp32": agree_f32,
        "argmax_agreement_bf16_path": agree_bf16,
        # A logit-space diff is only meaningful next to the spread of the logits.
        "logit_std": float(b.std()),
    }

    report = {"device": device, "prompts": per_prompt, "sample": sample}

    worst_fp32 = max([p["max_abs_diff_fp32"] for p in per_prompt]
                     + [sample["max_abs_diff_fp32"]])
    report["worst_max_abs_diff_fp32"] = worst_fp32

    print(f"[verify] MLX fp32/cpu vs HF fp32: max|Δlogit| {worst_fp32:.3e}, "
          f"argmax agreement {agree_f32*100:.1f}% over {sample['n_positions']} positions")
    print(f"[verify] MLX bf16/gpu (the real inference path) vs HF fp32: "
          f"max|Δlogit| {sample['max_abs_diff_bf16_path']:.3e}, "
          f"argmax agreement {agree_bf16*100:.1f}%")

    ok = agree_f32 >= ARGMAX_AGREEMENT_FLOOR and worst_fp32 <= LOGIT_DIFF_CEILING_FP32
    report["passed"] = ok
    if not ok:
        raise SystemExit(
            "EXPORT VERIFICATION FAILED — refusing to benchmark a model that does not "
            f"match the checkpoint.\n  argmax agreement {agree_f32*100:.1f}% "
            f"(need >= {ARGMAX_AGREEMENT_FLOOR*100:.0f}%)\n"
            f"  max |Δlogit| {worst_fp32:.4g} (need <= {LOGIT_DIFF_CEILING_FP32})\n"
            "  Suspect the weight mapping or the RoPE pairing in src/eval/export_hf.py."
        )
    print("[verify] PASS")
    return report


# --------------------------------------------------------------------------
# Stage 3 — lm-eval
# --------------------------------------------------------------------------


def pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_harness(hf_dir: Path, tasks: list[str], limit: int | None,
                device: str, batch_size: str, seed: int) -> dict:
    import lm_eval

    try:
        from lm_eval.models.huggingface import HFLM
    except ModuleNotFoundError as e:
        if e.name != "accelerate":
            raise
        # lm-eval's HF wrapper imports accelerate unconditionally, and accelerate
        # is not a project dependency.  Rather than mutate the project venv for an
        # eval-only need, run this script under a uv overlay.
        raise SystemExit(
            "lm-eval needs `accelerate`, which is not in the project venv.\n"
            "Re-run with uv's ephemeral overlay (leaves the project env untouched):\n"
            "  uv run --with accelerate python " + " ".join(sys.argv)
        ) from e

    bs = int(batch_size) if str(batch_size).isdigit() else batch_size
    print(f"[lm-eval] tasks={','.join(tasks)} device={device} batch_size={bs} "
          f"limit={limit or 'full'}")
    lm = HFLM(
        pretrained=str(hf_dir),
        device=device,
        batch_size=bs,
        dtype="float32",
        max_length=1024,          # our trained context; do not let the harness guess
        trust_remote_code=False,
    )
    t0 = time.time()
    res = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        limit=limit,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        bootstrap_iters=1000 if limit is None else 100,
    )
    res["_wall_seconds"] = time.time() - t0
    print(f"[lm-eval] finished in {res['_wall_seconds']/60:.1f} min")
    return res


def extract_scores(results: dict, tasks: list[str]) -> dict:
    """task -> {metric, value_pct, stderr_pct, all_metrics}."""
    out = {}
    for task in tasks:
        r = results.get("results", {}).get(task)
        if not r:
            continue
        metric = ANCH.TASK_METRIC.get(task, "acc")
        key = next((k for k in (f"{metric},none", metric) if k in r), None)
        if key is None:  # task reports something else entirely
            key = next((k for k in r if k.startswith("acc")), None)
            metric = key.split(",")[0] if key else "?"
        val = r.get(key)
        se = r.get(f"{metric}_stderr,none", r.get(f"{metric}_stderr"))
        out[task] = {
            "metric": metric,
            "value_pct": None if val is None else 100.0 * float(val),
            "stderr_pct": None if not isinstance(se, (int, float)) else 100.0 * float(se),
            "all": {k: v for k, v in r.items() if isinstance(v, (int, float))},
        }
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def fmt(v, digits=1, suffix=""):
    return "—" if v is None else f"{v:.{digits}f}{suffix}"


def render_report(ckpt: Path, hf_dir: Path, cfg, state: dict, tasks: list[str],
                  scores: dict, verify_report: dict, results: dict,
                  limit: int | None, device: str) -> str:
    L: list[str] = []
    A = L.append

    from src.config import n_params as count_params
    n_params = count_params(cfg.model)

    A(f"# lm-eval results — {cfg.run_name} @ {ckpt.name}")
    A("")
    A(f"_generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}_")
    A("")
    if limit is not None:
        A(f"> **SMOKE RUN — `--limit {limit}`.** Only {limit} documents per task were "
          f"scored, so these numbers carry huge error bars and are here to prove the "
          f"pipeline runs, not to be compared with anything.")
        A("")
    A("## The model")
    A("")
    A("| | |")
    A("|---|---|")
    A(f"| checkpoint | `{ckpt}` |")
    A(f"| step | {state.get('step', '?'):,} |" if isinstance(state.get("step"), int)
      else f"| step | {state.get('step', '?')} |")
    A(f"| tokens seen | {state.get('tokens', 0)/1e9:.2f}B |")
    A(f"| parameters | {n_params/1e6:.2f}M (tied embeddings) |")
    A(f"| architecture | {cfg.model.n_layers}L / d_model {cfg.model.d_model} / "
      f"{cfg.model.n_heads} heads / SwiGLU {cfg.model.ffn_hidden} / RMSNorm / RoPE θ={cfg.model.rope_theta:g} |")
    A(f"| vocab / context | {cfg.model.vocab_size} / {cfg.model.seq_len} |")
    A(f"| HF export | `{hf_dir}` |")
    A(f"| eval device | {device}, float32 |")
    A("")

    # -- export verification ------------------------------------------------
    A("## Export verification")
    A("")
    A("The MLX checkpoint and the exported HuggingFace model were fed identical "
      "tokens and their next-token logits compared. The **fp32/cpu** column isolates "
      "the weight mapping — both sides in true IEEE float32, so any real difference "
      "here is a bug. The **bf16/gpu** column is the path the model actually runs on, "
      "measured against that fp32 reference: it is the rounding we accept for free, "
      "not an error.")
    A("")
    A("| probe | tokens | max abs Δlogit (fp32/cpu) | mean abs Δ (fp32/cpu) | max abs Δ (bf16/gpu) |")
    A("|---|---:|---:|---:|---:|")
    for p in verify_report["prompts"]:
        A(f"| `{p['prompt'][:46]}…` | {p['n_tokens']} | {p['max_abs_diff_fp32']:.3e} | "
          f"{p['mean_abs_diff_fp32']:.3e} | {p['max_abs_diff_bf16_path']:.3e} |")
    s = verify_report["sample"]
    A(f"| corpus sample (general_val) | {s['n_positions']} | {s['max_abs_diff_fp32']:.3e} | "
      f"{s['mean_abs_diff_fp32']:.3e} | {s['max_abs_diff_bf16_path']:.3e} |")
    A("")
    A(f"- next-token **argmax agreement** over {s['n_positions']} positions: "
      f"**{s['argmax_agreement_fp32']*100:.1f}%** (fp32/cpu), "
      f"{s['argmax_agreement_bf16_path']*100:.1f}% (bf16/gpu) — "
      f"gate is ≥{ARGMAX_AGREEMENT_FLOOR*100:.0f}%")
    A(f"- logit spread for scale: std {s['logit_std']:.2f}")
    A("- **RoPE:** our `mx.fast.rope(traditional=False)` is the half-split "
      "(GPT-NeoX / LLaMA) pairing, which is exactly what `LlamaForCausalLM` expects, "
      "so **no q/k row permutation was applied**. The exporter re-derives this from "
      "MLX at export time rather than trusting a comment.")
    A("- **Why the reference runs on CPU:** MLX's Metal float32 matmul is not IEEE "
      "float32 (~7.6e-4 relative error on a single 1024-dim product, against ~1.1e-6 "
      "for torch and for MLX's own CPU backend). Across 22 layers that compounds into "
      "~2.4e-3 relative error, and on a model whose residual stream peaks above 1000 "
      "that is a *whole logit* of absolute difference — which reads exactly like a "
      "broken export. Pinning the reference forward pass to MLX's CPU stream closes "
      "the gap to ~1e-4.")
    A(f"- **result: {'PASS' if verify_report.get('passed') else 'FAIL'}**")
    A("")

    # -- our scores ---------------------------------------------------------
    A("## Results")
    A("")
    A("| task | metric | score | stderr |")
    A("|---|---|---:|---:|")
    for task in tasks:
        sc = scores.get(task)
        if not sc:
            A(f"| {task} | — | not run | |")
            continue
        se = "—" if sc["stderr_pct"] is None else f"±{fmt(sc['stderr_pct'])}"
        A(f"| {task} | {sc['metric']} | **{fmt(sc['value_pct'])}** | {se} |")
    A("")
    # lambada also reports perplexity, which is a more sensitive read on a small
    # model than its accuracy is — worth carrying through.
    lam = scores.get("lambada_openai", {}).get("all", {})
    if "perplexity,none" in lam:
        A(f"lambada_openai perplexity: **{lam['perplexity,none']:.2f}**")
        A("")

    # -- anchors ------------------------------------------------------------
    A("## Against published models")
    A("")
    if not ANCH.ANCHORS:
        A("_No anchor table configured (see `src/eval/anchors.py`)._")
    else:
        models = list(ANCH.ANCHORS)
        A("| task | metric | **this model** | " + " | ".join(models) + " |")
        A("|---|---|---:|" + "---:|" * len(models))
        for task in tasks:
            sc = scores.get(task)
            mine = f"**{fmt(sc['value_pct'])}**" if sc else "—"
            cells = []
            for m in models:
                v, conf = ANCH.ANCHORS[m].get(task, (None, ""))
                # A missing value is already maximally uncertain; "—?" just looks broken.
                cells.append("—" if v is None else fmt(v) + (conf or ""))
            A(f"| {task} | {ANCH.TASK_METRIC.get(task, 'acc')} | {mine} | " + " | ".join(cells) + " |")
        A("")
        A(getattr(ANCH, "LEGEND", "`?` marks an untraced, approximate value."))
        if ANCH.NOTES:
            A("")
            A(ANCH.NOTES)
        if ANCH.SOURCES:
            A("")
            A("**Anchor sources**")
            A("")
            for src in ANCH.SOURCES:
                A(f"- {src}")
    A("")
    A("Anchors are a mix of published runs and re-measurements (see the legend and "
      "sources above), so harness version, prompt formatting and normalisation are "
      "worth a point or so in either direction. Compare shapes, not decimals.")
    A("")

    # -- provenance ---------------------------------------------------------
    A("## Run details")
    A("")
    import lm_eval
    A(f"- lm-eval {getattr(lm_eval, '__version__', '?')}, `--model hf` on the export "
      f"above, num_fewshot=0, limit={limit or 'none'}, batch float32, "
      f"wall {results.get('_wall_seconds', 0)/60:.1f} min")
    versions = results.get("versions", {})
    if versions:
        A(f"- task versions: {json.dumps(versions)}")
    A(f"- reproduce: `uv run --with accelerate python src/eval/run_lm_eval.py "
      f"--ckpt {ckpt} --tasks {','.join(tasks)}`")
    A("")
    return "\n".join(L)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="export the checkpoint to HF, verify it, and run lm-eval")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--limit", type=int, default=None,
                    help="documents per task (smoke runs only)")
    ap.add_argument("--out", default=None,
                    help="markdown out (default runs/<run>/lm_eval_results.md)")
    ap.add_argument("--export-dir", default=None,
                    help="default runs/hf_export/step_<N>")
    ap.add_argument("--device", default="auto", help="auto|mps|cpu|cuda")
    ap.add_argument("--verify-device", default="cpu",
                    help="device for the logit check (cpu is the reliable fp32 reference)")
    ap.add_argument("--batch-size", default="16")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--rope-permute", default="auto", choices=["auto", "yes", "no"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--skip-export", action="store_true",
                    help="reuse an existing export directory")
    ap.add_argument("--export-only", action="store_true",
                    help="export + verify, then stop (no benchmarks)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    ckpt = resolve_ckpt(args.ckpt)
    cfg = load_ckpt_config(ckpt)
    state = ckpt_state(ckpt)
    run_dir = run_dir_for(ckpt)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    step = state.get("step", ckpt.name.replace("step_", ""))
    hf_dir = Path(args.export_dir) if args.export_dir else REPO_ROOT / "runs" / "hf_export" / f"step_{step}"
    out_path = Path(args.out) if args.out else run_dir / "lm_eval_results.md"

    tokenizer_file = Path(args.tokenizer) if args.tokenizer else None
    tokenizer = load_tokenizer(tokenizer_file)
    from src.eval._common import DEFAULT_TOKENIZER
    tok_file = tokenizer_file or DEFAULT_TOKENIZER

    # 1 — export
    if args.skip_export and (hf_dir / "config.json").exists():
        print(f"[export] reusing {hf_dir}")
    else:
        export(ckpt, hf_dir, tok_file, rope_permute=args.rope_permute)

    # 2 — verify (mandatory; raises on failure)
    verify_report = verify(ckpt, hf_dir, tokenizer, device=args.verify_device)
    (hf_dir / "verify_report.json").write_text(json.dumps(verify_report, indent=2))

    if args.export_only:
        print("[done] --export-only: skipping benchmarks")
        return 0

    # 3 — benchmark
    device = pick_device(args.device)
    try:
        results = run_harness(hf_dir, tasks, args.limit, device, args.batch_size, args.seed)
    except Exception as e:
        if device == "cpu":
            raise
        print(f"[lm-eval] {device} failed ({type(e).__name__}: {e}); retrying on cpu")
        device = "cpu"
        results = run_harness(hf_dir, tasks, args.limit, device, args.batch_size, args.seed)

    scores = extract_scores(results, tasks)
    for task, sc in scores.items():
        print(f"  {task:18s} {sc['metric']:9s} {fmt(sc['value_pct'])}"
              + (f" ± {fmt(sc['stderr_pct'])}" if sc["stderr_pct"] is not None else ""))

    md = render_report(ckpt, hf_dir, cfg, state, tasks, scores,
                       verify_report, results, args.limit, device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)

    raw = out_path.with_suffix(".json")
    raw.write_text(json.dumps(
        {"results": results.get("results"), "versions": results.get("versions"),
         "n-samples": results.get("n-samples"), "limit": args.limit,
         "device": device, "ckpt": str(ckpt), "verify": verify_report},
        indent=2, default=str))

    print(f"\n[done] wrote {out_path}\n[done] wrote {raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
