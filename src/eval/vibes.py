"""Final vibe eval — one fixed-seed continuation per prompt, as readable markdown.

    uv run python src/eval/vibes.py --ckpt runs/run01/ckpt/step_366211

This is the same sampling the trainer did every 100M tokens (src/train.py's
`run_vibe_samples`), but aimed at one checkpoint and rendered for a human
instead of appended to samples.jsonl.  Seeds are `seed + i` per prompt — the
trainer's convention — so prompt 7 gets the same noise here as it did mid-run
and two reports can be diffed line by line.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval._common import (  # noqa: E402
    EOS_ID,
    REPO_ROOT,
    ckpt_state,
    load_mlx_model,
    load_tokenizer,
    read_prompts,
    resolve_ckpt,
    run_dir_for,
)

DEFAULT_PROMPTS = REPO_ROOT / "prompts" / "vibes.txt"


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="fixed-seed vibe samples from a checkpoint")
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (or run dir / ckpt/latest)")
    ap.add_argument("--out", default=None, help="markdown out (default runs/<run>/final_vibes.md)")
    ap.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer", default=None)
    return ap


def as_blockquote(text: str) -> str:
    """Render a completion as a markdown blockquote, keeping its line breaks.

    Generated text is full of newlines (the model loves a paragraph break), and
    an unquoted blank line would end the quote block, so every line — empty ones
    included — gets its own `>`.
    """
    lines = text.split("\n")
    return "\n".join(("> " + ln).rstrip() for ln in lines)


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    ckpt = resolve_ckpt(args.ckpt)
    run_dir = run_dir_for(ckpt)
    out_path = Path(args.out) if args.out else run_dir / "final_vibes.md"

    prompts = read_prompts(Path(args.prompts))
    tok = load_tokenizer(args.tokenizer)

    t0 = time.time()
    model, cfg = load_mlx_model(ckpt)
    load_s = time.time() - t0
    state = ckpt_state(ckpt)

    header = [
        f"# Vibe eval — {cfg.run_name} @ {ckpt.name}",
        "",
        f"- checkpoint: `{ckpt}`",
        f"- step {state.get('step', '?')}, {state.get('tokens', 0)/1e9:.2f}B tokens seen",
        f"- model: d_model {cfg.model.d_model}, {cfg.model.n_layers} layers, "
        f"{cfg.model.n_heads} heads, vocab {cfg.model.vocab_size}",
        f"- sampling: temp {args.temp}, top_p {args.top_p}, max_new {args.max_new}, "
        f"seed {args.seed} (+i per prompt)",
        f"- prompts: `{args.prompts}` ({len(prompts)} prompts)",
        "",
        "---",
        "",
        "",
    ]
    body: list[str] = []
    print("\n".join(header))

    gen_t0 = time.time()
    for i, prompt in enumerate(prompts):
        ids = tok.encode(prompt).ids
        if not ids:
            continue
        out_ids = model.generate(
            ids,
            max_new=args.max_new,
            temp=args.temp,
            top_p=args.top_p,
            seed=args.seed + i,
            eos_id=EOS_ID,
        )
        text = tok.decode(out_ids)
        chunk = f"## {prompt}\n\n{as_blockquote(prompt + text)}\n"
        body.append(chunk)
        print(chunk, flush=True)

    elapsed = time.time() - gen_t0
    footer = [
        "---",
        "",
        f"_{len(body)} samples in {elapsed:.1f}s (model load {load_s:.1f}s)._",
        "",
    ]
    print("\n".join(footer))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(header) + "\n".join(body) + "\n" + "\n".join(footer))
    print(f"[vibes] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
