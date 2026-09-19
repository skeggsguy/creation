"""Export our MLX checkpoint to a HuggingFace `LlamaForCausalLM` directory.

WHY THIS IS A PURE RENAME
=========================
Our architecture (src/model.py) is the LLaMA recipe with nothing extra:
pre-norm blocks, RMSNorm without bias, SwiGLU MLP, RoPE, no biases anywhere,
tied embeddings, MHA (n_kv_heads == n_heads).  Every tensor therefore has an
exact counterpart in `LlamaForCausalLM`, with the same shape and the same
orientation, so the export is a dictionary of renames plus a config object.
Two details are worth stating explicitly, because they are where naive
converters break:

**Linear orientation.**  Our `Linear` stores `weight` as `(out_features,
in_features)` and computes `x @ weight.T` — byte-for-byte the same layout and
the same maths as `torch.nn.Linear`.  No transposes.

**RoPE pairing.**  There are two conventions for which dimension pairs get
rotated together.  GPT-J / "traditional" rotates *adjacent* dims (0,1), (2,3),
...; GPT-NeoX and HF-LLaMA rotate *half-split* pairs (i, i + head_dim/2).  A
converter that gets this wrong still loads cleanly and still produces
plausible-looking logits — it just quietly destroys positional information,
which is why GPT-NeoX->LLaMA converters permute the rows of q_proj/k_proj.
We call `mx.fast.rope(..., traditional=False)`, which *is* the half-split
convention, so our q/k rows already sit in LLaMA order and no permutation is
needed.  `assert_rope_convention()` below re-derives that from MLX at export
time rather than trusting this comment, and `--rope-permute` exists so the fix
is one flag away if MLX ever changes its semantics.

The head layout matches too: `wq(x).reshape(B, T, n_heads, head_dim)` groups the
output features head-major, exactly as LLaMA slices q_proj.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval._common import (  # noqa: E402
    EOS_ID,
    EOS_TOKEN,
    ckpt_state,
    load_ckpt_config,
)


# --------------------------------------------------------------------------
# RoPE convention guard
# --------------------------------------------------------------------------


def assert_rope_convention() -> str:
    """Return "half_split" | "interleaved" for what `mx.fast.rope(traditional=False)` does.

    We rebuild HF-LLaMA's rotary by hand on a tiny tensor and see which MLX mode
    reproduces it.  Cheap, and it turns a silent correctness bug into a loud
    failure at export time.
    """
    import mlx.core as mx
    import torch

    d, T = 8, 4
    x = mx.arange(T * d, dtype=mx.float32).reshape(1, 1, T, d) / 10.0
    xt = torch.tensor(x.tolist(), dtype=torch.float32)

    inv = 1.0 / (10000.0 ** (torch.arange(0, d, 2).float() / d))
    freqs = torch.outer(torch.arange(T).float(), inv)
    emb = torch.cat([freqs, freqs], dim=-1)

    def rotate_half(t):
        h = t.shape[-1] // 2
        return torch.cat([-t[..., h:], t[..., :h]], dim=-1)

    llama = xt * emb.cos() + rotate_half(xt) * emb.sin()

    def diff(traditional: bool) -> float:
        r = mx.fast.rope(x, d, traditional=traditional, base=10000.0, scale=1.0, offset=0)
        return (torch.tensor(r.tolist(), dtype=torch.float32) - llama).abs().max().item()

    d_half, d_trad = diff(False), diff(True)
    if d_half < 1e-4:
        return "half_split"
    if d_trad < 1e-4:
        return "interleaved"
    raise SystemExit(
        f"cannot identify MLX rope convention (half-split diff {d_half:.4g}, "
        f"interleaved diff {d_trad:.4g}) — export would be unsafe"
    )


def permute_for_llama(w, n_heads: int, head_dim: int):
    """Interleaved (GPT-J) q/k rows -> half-split (LLaMA) rows.

    Only needed if our RoPE ever becomes `traditional=True`.  This is the same
    permutation the official LLaMA/NeoX conversion scripts apply.
    """
    out, in_ = w.shape
    return (
        w.reshape(n_heads, head_dim // 2, 2, in_)
        .transpose(1, 2)
        .reshape(out, in_)
    )


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def build_llama_config(mcfg):
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=mcfg.vocab_size,
        hidden_size=mcfg.d_model,
        intermediate_size=mcfg.ffn_hidden,
        num_hidden_layers=mcfg.n_layers,
        num_attention_heads=mcfg.n_heads,
        num_key_value_heads=mcfg.n_heads,     # MHA, not GQA
        hidden_act="silu",                    # SwiGLU's gate nonlinearity
        max_position_embeddings=mcfg.seq_len,
        rms_norm_eps=mcfg.norm_eps,
        rope_theta=mcfg.rope_theta,
        tie_word_embeddings=mcfg.tied_embeddings,
        attention_bias=False,
        mlp_bias=False,
        bos_token_id=EOS_ID,
        eos_token_id=EOS_ID,
        pad_token_id=EOS_ID,
    )


def map_weights(weights: dict, mcfg, rope_permute: bool = False) -> dict:
    """MLX param names -> LLaMA param names, as torch fp32 tensors."""
    import numpy as np
    import torch

    head_dim = mcfg.d_model // mcfg.n_heads

    def t(name):
        arr = weights[name]
        return torch.from_numpy(np.array(arr, dtype=np.float32, copy=True))

    sd: dict = {"model.embed_tokens.weight": t("tok_emb")}

    for i in range(mcfg.n_layers):
        p, q = f"blocks.{i}", f"model.layers.{i}"
        wq, wk = t(f"{p}.attn.wq.weight"), t(f"{p}.attn.wk.weight")
        if rope_permute:
            wq = permute_for_llama(wq, mcfg.n_heads, head_dim)
            wk = permute_for_llama(wk, mcfg.n_heads, head_dim)
        sd[f"{q}.self_attn.q_proj.weight"] = wq
        sd[f"{q}.self_attn.k_proj.weight"] = wk
        sd[f"{q}.self_attn.v_proj.weight"] = t(f"{p}.attn.wv.weight")
        sd[f"{q}.self_attn.o_proj.weight"] = t(f"{p}.attn.wo.weight")
        sd[f"{q}.mlp.gate_proj.weight"] = t(f"{p}.mlp.w1.weight")   # silu'd branch
        sd[f"{q}.mlp.up_proj.weight"] = t(f"{p}.mlp.w3.weight")     # value branch
        sd[f"{q}.mlp.down_proj.weight"] = t(f"{p}.mlp.w2.weight")
        sd[f"{q}.input_layernorm.weight"] = t(f"{p}.attn_norm.weight")
        sd[f"{q}.post_attention_layernorm.weight"] = t(f"{p}.mlp_norm.weight")

    sd["model.norm.weight"] = t("out_norm.weight")
    if not mcfg.tied_embeddings:
        sd["lm_head.weight"] = t("lm_head.weight")
    return sd


EXPECTED_SHAPES = {
    "q_proj": ("d", "d"), "k_proj": ("d", "d"), "v_proj": ("d", "d"), "o_proj": ("d", "d"),
    "gate_proj": ("f", "d"), "up_proj": ("f", "d"), "down_proj": ("d", "f"),
}


def check_shapes(sd: dict, mcfg) -> None:
    dims = {"d": mcfg.d_model, "f": mcfg.ffn_hidden}
    bad = []
    for name, w in sd.items():
        leaf = name.split(".")[-2]
        if leaf in EXPECTED_SHAPES:
            want = tuple(dims[c] for c in EXPECTED_SHAPES[leaf])
            if tuple(w.shape) != want:
                bad.append(f"{name}: {tuple(w.shape)} != {want}")
        elif name.endswith("norm.weight") and tuple(w.shape) != (mcfg.d_model,):
            bad.append(f"{name}: {tuple(w.shape)} != ({mcfg.d_model},)")
    if tuple(sd["model.embed_tokens.weight"].shape) != (mcfg.vocab_size, mcfg.d_model):
        bad.append("embed_tokens shape mismatch")
    if bad:
        raise SystemExit("weight shape check failed:\n  " + "\n  ".join(bad))


def export_tokenizer(out_dir: Path, tokenizer_file: Path) -> None:
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_file),
        eos_token=EOS_TOKEN,
        bos_token=EOS_TOKEN,
        unk_token=EOS_TOKEN,
        pad_token=EOS_TOKEN,
    )
    tok.save_pretrained(str(out_dir))


def export(ckpt: Path, out_dir: Path, tokenizer_file: Path,
           rope_permute: str = "auto", quiet: bool = False) -> Path:
    """Write a complete HF model directory.  Returns `out_dir`."""
    import mlx.core as mx
    import torch
    from transformers import LlamaForCausalLM

    cfg = load_ckpt_config(ckpt)
    mcfg = cfg.model

    convention = assert_rope_convention()
    if rope_permute == "auto":
        do_permute = convention == "interleaved"
    else:
        do_permute = rope_permute == "yes"
    if not quiet:
        print(f"[export] MLX rope convention: {convention} "
              f"(LLaMA wants half_split) -> permute q/k rows: {do_permute}")

    weights = mx.load(str(ckpt / "model.safetensors"))
    sd = map_weights(weights, mcfg, rope_permute=do_permute)
    check_shapes(sd, mcfg)

    lcfg = build_llama_config(mcfg)
    model = LlamaForCausalLM(lcfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # With tied embeddings `lm_head.weight` is a view of `embed_tokens.weight`,
    # so it is legitimately absent from our dict; anything else missing is a bug.
    allowed_missing = {"lm_head.weight"} if mcfg.tied_embeddings else set()
    if set(missing) - allowed_missing or unexpected:
        raise SystemExit(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
    model.tie_weights()
    model = model.to(dtype=torch.float32).eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    export_tokenizer(out_dir, tokenizer_file)

    state = ckpt_state(ckpt)
    (out_dir / "export_info.json").write_text(json.dumps({
        "source_ckpt": str(ckpt),
        "step": state.get("step"),
        "tokens": state.get("tokens"),
        "run_name": cfg.run_name,
        "mlx_rope_convention": convention,
        "rope_permute_applied": do_permute,
        "n_tensors_mapped": len(sd),
    }, indent=2))

    if not quiet:
        n = sum(p.numel() for p in model.parameters())
        print(f"[export] {len(sd)} tensors -> {out_dir} ({n/1e6:.2f}M params, tied={mcfg.tied_embeddings})")
    return out_dir
