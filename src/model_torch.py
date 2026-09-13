"""The transformer — PyTorch/MPS implementation.

A line-for-line twin of `src/model.py` (MLX).  Read that file first: it carries
the full explanation of *why* each component is here (pre-norm, RMSNorm, RoPE,
SwiGLU, tied embeddings, no biases, the 1/sqrt(2*n_layers) output-projection
init, and the mixed-precision policy).  This file repeats the *what* with
PyTorch spellings and only comments on the places where the two frameworks
genuinely differ.

WHY TWO IMPLEMENTATIONS AT ALL?
On Apple silicon the fast path is not settled: MLX is designed for unified
memory and has lower kernel-launch overhead, but its bf16 matmul on M5 is
currently ~1.1–1.2x slower than PyTorch's MPS backend (mlx#3196).  Rather than
guess, we build both and let `calibrate.py` measure tok/s on the real model.
The cost is this file; the benefit is picking the faster framework for a
multi-day training run, which is worth far more than the duplication.

The same mixed-precision policy applies: parameters (and therefore optimizer
state) live in fp32, every matmul casts its weight to bf16, and logits/loss
come back to fp32.  We cast explicitly instead of using `torch.autocast` so the
precision boundaries are visible in the code and identical to the MLX version.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ModelConfig, n_params  # noqa: E402

COMPUTE_DTYPE = torch.bfloat16


def best_device() -> torch.device:
    """MPS if present, else CPU (CI / fallback)."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


class Linear(nn.Module):
    """Bias-free linear; fp32 master weight, bf16 compute."""

    def __init__(self, in_features: int, out_features: int, std: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.rms_norm upcasts the reduction internally, matching mx.fast.rms_norm.
        return F.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.eps)


def build_rope_cache(head_dim: int, max_seq: int, theta: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables of shape (max_seq, head_dim/2).

    MLX ships a fused `mx.fast.rope`; PyTorch does not, so we build the angle
    table once (it never changes) and index it.  Frequencies fall off
    geometrically: dimension pair i rotates at 1/theta^(2i/d), so the low pairs
    spin fast (they encode "a few tokens apart") and the high pairs spin slowly
    (they encode "thousands of tokens apart").
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_seq, device=device).float()
    ang = torch.outer(pos, inv_freq)             # (max_seq, head_dim/2)
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int) -> torch.Tensor:
    """Rotate (B, n_heads, T, head_dim) by the angles for positions offset..offset+T.

    "Non-traditional" / LLaMA-style pairing: the first half of the head
    dimension is paired with the second half (x1, x2) -> (x1 cos - x2 sin,
    x1 sin + x2 cos).  `mx.fast.rope(traditional=False)` uses the same pairing,
    so the two backends implement the identical function.
    """
    T = x.shape[-2]
    c = cos[offset : offset + T].to(x.dtype)[None, None, :, :]
    s = sin[offset : offset + T].to(x.dtype)[None, None, :, :]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, init_std: float, out_std: float):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        assert self.head_dim * cfg.n_heads == cfg.d_model, "d_model must divide by n_heads"
        self.wq = Linear(cfg.d_model, cfg.d_model, init_std)
        self.wk = Linear(cfg.d_model, cfg.d_model, init_std)
        self.wv = Linear(cfg.d_model, cfg.d_model, init_std)
        self.wo = Linear(cfg.d_model, cfg.d_model, out_std)

    def forward(self, x, cos, sin, cache: list | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        offset = cache[0].shape[2] if (cache is not None and cache) else 0
        q = apply_rope(q, cos, sin, offset)
        k = apply_rope(k, cos, sin, offset)

        if cache is not None:
            if cache:
                k = torch.cat([cache[0], k], dim=2)
                v = torch.cat([cache[1], v], dim=2)
            cache[:] = [k, v]

        # F.scaled_dot_product_attention picks a fused kernel and never
        # materialises the (B, n_heads, T, T) score matrix — at batch 32, 16
        # heads and T=1024 that array alone would be 34GB per layer.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig, init_std: float, out_std: float):
        super().__init__()
        self.w1 = Linear(cfg.d_model, cfg.ffn_hidden, init_std)
        self.w3 = Linear(cfg.d_model, cfg.ffn_hidden, init_std)
        self.w2 = Linear(cfg.ffn_hidden, cfg.d_model, out_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, init_std: float, out_std: float):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, init_std, out_std)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg, init_std, out_std)

    def forward(self, x, cos, sin, cache: list | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------


class Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        init_std = 0.02
        out_std = init_std / math.sqrt(2 * cfg.n_layers)

        self.tok_emb = nn.Parameter(torch.empty(cfg.vocab_size, cfg.d_model))
        nn.init.normal_(self.tok_emb, mean=0.0, std=init_std)
        self.blocks = nn.ModuleList([Block(cfg, init_std, out_std) for _ in range(cfg.n_layers)])
        self.out_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        if not cfg.tied_embeddings:
            self.lm_head = Linear(cfg.d_model, cfg.vocab_size, init_std)

        head_dim = cfg.d_model // cfg.n_heads
        cos, sin = build_rope_cache(head_dim, cfg.seq_len, cfg.rope_theta, "cpu")
        # Buffers (not parameters): they move with `.to(device)` and are saved
        # nowhere, because they are a pure function of the config.
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _rope_tables(self, needed: int):
        """Grow the RoPE cache if generation runs past seq_len."""
        if needed > self.rope_cos.shape[0]:
            cos, sin = build_rope_cache(
                self.cfg.d_model // self.cfg.n_heads, needed, self.cfg.rope_theta,
                self.rope_cos.device,
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        return self.rope_cos, self.rope_sin

    def hidden(self, tokens: torch.Tensor, cache: list | None = None) -> torch.Tensor:
        """tokens (B, T) int64 -> final hidden states (B, T, d_model), bf16."""
        B, T = tokens.shape
        offset = cache[0][0].shape[2] if (cache is not None and cache[0]) else 0
        cos, sin = self._rope_tables(offset + T)

        h = self.tok_emb[tokens].to(COMPUTE_DTYPE)
        for i, block in enumerate(self.blocks):
            h = block(h, cos, sin, None if cache is None else cache[i])
        return self.out_norm(h)

    def _unembed(self, h: torch.Tensor) -> torch.Tensor:
        if self.cfg.tied_embeddings:
            return h @ self.tok_emb.to(h.dtype).t()
        return self.lm_head(h)

    def forward(self, tokens: torch.Tensor, cache: list | None = None) -> torch.Tensor:
        """tokens (B, T) int64 -> logits (B, T, vocab) fp32 (inference path)."""
        return self._unembed(self.hidden(tokens, cache)).float()

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean cross-entropy, reduced in fp32.  See model.py's `loss` for why
        the logits themselves stay in bf16: that tensor (B*T*vocab) is the
        largest activation in the model and an fp32 copy of it costs more peak
        memory than the entire parameter set."""
        logits = self._unembed(self.hidden(tokens))
        ce = F.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1).long(),
            reduction="none",
        )
        return ce.float().mean()

    # -- sampling --------------------------------------------------------

    def new_cache(self) -> list[list]:
        return [[] for _ in self.blocks]

    @torch.no_grad()
    def generate(
        self,
        prompt_ids,
        max_new: int = 64,
        temp: float = 0.8,
        top_p: float = 0.95,
        seed: int = 0,
        eos_id: int | None = None,
    ) -> list[int]:
        device = self.tok_emb.device
        # A private generator keeps vibe sampling from perturbing the training
        # RNG stream (there isn't one today — no dropout — but a global
        # manual_seed here would be a nasty surprise later).
        gen = torch.Generator(device="cpu").manual_seed(seed)
        ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
        cache = self.new_cache()
        out: list[int] = []

        logits = self(ids, cache)[:, -1, :]
        for _ in range(max_new):
            if temp <= 0:
                nxt = int(torch.argmax(logits, dim=-1).item())
            else:
                probs = torch.softmax(logits / temp, dim=-1).cpu()
                nxt = _top_p_sample(probs, top_p, gen)
            if eos_id is not None and nxt == eos_id:
                break
            out.append(nxt)
            logits = self(torch.tensor([[nxt]], dtype=torch.long, device=device), cache)[:, -1, :]
        return out


def _top_p_sample(probs: torch.Tensor, top_p: float, gen: torch.Generator) -> int:
    if top_p < 1.0:
        sorted_p, order = torch.sort(probs, descending=True, dim=-1)
        cum = sorted_p.cumsum(dim=-1)
        keep = (cum - sorted_p) < top_p          # always keeps the top token
        sorted_p = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
        sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True)
        pick = torch.multinomial(sorted_p, 1, generator=gen)
        return int(order.gather(-1, pick).item())
    return int(torch.multinomial(probs, 1, generator=gen).item())


def count_params(model: Model) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------
# Smoke test: `uv run python -m src.model_torch`
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import time

    from src.config import load_config

    ap = argparse.ArgumentParser(description="PyTorch transformer smoke test")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dev = torch.device(args.device) if args.device else best_device()
    torch.manual_seed(0)
    t0 = time.time()
    model = Model(cfg.model).to(dev)
    print(f"built on {dev} in {time.time()-t0:.1f}s")

    got, want = count_params(model), n_params(cfg.model)
    err = abs(got - want) / want
    print(f"params: {got/1e6:.2f}M (config.n_params says {want/1e6:.2f}M, err {err*100:.3f}%)")

    B, T = 2, 128
    x = torch.randint(0, cfg.model.vocab_size, (B, T), device=dev)
    y = torch.randint(0, cfg.model.vocab_size, (B, T), device=dev)
    t0 = time.time()
    with torch.no_grad():
        logits = model(x)
        loss = model.loss(x, y)
    if dev.type == "mps":
        torch.mps.synchronize()
    print(f"forward {tuple(logits.shape)} dtype={logits.dtype} in {time.time()-t0:.2f}s")
    print(f"init loss {loss.item():.4f} (expect ~ln({cfg.model.vocab_size}) = "
          f"{math.log(cfg.model.vocab_size):.4f})")

    t0 = time.time()
    gen = model.generate([1, 2, 3, 4], max_new=20, temp=0.8, top_p=0.95, seed=0)
    print(f"generated 20 tokens in {time.time()-t0:.2f}s: {gen}")

    ok = err < 0.01 and abs(loss.item() - math.log(cfg.model.vocab_size)) < 0.5 and len(gen) == 20
    print("OK  model_torch.py smoke test" if ok else "FAILED  model_torch.py smoke test")
    sys.exit(0 if ok else 1)
