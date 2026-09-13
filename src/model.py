"""The transformer — MLX implementation.

This file and `model_torch.py` describe the *same* architecture twice, once per
backend, because on an M5 Max it is not obvious which framework is faster
(decisions.md: MLX's bf16 matmul currently regresses ~1.1–1.2x vs PyTorch MPS,
mlx#3196) and we want `calibrate.py` to settle it with measurements rather than
vibes.  Keep the two files in lockstep: same modules, same init, same maths.

WHAT WE BUILD, AND WHY EACH PIECE EXISTS
========================================
A decoder-only, pre-norm transformer — the GPT/LLaMA lineage:

    tokens -> embedding -> [ Block x n_layers ] -> RMSNorm -> tied unembedding

    Block(x) = x + Attention(RMSNorm(x))
               x + SwiGLU_MLP(RMSNorm(x))

* **Pre-norm** (normalise *inside* the branch, not after the residual add).
  Post-norm — the original 2017 arrangement — puts a LayerNorm on the residual
  highway itself, so the identity path is rescaled at every layer and gradients
  shrink/explode with depth; 22 layers of that needs careful warmup babysitting.
  Pre-norm leaves a clean, un-normalised additive path from the embedding to
  the final norm, which is why every large model since GPT-2 uses it.

* **RMSNorm instead of LayerNorm.**  LayerNorm subtracts the mean and divides
  by the standard deviation.  RMSNorm skips the mean-subtraction and just
  divides by the root-mean-square.  Empirically the centring does nothing for
  transformer quality, and dropping it removes a full pass over the activation
  plus the mean/variance bookkeeping — a real win on a bandwidth-bound machine.
  We also drop the bias (see "no biases" below).

* **RoPE instead of learned position embeddings.**  Rotary embeddings rotate
  the query and key vectors by an angle proportional to their absolute
  position; because attention scores are inner products, the score between
  positions i and j ends up depending only on (i - j).  So the model gets
  *relative* position information for free, with zero parameters, and nothing
  breaks if we later want to extend the context.

* **SwiGLU MLP.**  The classic MLP is `W2 @ gelu(W1 @ x)`.  SwiGLU splits the
  up-projection in two, `silu(W1 @ x) * (W3 @ x)`, so one half acts as a
  learned, input-dependent *gate* on the other.  It costs a third matrix, which
  is why the hidden dim is ~2.67x d_model instead of 4x — that keeps the
  parameter count (and FLOPs) the same as a 4x GELU MLP while measurably
  improving loss.

* **Tied input/output embeddings.**  The unembedding reuses the embedding
  matrix.  At vocab 24576 x d_model 1024 that is 25M parameters — 8% of the
  model — saved, and the two matrices want to learn the same token-similarity
  structure anyway.  Tying is close to free in quality at this scale and lets
  us spend the parameters on depth instead.

* **No biases anywhere** (no bias in Linear, no bias in the norms).  Biases add
  parameters and memory traffic, and at this scale they contribute nothing
  measurable — everyone from PaLM to LLaMA drops them.

* **Init: N(0, 0.02), with output projections scaled by 1/sqrt(2*n_layers).**
  Every block adds two contributions (attention, MLP) to the residual stream.
  With n_layers blocks, the variance of the residual stream would grow like
  2*n_layers if each contribution had unit-ish scale, so activations at the
  top would be far larger than at the bottom.  Scaling the *output* projection
  of each branch by 1/sqrt(2*n_layers) keeps the residual stream's variance
  roughly constant with depth.  This is the GPT-2 trick, and it is the
  difference between a stable start and an early loss spike.

PRECISION POLICY (mixed, not pure-bf16)
=======================================
Parameters are stored in **fp32** and are the master copy; every matmul casts
the weight down to **bf16** on the fly, so the heavy arithmetic runs at bf16
speed while the optimizer accumulates updates in fp32.  Pure-bf16 training
(bf16 optimizer state too) measurably hurt final loss in comparable runs: an
update of ~1e-7 against a weight of ~1e-2 simply vanishes in bf16's 8 mantissa
bits.  The casts are cheap (bandwidth, not FLOPs) and autodiff carries the
gradient back through `astype` into fp32 for us, which is exactly the
"fp32 master weights" scheme without the bookkeeping of keeping two copies.
The loss is computed on fp32 logits, and grad-norm/clipping happen in fp32.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ModelConfig, n_params  # noqa: E402

COMPUTE_DTYPE = mx.bfloat16


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


class Linear(nn.Module):
    """Bias-free linear layer that keeps fp32 weights but computes in bf16.

    We do not use `nn.Linear` because we want the cast to be explicit and
    visible: `self.weight` is the fp32 master copy that AdamW updates, and
    `.astype(x.dtype)` is the only place precision is lost.
    """

    def __init__(self, in_features: int, out_features: int, std: float):
        super().__init__()
        self.weight = mx.random.normal((out_features, in_features), scale=std)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T


class RMSNorm(nn.Module):
    """x / rms(x) * weight, with no mean subtraction and no bias.

    `mx.fast.rms_norm` is a fused kernel that accumulates the sum of squares in
    fp32 even when the input is bf16 — important, because summing 1024 squared
    bf16 values in bf16 would lose precision exactly where we need it.
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight.astype(x.dtype), self.eps)


class Attention(nn.Module):
    """Causal multi-head self-attention with RoPE.

    Shapes: (B, T, d_model) -> project to (B, n_heads, T, head_dim) -> attend
    -> merge heads -> (B, T, d_model).  The head split is a reshape, not a
    parameter: each head gets its own `head_dim`-sized slice of the projection
    so it can specialise (induction heads, positional heads, ...).
    """

    def __init__(self, cfg: ModelConfig, init_std: float, out_std: float):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        assert self.head_dim * cfg.n_heads == cfg.d_model, "d_model must divide by n_heads"
        # 1/sqrt(head_dim): without it, the dot product of two random
        # head_dim-dimensional vectors grows like sqrt(head_dim) and pushes the
        # softmax into its saturated, near-zero-gradient regime.
        self.scale = self.head_dim ** -0.5
        self.rope_theta = cfg.rope_theta

        self.wq = Linear(cfg.d_model, cfg.d_model, init_std)
        self.wk = Linear(cfg.d_model, cfg.d_model, init_std)
        self.wv = Linear(cfg.d_model, cfg.d_model, init_std)
        self.wo = Linear(cfg.d_model, cfg.d_model, out_std)  # residual-scaled init

    def __call__(self, x: mx.array, cache: list | None = None) -> mx.array:
        B, T, _ = x.shape
        # (B, T, d) -> (B, n_heads, T, head_dim)
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # During cached generation the new token sits at absolute position
        # `offset`, so RoPE must be applied with that offset or the model would
        # think every generated token is at position 0.
        offset = 0 if cache is None or not cache else cache[0].shape[2]
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.rope_theta,
                         scale=1.0, offset=offset)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.rope_theta,
                         scale=1.0, offset=offset)

        if cache is not None:
            if cache:
                k = mx.concatenate([cache[0], k], axis=2)
                v = mx.concatenate([cache[1], v], axis=2)
            cache[:] = [k, v]

        # mask="causal" is the whole point of a decoder-only model: position t
        # may attend to 0..t but never to the future, so a single forward pass
        # yields T independent next-token predictions instead of one.  The fused
        # kernel also never materialises the (B, n_heads, T, T) score matrix,
        # which at batch 32 / 16 heads / T=1024 would be 34GB per layer.
        # When generating with a cache we feed a single query against all cached
        # keys — everything in the cache is already in the past, so no mask.
        mask = "causal" if T > 1 else None
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    """silu(W1 x) * (W3 x) -> W2.  See the header for why it is gated."""

    def __init__(self, cfg: ModelConfig, init_std: float, out_std: float):
        super().__init__()
        self.w1 = Linear(cfg.d_model, cfg.ffn_hidden, init_std)     # gate branch
        self.w3 = Linear(cfg.d_model, cfg.ffn_hidden, init_std)     # value branch
        self.w2 = Linear(cfg.ffn_hidden, cfg.d_model, out_std)      # residual-scaled init

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    """One pre-norm transformer layer: attention then MLP, both residual."""

    def __init__(self, cfg: ModelConfig, init_std: float, out_std: float):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, init_std, out_std)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg, init_std, out_std)

    def __call__(self, x: mx.array, cache: list | None = None) -> mx.array:
        x = x + self.attn(self.attn_norm(x), cache)
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
        # See the header: shrink every branch's *output* projection so the
        # residual stream's variance does not grow with depth.
        out_std = init_std / math.sqrt(2 * cfg.n_layers)

        self.tok_emb = mx.random.normal((cfg.vocab_size, cfg.d_model), scale=init_std)
        self.blocks = [Block(cfg, init_std, out_std) for _ in range(cfg.n_layers)]
        self.out_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        if not cfg.tied_embeddings:
            self.lm_head = Linear(cfg.d_model, cfg.vocab_size, init_std)

    # -- forward ---------------------------------------------------------

    def hidden(self, tokens: mx.array, cache: list | None = None) -> mx.array:
        """tokens (B, T) int -> final hidden states (B, T, d_model), bf16."""
        # Embedding lookup is a gather from the fp32 table; we cast the *result*
        # (B*T*d values) rather than the table (vocab*d values) — far less
        # traffic — and from here on the activations are bf16.
        h = self.tok_emb[tokens].astype(COMPUTE_DTYPE)
        for i, block in enumerate(self.blocks):
            h = block(h, None if cache is None else cache[i])
        return self.out_norm(h)

    def _unembed(self, h: mx.array) -> mx.array:
        if self.cfg.tied_embeddings:
            return h @ self.tok_emb.astype(h.dtype).T
        return self.lm_head(h)

    def __call__(self, tokens: mx.array, cache: list | None = None) -> mx.array:
        """tokens (B, T) int -> logits (B, T, vocab), fp32.

        fp32 here is for convenience at *inference* scale (sampling, eval), where
        the tensor is one row.  Training uses `loss()` below, which deliberately
        does not materialise an fp32 copy — see the comment there.
        """
        return self._unembed(self.hidden(tokens, cache)).astype(mx.float32)

    def loss(self, tokens: mx.array, targets: mx.array) -> mx.array:
        """Mean next-token cross-entropy, reduced in fp32.  We differentiate this.

        THE LOGITS TENSOR IS THE BIGGEST ACTIVATION IN THE MODEL.  At batch 16 x
        seq 1024 x vocab 24576 it holds 400M values — bigger than the model's
        303M parameters — and autograd keeps it alive, along with the softmax
        and its gradient, until the backward pass reaches it.  In bf16 that is
        ~0.8GB per copy; in fp32 it is 1.6GB per copy, and there are several.
        Casting it up measured +12% peak memory for no benefit: with bf16 logits
        and only the final reduction in fp32, the loss differs from the fully
        fp32 computation by ~1e-4 and the gradients differ by less than bf16's
        own rounding.  (The logits already came out of a bf16 matmul, so the
        precision was spent before the cast anyway.)  So: per-token
        cross-entropy in bf16, mean in fp32.

        Activation memory is *the* limit on batch size here, not parameters —
        the weights and Adam state are only ~4.8GB.  That is why calibrate.py
        sweeps the batch size instead of assuming a big one fits.
        """
        logits = self._unembed(self.hidden(tokens))
        ce = nn.losses.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1), reduction="none"
        )
        return ce.astype(mx.float32).mean()

    # -- sampling --------------------------------------------------------

    def new_cache(self) -> list[list]:
        """One (empty) key/value slot per layer.

        The KV cache is what makes generation O(n) instead of O(n^2): keys and
        values for tokens we have already processed never change, so we compute
        them once and only run the new token through the projections.
        """
        return [[] for _ in self.blocks]

    def generate(
        self,
        prompt_ids,
        max_new: int = 64,
        temp: float = 0.8,
        top_p: float = 0.95,
        seed: int = 0,
        eos_id: int | None = None,
    ) -> list[int]:
        """Sample a continuation. Returns the generated ids (prompt excluded).

        `temp` flattens (>1) or sharpens (<1) the distribution; `top_p`
        (nucleus) keeps only the smallest set of tokens whose probability mass
        reaches p, which cuts the long tail of garbage tokens without the
        fixed-k rigidity of top-k.  `seed` is explicit so the vibe samples in
        samples.jsonl are comparable across checkpoints.
        """
        key = mx.random.key(seed)
        ids = mx.array(list(prompt_ids), dtype=mx.int32)[None]  # (1, T)
        cache = self.new_cache()
        out: list[int] = []

        # Prefill: run the whole prompt once, keep only the last position's
        # logits (the only one that predicts a token we have not seen).
        logits = self(ids, cache)[:, -1, :]
        for _ in range(max_new):
            if temp <= 0:
                nxt = mx.argmax(logits, axis=-1)  # greedy
            else:
                probs = mx.softmax(logits / temp, axis=-1)
                nxt = _top_p_sample(probs, top_p, key)
                key, _ = mx.random.split(key)
            tok = int(nxt.item())
            if eos_id is not None and tok == eos_id:
                break
            out.append(tok)
            logits = self(mx.array([[tok]], dtype=mx.int32), cache)[:, -1, :]
        return out


def _top_p_sample(probs: mx.array, top_p: float, key: mx.array) -> mx.array:
    """Nucleus sampling over a (1, vocab) probability row."""
    if top_p >= 1.0:
        return mx.random.categorical(mx.log(probs), key=key)
    order = mx.argsort(-probs, axis=-1)
    sorted_p = mx.take_along_axis(probs, order, axis=-1)
    cum = mx.cumsum(sorted_p, axis=-1)
    # Keep tokens up to and including the one that crosses top_p: shifting the
    # cumulative sum means the first token is always kept even if it alone
    # exceeds p (otherwise a confident model could have nothing to sample).
    keep = (cum - sorted_p) < top_p
    sorted_p = mx.where(keep, sorted_p, 0.0)
    sorted_p = sorted_p / sorted_p.sum(axis=-1, keepdims=True)
    pick = mx.random.categorical(mx.log(sorted_p + 1e-20), key=key)
    return mx.take_along_axis(order, pick[:, None], axis=-1)[:, 0]


def count_params(model: Model) -> int:
    from mlx.utils import tree_flatten

    return sum(v.size for _, v in tree_flatten(model.parameters()))


# --------------------------------------------------------------------------
# Smoke test: `uv run python -m src.model`
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import time

    from src.config import load_config

    ap = argparse.ArgumentParser(description="MLX transformer smoke test")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    mx.random.seed(0)
    t0 = time.time()
    model = Model(cfg.model)
    mx.eval(model.parameters())
    print(f"built in {time.time()-t0:.1f}s")

    got, want = count_params(model), n_params(cfg.model)
    err = abs(got - want) / want
    print(f"params: {got/1e6:.2f}M (config.n_params says {want/1e6:.2f}M, err {err*100:.3f}%)")

    B, T = 2, 128
    x = mx.random.randint(0, cfg.model.vocab_size, (B, T))
    y = mx.random.randint(0, cfg.model.vocab_size, (B, T))
    t0 = time.time()
    logits = model(x)
    mx.eval(logits)
    print(f"forward {tuple(logits.shape)} dtype={logits.dtype} in {time.time()-t0:.2f}s")
    loss = model.loss(x, y)
    mx.eval(loss)
    # A freshly initialised model is uniform over the vocab, so the loss must
    # start at ln(vocab_size).  If it doesn't, the init or the tying is broken.
    print(f"init loss {loss.item():.4f} (expect ~ln({cfg.model.vocab_size}) = "
          f"{math.log(cfg.model.vocab_size):.4f})")

    t0 = time.time()
    gen = model.generate([1, 2, 3, 4], max_new=20, temp=0.8, top_p=0.95, seed=0)
    print(f"generated 20 tokens in {time.time()-t0:.2f}s: {gen}")

    ok = err < 0.01 and abs(loss.item() - math.log(cfg.model.vocab_size)) < 0.5 and len(gen) == 20
    print("OK  model.py smoke test" if ok else "FAILED  model.py smoke test")
    sys.exit(0 if ok else 1)
