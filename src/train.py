"""The training loop.

WHAT THIS FILE IS RESPONSIBLE FOR
=================================
Everything that turns "a model" into "a run you can leave alone overnight and
trust in the morning":

  * the step itself (forward, backward, clip, AdamW update, LR from the WSD
    schedule),
  * mixed precision (fp32 master weights + optimizer state, bf16 compute),
  * durability (atomic checkpoints, auto-resume, deterministic data stream),
  * observability (metrics.jsonl, heartbeat, per-domain val, vibe samples),
  * and failure handling (NaN / loss-spike detection with a rollback exit code
    the supervisor understands).

WHY THE LOOP LOOKS LIKE THIS ON AN M5 MAX
=========================================
* **No gradient accumulation.**  Accumulation exists to simulate a big batch on
  a small-memory GPU.  We have one device and 128GB of unified memory, so we
  just use the biggest batch that runs at full speed.  Accumulation would only
  add kernel launches and sequential dependencies.
  (Measured reality check: throughput is flat from batch 4 to batch 16 and then
  *falls*, because activation memory — not parameters — is what fills the
  machine: ~1.9GB per batch row at seq 1024.  So "biggest batch that fits" turns
  out to mean ~16, not 64.  calibrate.py finds the exact number.)
* **One fused, compiled step.**  `mx.compile` fuses the elementwise parts of the
  optimizer and the activations into far fewer kernels — measured +30% here.
  `torch.compile` is attempted too and silently skipped if it fails on MPS.
* **Sync once per step.**  We read the loss back to the CPU once per step to
  drive the NaN guard.  At ~1s/step that costs nothing and keeps the guard
  simple and immediate.

THE TWO BACKENDS
================
`MLXTrainer` and `TorchTrainer` implement the same tiny interface
(`train_step`, `eval_loss`, `generate`, `save`, `load`, `mem_gb`), so the loop
below is written exactly once and is backend-agnostic.  `calibrate.py` decides
which one a given session uses; `--backend` overrides it.

Because both `model.py` and `model_torch.py` use identical parameter names, the
`model.safetensors` inside a checkpoint is *portable between backends* — you
can calibrate again mid-run and switch frameworks without losing progress
(optimizer state is backend-specific and would restart, so only do that at a
session boundary).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import sys
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, RUNS_DIR, load_config, n_params  # noqa: E402
from src.data_loader import Corpus, TrainStream, val_batches  # noqa: E402
from src.schedule import decay_end_tokens, lr_at, phase_at  # noqa: E402

# --- tuning knobs that are *not* hyperparameters (they don't affect the model,
# only how often we stop to look at it).  Kept here rather than in config.py so
# the config file stays a description of the experiment. --------------------
METRICS_EVERY_S = 10.0        # aggregate + append a train record this often
HEARTBEAT_EVERY_S = 30.0      # contract says <= 60s; halve it for safety margin
VAL_BATCHES_PER_DOMAIN = 8    # at batch 16 that is ~130k held-out tokens per
                              # domain — the standard error of the mean loss is
                              # then well under 0.01, far finer than the changes
                              # we care about — for ~10s every val_every_tokens
SAMPLE_MAX_NEW = 80           # tokens per vibe prompt
SAMPLE_TEMP = 0.8
SAMPLE_TOP_P = 0.95
GUARD_MIN_HISTORY = 20        # steps of history before the spike guard arms
GUARD_WINDOW = 100            # trailing-mean window
GUARD_SPIKE_FACTOR = 2.0      # loss > factor * trailing mean => rollback
ADAM_EPS = 1e-8
PEAK_FLOPS = 60e12            # M5 Max bf16 peak; only used to print MFU in the log
                              # (calibrate.py has its own copy — it must stay
                              # importable without pulling in the trainer)

EXIT_OK = 0
EXIT_ROLLBACK = 3             # docs/contracts.md: supervisor restarts w/ halved LR


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ==========================================================================
# Backend: MLX
# ==========================================================================


class MLXTrainer:
    """MLX training state: model, AdamW, and one compiled step function."""

    name = "mlx"

    def __init__(self, cfg: Config, compile_step: bool = True):
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten, tree_map_with_path

        from src.model import Model

        self.mx, self.nn, self.optim = mx, nn, optim
        self._tree_flatten = tree_flatten
        self.cfg = cfg
        mx.random.seed(cfg.train.seed)
        self.model = Model(cfg.model)
        mx.eval(self.model.parameters())

        # AdamW with weight_decay=0: we apply *decoupled* weight decay ourselves
        # (below) because MLX's optimizer takes a single scalar for every
        # parameter and we need to exempt norms and embeddings.
        self.opt = optim.AdamW(
            learning_rate=cfg.train.peak_lr,
            betas=[cfg.train.adam_beta1, cfg.train.adam_beta2],
            eps=ADAM_EPS,
            weight_decay=0.0,
            bias_correction=True,   # match PyTorch's AdamW exactly
        )

        self.decay_paths = {p for p, w in tree_flatten(self.model.parameters()) if _wants_wd(p, w.ndim)}
        self._want_compile = compile_step
        self._compiled = compile_step
        self._step = None  # built lazily; see _build_step
        self.n_params = sum(v.size for _, v in tree_flatten(self.model.parameters()))

    def _build_step(self):
        """Create (and optionally compile) the fused train step.

        Built lazily — and thrown away by `load()` — because `mx.compile`
        captures the *identity* of the state containers it is given.  Restoring
        a checkpoint rebinds `optimizer.state`, so a step function compiled
        before the restore would keep writing into the orphaned pre-resume
        arrays ("Attempting to eval an array without a primitive").  Rebuilding
        after any state swap is the cheap, obviously-correct fix.
        """
        mx, nn, optim = self.mx, self.nn, self.optim
        from mlx.utils import tree_map_with_path

        cfg = self.cfg
        wd = cfg.train.weight_decay
        clip = cfg.train.grad_clip
        model, opt = self.model, self.opt
        decay_paths = self.decay_paths

        def loss_fn(m, x, y):
            return m.loss(x, y)

        grad_fn = nn.value_and_grad(model, loss_fn)

        def _step(x, y):
            loss, grads = grad_fn(model, x, y)
            # Global-norm clipping: rescale *all* gradients by one factor when
            # their combined norm exceeds grad_clip.  Per-tensor clipping would
            # change the direction of the update; global clipping only changes
            # its length, which is the point — we want to survive a bad batch,
            # not to distort every good one.
            grads, gnorm = optim.clip_grad_norm(grads, clip)
            if wd > 0:
                # Decoupled weight decay, spelled out: shrink the weight itself
                # by (1 - lr*wd) *before* the Adam update, instead of adding
                # wd*w to the gradient.  Adding it to the gradient (classic L2)
                # would let Adam's per-parameter normalisation rescale the
                # decay, so weights with small gradients would decay far more
                # than intended.  Norms and embeddings are exempt: shrinking a
                # norm's gain fights the normalisation, and shrinking embeddings
                # of rare tokens (which get gradients only occasionally) would
                # decay them toward zero between appearances.
                lr = opt.learning_rate
                model.update(tree_map_with_path(
                    lambda path, p: p * (1.0 - lr * wd) if path in decay_paths else p,
                    model.parameters(),
                ))
            opt.update(model, grads)
            return loss, gnorm

        if not self._want_compile:
            return _step
        from functools import partial
        # `inputs`/`outputs` tell mx.compile which mutable state the graph
        # reads and writes, so changing the LR (an array inside opt.state)
        # does not trigger a recompile.
        state = [model.state, opt.state, mx.random.state]
        return partial(mx.compile, inputs=state, outputs=state)(_step)

    # -- interface -------------------------------------------------------

    def set_lr(self, lr: float) -> None:
        self.opt.learning_rate = lr

    def train_step(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        mx = self.mx
        if self._step is None:
            self._step = self._build_step()
        loss, gnorm = self._step(mx.array(x), mx.array(y))
        mx.eval(loss, gnorm, self.model.state, self.opt.state)
        return float(loss.item()), float(gnorm.item())

    def eval_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        mx = self.mx
        loss = self.model.loss(mx.array(x), mx.array(y))
        mx.eval(loss)
        return float(loss.item())

    def generate(self, prompt_ids, max_new, temp, top_p, seed):
        return self.model.generate(prompt_ids, max_new, temp, top_p, seed)

    def mem_gb(self) -> float:
        return self.mx.get_peak_memory() / (1 << 30)

    def rng_state(self) -> dict:
        # docs/contracts.md asks for RNG state in the checkpoint.  MLX exposes
        # its key for reading but has no setter, so this is a record of where
        # the stream was, not a restore point.  That costs us nothing: after
        # initialisation *nothing* in training draws a random number (no
        # dropout), the data stream is seeded from (seed, step), and sampling
        # passes its seed explicitly.  Determinism here is structural, not
        # RNG-state-dependent — which is why test_resume reproduces a run
        # bit-exactly across processes.
        return {"mlx_key": [int(v) for v in self.mx.random.state[0].tolist()]}

    def set_rng_state(self, d: dict) -> None:
        return  # see rng_state(): MLX provides no way to restore a key

    def save(self, d: Path) -> None:
        from mlx.utils import tree_flatten
        self.mx.save_safetensors(
            str(d / "model.safetensors"), dict(tree_flatten(self.model.parameters()))
        )
        self.mx.save_safetensors(
            str(d / "optim.safetensors"), dict(tree_flatten(self.opt.state))
        )

    def load(self, d: Path) -> None:
        from mlx.utils import tree_unflatten
        weights = self.mx.load(str(d / "model.safetensors"))
        self.model.update(tree_unflatten(list(weights.items())))
        op = d / "optim.safetensors"
        if op.exists():
            self.opt.state = tree_unflatten(list(self.mx.load(str(op)).items()))
        self.mx.eval(self.model.parameters(), self.opt.state)
        self._step = None  # state containers changed identity — recompile


# ==========================================================================
# Backend: PyTorch / MPS
# ==========================================================================


class TorchTrainer:
    name = "torch"

    def __init__(self, cfg: Config, compile_step: bool = True):
        import torch

        from src.model_torch import Model, best_device

        self.torch = torch
        self.cfg = cfg
        self.device = best_device()
        torch.manual_seed(cfg.train.seed)
        self.model = Model(cfg.model).to(self.device)
        self.raw_model = self.model  # torch.compile wraps; save from the original

        # Two parameter groups so weight decay skips norms and embeddings — same
        # reasoning as the MLX branch above.
        decay, no_decay = [], []
        for pname, p in self.model.named_parameters():
            (decay if _wants_wd(pname, p.dim()) else no_decay).append(p)
        self.opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": cfg.train.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg.train.peak_lr,
            betas=(cfg.train.adam_beta1, cfg.train.adam_beta2),
            eps=ADAM_EPS,
        )
        self.clip = cfg.train.grad_clip
        self.n_params = sum(p.numel() for p in self.model.parameters())

        # NOTE: we compile `raw_model.loss`, not the module.  `torch.compile(m)`
        # only intercepts `m(...)` / `m.forward(...)`; any *other* method reached
        # through the wrapper falls straight through to the original module and
        # runs eager.  Since the training step calls `.loss()`, compiling the
        # module would have been a silent no-op — the kind of mistake that shows
        # up as "compile made no difference" rather than as an error.
        self._loss_fn = self.raw_model.loss
        self._compiled = False
        if compile_step:
            # torch.compile support on MPS is patchy; a silent fallback is the
            # contract here — a slower run beats a crashed run.
            try:
                self._loss_fn = torch.compile(self.raw_model.loss)
                self._compiled = True
            except Exception as e:  # pragma: no cover - depends on torch build
                print(f"[torch] torch.compile unavailable ({e}); running eager")

    def _uncompile(self, why: str) -> None:
        print(f"[torch] torch.compile failed at runtime ({why}); falling back to eager")
        self._loss_fn = self.raw_model.loss
        self._compiled = False

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    def train_step(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        torch = self.torch
        xb = torch.from_numpy(x).to(self.device, dtype=torch.long)
        yb = torch.from_numpy(y).to(self.device, dtype=torch.long)
        self.opt.zero_grad(set_to_none=True)
        try:
            loss = self._loss_fn(xb, yb)
        except Exception as e:
            if not self._compiled:
                raise
            self._uncompile(repr(e)[:120])
            loss = self._loss_fn(xb, yb)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.clip)
        self.opt.step()
        return float(loss.item()), float(gnorm.item())

    def eval_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        torch = self.torch
        with torch.no_grad():
            xb = torch.from_numpy(x).to(self.device, dtype=torch.long)
            yb = torch.from_numpy(y).to(self.device, dtype=torch.long)
            return float(self.raw_model.loss(xb, yb).item())

    def generate(self, prompt_ids, max_new, temp, top_p, seed):
        return self.raw_model.generate(prompt_ids, max_new, temp, top_p, seed)

    def mem_gb(self) -> float:
        if self.device.type == "mps":
            return self.torch.mps.driver_allocated_memory() / (1 << 30)
        return 0.0

    def rng_state(self) -> dict:
        """See MLXTrainer.rng_state — torch, unlike MLX, can restore it."""
        out = {"cpu": self.torch.get_rng_state().tolist()}
        try:
            out["mps"] = self.torch.mps.get_rng_state().tolist()
        except Exception:
            pass
        return out

    def set_rng_state(self, d: dict) -> None:
        torch = self.torch
        try:
            if "cpu" in d:
                torch.set_rng_state(torch.tensor(d["cpu"], dtype=torch.uint8))
            if "mps" in d:
                torch.mps.set_rng_state(torch.tensor(d["mps"], dtype=torch.uint8))
        except Exception as e:  # pragma: no cover
            print(f"[torch] could not restore RNG state ({e}); continuing")

    def save(self, d: Path) -> None:
        from safetensors.torch import save_file
        sd = {k: v.detach().to("cpu").contiguous() for k, v in self.raw_model.state_dict().items()}
        save_file(sd, str(d / "model.safetensors"))
        self.torch.save(self.opt.state_dict(), d / "optim.pt")

    def load(self, d: Path) -> None:
        from safetensors.torch import load_file
        sd = load_file(str(d / "model.safetensors"))
        self.raw_model.load_state_dict(sd, strict=True)
        op = d / "optim.pt"
        if op.exists():
            self.opt.load_state_dict(self.torch.load(op, map_location=self.device, weights_only=False))


def _wants_wd(param_path: str, ndim: int) -> bool:
    """Weight decay applies to matmul weights only.

    Excluded: RMSNorm gains (1-D) and the token embedding table.  See the long
    comment in MLXTrainer for why.
    """
    if ndim < 2:
        return False
    return "tok_emb" not in param_path


def make_trainer(cfg: Config, backend: str | None = None, compile_step: bool = True):
    b = (backend or cfg.train.backend).lower()
    if b == "mlx":
        return MLXTrainer(cfg, compile_step)
    if b in ("torch", "pytorch", "mps"):
        return TorchTrainer(cfg, compile_step)
    raise ValueError(f"unknown backend {b!r} (expected 'mlx' or 'torch')")


# ==========================================================================
# Run directory plumbing (metrics / heartbeat / checkpoints)
# ==========================================================================


class RunDir:
    """Owns runs/<run_name>/ and every file docs/contracts.md promises is there."""

    def __init__(self, run_name: str, runs_dir: Path | None = None):
        self.root = (runs_dir or RUNS_DIR) / run_name
        self.ckpt = self.root / "ckpt"
        self.tmp = self.ckpt / ".tmp"
        self.root.mkdir(parents=True, exist_ok=True)
        self.ckpt.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.root / "metrics.jsonl"
        self.samples_path = self.root / "samples.jsonl"
        self.heartbeat_path = self.root / "heartbeat"
        self.log_path = self.root / "session.log"
        self._log = open(self.log_path, "a", buffering=1)
        self._last_beat = 0.0

    # -- logging ---------------------------------------------------------

    def log(self, msg: str) -> None:
        line = f"[{utcnow()}] {msg}"
        print(line, flush=True)
        self._log.write(line + "\n")

    def _append(self, path: Path, obj: dict) -> None:
        # Append + flush (not fsync): the dashboard tails these files, so
        # visibility matters every line; durability only matters at checkpoints.
        with open(path, "a") as f:
            f.write(json.dumps(obj, separators=(",", ":")) + "\n")

    def metric(self, obj: dict) -> None:
        self._append(self.metrics_path, obj)

    def event(self, session: int, kind: str, detail: str = "") -> None:
        self.metric({"type": "event", "t": utcnow(), "session": session,
                     "kind": kind, "detail": detail})
        self.log(f"event {kind}: {detail}")

    def sample(self, obj: dict) -> None:
        self._append(self.samples_path, obj)

    def beat(self, force: bool = False) -> None:
        """Touch the heartbeat file.  The supervisor reads its mtime."""
        now = time.time()
        if force or now - self._last_beat >= HEARTBEAT_EVERY_S:
            self.heartbeat_path.touch()   # creates it, or bumps mtime
            self._last_beat = now

    # -- checkpoints -----------------------------------------------------

    def save_checkpoint(self, trainer, cfg: Config, state: dict, milestone: bool) -> Path:
        """Write ckpt/step_<N>/ atomically, then repoint ckpt/latest.

        Atomic = build the whole directory under `.tmp/` and `os.replace` it
        into place.  A checkpoint is several files; if we wrote them in place
        and lost power halfway, `latest` would point at a directory holding a
        model from step N and an optimizer from step N-1 — silently wrong, and
        far worse than no checkpoint at all.  Rename is atomic on APFS, so
        `ckpt/step_N` either does not exist or is complete.
        """
        if self.tmp.exists():
            shutil.rmtree(self.tmp)
        self.tmp.mkdir(parents=True)
        trainer.save(self.tmp)
        (self.tmp / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        (self.tmp / "state.json").write_text(json.dumps(state, indent=2))
        if milestone:
            (self.tmp / "MILESTONE").write_text(f"{state['tokens']}\n")

        dest = self.ckpt / f"step_{state['step']}"
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(self.tmp, dest)

        # Repoint `latest` atomically too: symlink to a temp name, then rename
        # over the old one.  A relative target keeps the run dir movable.
        link_tmp = self.ckpt / ".latest.tmp"
        if link_tmp.exists() or link_tmp.is_symlink():
            link_tmp.unlink()
        os.symlink(dest.name, link_tmp)
        os.replace(link_tmp, self.ckpt / "latest")
        self._prune()
        return dest

    def _prune(self, keep_last: int = 3) -> None:
        """Keep the last `keep_last` checkpoints plus every 1B-token milestone.

        Each checkpoint of a 303M model is ~3.6GB (fp32 weights + two Adam
        moments), so unbounded retention fills a disk in a day.  Milestones are
        kept forever because they are the artefacts we compare across sessions.
        """
        dirs = sorted(
            (p for p in self.ckpt.glob("step_*") if p.is_dir()),
            key=lambda p: int(p.name.split("_")[1]),
        )
        doomed = [p for p in dirs[:-keep_last] if not (p / "MILESTONE").exists()]
        for p in doomed:
            shutil.rmtree(p, ignore_errors=True)
            self.log(f"pruned old checkpoint {p.name}")

    def latest_checkpoint(self) -> Path | None:
        link = self.ckpt / "latest"
        if link.exists():
            return link.resolve()
        dirs = sorted((p for p in self.ckpt.glob("step_*") if (p / "state.json").exists()),
                      key=lambda p: int(p.name.split("_")[1]))
        return dirs[-1] if dirs else None


# ==========================================================================
# Vibe sampling
# ==========================================================================


def load_vibe_prompts(path: Path) -> list[str]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


class Tokenizer:
    """Thin wrapper so a missing tokenizer degrades to 'no vibe samples'.

    The tokenizer is produced by the data pipeline (another component), so the
    trainer must not hard-fail when it is absent — e.g. on synthetic test data.
    """

    def __init__(self, path: Path | None):
        self.tk = None
        if path and Path(path).exists():
            try:
                from tokenizers import Tokenizer as HFTok
                self.tk = HFTok.from_file(str(path))
            except Exception as e:  # pragma: no cover
                print(f"[train] tokenizer load failed ({e}); vibe sampling disabled")

    @property
    def ok(self) -> bool:
        return self.tk is not None

    def encode(self, s: str) -> list[int]:
        return self.tk.encode(s).ids

    def decode(self, ids) -> str:
        return self.tk.decode(list(ids))


# ==========================================================================
# The loop
# ==========================================================================


_STOP = {"flag": False, "why": ""}


def _install_signal_handlers(rd: RunDir) -> None:
    def handler(signum, _frame):
        # Do NOT checkpoint from inside the handler: it can fire in the middle
        # of a GPU op or a file write.  Set a flag; the loop notices after the
        # current step and shuts down cleanly.
        _STOP["flag"] = True
        _STOP["why"] = signal.Signals(signum).name
        rd.log(f"caught {_STOP['why']} — will checkpoint and exit after this step")

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def train(args) -> int:
    cfg: Config = load_config(args.config)
    if args.backend:
        cfg.train.backend = args.backend
    if args.batch_size:
        cfg.train.batch_size = args.batch_size

    rd = RunDir(cfg.run_name, Path(args.runs_dir) if args.runs_dir else None)
    _install_signal_handlers(rd)

    # ---- resume decision ------------------------------------------------
    # Default is auto-resume: an overnight run that is restarted by the
    # supervisor must never silently start from scratch and overwrite a good
    # run's metrics.  `--fresh` is the explicit opt-out.
    ckpt = None if args.fresh else rd.latest_checkpoint()
    resume_state: dict = {}
    if ckpt is not None and (ckpt / "state.json").exists():
        resume_state = json.loads((ckpt / "state.json").read_text())
    elif args.resume and not args.fresh:
        rd.log("--resume requested but no checkpoint found; starting fresh")

    step = int(resume_state.get("step", 0))
    tokens = int(resume_state.get("tokens", 0))
    session = int(resume_state.get("session", 0)) + 1
    lr_scale = float(resume_state.get("lr_scale", 1.0))
    decay_start = resume_state.get("decay_start")
    wall_before = float(resume_state.get("wall_seconds", 0.0))

    # --halve-lr is how the supervisor responds to a rollback (exit 3).  We
    # persist the accumulated scale in state.json so a *later* plain resume
    # keeps the reduced LR instead of jumping back to peak and re-diverging.
    if args.halve_lr:
        lr_scale *= 0.5
    cfg.train.peak_lr *= lr_scale

    # --decay-now: start the WSD anneal here, from the current token count, and
    # exit when it completes.  This is how we mint a "finished" model at any
    # point without giving up the un-decayed checkpoint.
    if args.decay_now and decay_start is None:
        decay_start = float(tokens)
    stop_at_tokens = (
        decay_end_tokens(cfg, decay_start) if decay_start is not None else float(cfg.train.total_tokens)
    )

    # ---- data -----------------------------------------------------------
    corpus = Corpus(cfg.data, require_val=True)
    stream = TrainStream(corpus, cfg.train.batch_size, cfg.model.seq_len, cfg.train.seed)
    tok_per_step = stream.tokens_per_batch
    # meta.json's tokenizer path may be relative to the corpus dir (it is
    # written by the data pipeline, which is a different component).
    tok_path = Path(corpus.meta.get("tokenizer") or "tokenizer.json")
    if not tok_path.is_absolute():
        tok_path = corpus.dir / tok_path
    tokenizer = Tokenizer(tok_path)
    vibes = load_vibe_prompts(Path(args.prompts))

    # ---- model ----------------------------------------------------------
    t0_build = time.time()
    trainer = make_trainer(cfg, cfg.train.backend, compile_step=not args.no_compile)
    if ckpt is not None and resume_state:
        trainer.load(ckpt)
        if resume_state.get("rng"):
            trainer.set_rng_state(resume_state["rng"])

    rd.log(f"=== session {session} | backend={trainer.name} | run={cfg.run_name} ===")
    rd.log(corpus.summary())
    for w in corpus.epoch_warnings(cfg.train.total_tokens):
        rd.log(f"WARNING {w}")
    rd.log(f"params {trainer.n_params/1e6:.2f}M (config says {n_params(cfg.model)/1e6:.2f}M)  "
           f"batch={cfg.train.batch_size}x{cfg.model.seq_len}={tok_per_step} tok/step  "
           f"compiled={getattr(trainer, '_compiled', False)}  built in {time.time()-t0_build:.1f}s")
    if resume_state:
        rd.log(f"resumed from {ckpt} @ step {step}, {tokens/1e9:.4f}B tokens, lr_scale={lr_scale}")
        rd.event(session, "resume", f"step {step} tokens {tokens}")
    rd.event(session, "session_start",
             f"backend={trainer.name} batch={cfg.train.batch_size} peak_lr={cfg.train.peak_lr:g}"
             + (f" decay_now@{decay_start:.0f}" if decay_start is not None else ""))
    if decay_start is not None:
        rd.log(f"MINT MODE: decaying from {decay_start/1e9:.3f}B to {stop_at_tokens/1e9:.3f}B tokens")

    # ---- loop state -----------------------------------------------------
    t_start = time.time()
    deadline = t_start + args.max_hours * 3600 if args.max_hours else None
    # 0 => a record per step.  Only tests and debugging want that; it does not
    # change the record *schema*, just the cadence.
    metrics_every_s = METRICS_EVERY_S if args.metrics_every_s is None else args.metrics_every_s
    trail = deque(maxlen=GUARD_WINDOW)
    win_loss, win_steps, win_tokens = 0.0, 0, 0
    win_gnorm = 0.0
    t_window = time.time()
    next_val_tokens = _next_multiple(tokens, cfg.train.val_every_tokens)
    next_sample_tokens = _next_multiple(tokens, cfg.train.sample_every_tokens)
    last_ckpt_t = time.time()
    last_milestone = tokens // 1_000_000_000
    exit_code = EXIT_OK
    stop_reason = "reached token target"
    rd.beat(force=True)

    def snapshot() -> dict:
        return {"step": step, "tokens": tokens, "session": session,
                "backend": trainer.name, "lr_scale": lr_scale,
                "decay_start": decay_start,
                "wall_seconds": wall_before + (time.time() - t_start),
                "seed": cfg.train.seed, "batch_size": cfg.train.batch_size,
                "run_name": cfg.run_name, "t": utcnow(),
                "rng": trainer.rng_state()}

    while True:
        if tokens >= stop_at_tokens:
            stop_reason = "decay complete" if decay_start is not None else "reached token target"
            break
        if _STOP["flag"]:
            stop_reason = f"signal {_STOP['why']}"
            break
        if deadline and time.time() >= deadline:
            stop_reason = f"--max-hours {args.max_hours} elapsed"
            break
        if args.max_steps and step >= args.max_steps:
            stop_reason = f"--max-steps {args.max_steps} reached"
            break

        # --- one step ----------------------------------------------------
        lr = lr_at(tokens, cfg, decay_start)
        trainer.set_lr(lr)
        x, y, _ = stream.batch(step)
        loss, gnorm = trainer.train_step(x, y)

        # --- the guard ---------------------------------------------------
        # A NaN or a sudden loss explosion means the weights are already
        # poisoned; the *only* safe move is to write nothing and let the
        # supervisor restart us from the last good checkpoint with a halved LR
        # (exit code 3, docs/contracts.md).  Checkpointing here would overwrite
        # the good state with the broken one.
        bad = None
        if not math.isfinite(loss):
            bad = f"non-finite loss {loss}"
        elif len(trail) >= GUARD_MIN_HISTORY:
            mean = sum(trail) / len(trail)
            if loss > GUARD_SPIKE_FACTOR * mean:
                bad = f"loss {loss:.3f} > {GUARD_SPIKE_FACTOR}x trailing mean {mean:.3f}"
        if bad:
            rd.log(f"!! {bad} at step {step} (tokens {tokens}) — NOT checkpointing")
            rd.event(session, "nan_rollback", f"step {step}: {bad}")
            rd.beat(force=True)
            return EXIT_ROLLBACK

        trail.append(loss)
        step += 1
        tokens += tok_per_step
        win_loss += loss
        win_gnorm += gnorm
        win_steps += 1
        win_tokens += tok_per_step

        # --- aggregated train metric -------------------------------------
        now = time.time()
        if now - t_window >= metrics_every_s or (args.max_steps and step >= args.max_steps):
            dt = max(now - t_window, 1e-9)
            tps = win_tokens / dt
            rd.metric({"type": "train", "t": utcnow(), "session": session, "step": step,
                       "tokens": tokens, "loss": round(win_loss / win_steps, 4),
                       "lr": lr, "tps": round(tps, 1), "mem_gb": round(trainer.mem_gb(), 2)})
            # MFU = achieved FLOPs / peak FLOPs.  6*N per token is the standard
            # forward+backward estimate (2 for the forward matmuls, 4 for the
            # backward's two matmuls per weight).
            mfu = 6 * trainer.n_params * tps / PEAK_FLOPS
            rd.log(f"step {step:>7} | {tokens/1e9:7.4f}B tok | loss {win_loss/win_steps:6.4f} | "
                   f"lr {lr:.3e} [{phase_at(tokens, cfg, decay_start)}] | {tps:8.1f} tok/s | "
                   f"mfu {mfu*100:4.1f}% | gnorm {win_gnorm/win_steps:5.3f} | "
                   f"mem {trainer.mem_gb():5.2f}GB")
            win_loss = win_gnorm = 0.0
            win_steps = win_tokens = 0
            t_window = now
        rd.beat()

        # --- validation ---------------------------------------------------
        if tokens >= next_val_tokens:
            next_val_tokens = _next_multiple(tokens, cfg.train.val_every_tokens)
            run_validation(rd, trainer, corpus, cfg, session, step, tokens)

        # --- vibe samples --------------------------------------------------
        if tokens >= next_sample_tokens:
            next_sample_tokens = _next_multiple(tokens, cfg.train.sample_every_tokens)
            if vibes and tokenizer.ok:
                run_vibe_samples(rd, trainer, tokenizer, vibes, step, tokens, cfg.train.seed)

        # --- periodic checkpoint -------------------------------------------
        if time.time() - last_ckpt_t >= cfg.train.checkpoint_every_s:
            milestone = tokens // 1_000_000_000 > last_milestone
            dest = rd.save_checkpoint(trainer, cfg, snapshot(), milestone)
            last_milestone = max(last_milestone, tokens // 1_000_000_000)
            last_ckpt_t = time.time()
            rd.event(session, "checkpoint_saved", f"step {step}" + (" (1B milestone)" if milestone else ""))
            rd.beat(force=True)

    # ---- clean shutdown --------------------------------------------------
    # Flush whatever steps are sitting in the current aggregation window, so a
    # short session (a mint, a --max-hours stop) still leaves a train record
    # ending at its true final step rather than silently dropping it.
    if win_steps:
        dt = max(time.time() - t_window, 1e-9)
        rd.metric({"type": "train", "t": utcnow(), "session": session, "step": step,
                   "tokens": tokens, "loss": round(win_loss / win_steps, 4),
                   "lr": lr_at(tokens, cfg, decay_start), "tps": round(win_tokens / dt, 1),
                   "mem_gb": round(trainer.mem_gb(), 2)})

    # A completed mint is kept forever, like a 1B-token milestone: it is the
    # artefact the whole --decay-now run existed to produce.
    minted = decay_start is not None and tokens >= stop_at_tokens
    milestone = minted or tokens // 1_000_000_000 > last_milestone
    dest = rd.save_checkpoint(trainer, cfg, snapshot(), milestone)
    rd.event(session, "checkpoint_saved", f"step {step} (final)")
    rd.log(f"stopping: {stop_reason}; final checkpoint {dest.name}")
    if minted:
        rd.log("MINT COMPLETE — this checkpoint is the annealed/'finished' model "
               "(kept permanently, like a milestone)")
    rd.event(session, "session_end",
             f"{stop_reason}; step {step}, tokens {tokens}, "
             f"{(time.time()-t_start)/3600:.2f}h this session")
    rd.beat(force=True)
    return exit_code


def _next_multiple(tokens: int, every: int) -> int:
    """Next multiple of `every` strictly greater than `tokens`.

    Using absolute token milestones (rather than 'every N tokens from now')
    means val/sample cadence is identical whether a run is interrupted or not —
    one less source of drift between sessions.
    """
    if every <= 0:
        return 1 << 62
    return (tokens // every + 1) * every


def run_validation(rd: RunDir, trainer, corpus: Corpus, cfg: Config,
                   session: int, step: int, tokens: int) -> dict:
    """Per-domain held-out loss.

    Per-domain rather than one number because the whole point of the corpus mix
    is that different registers are learned at different rates: a rising
    philosophy loss while general keeps falling is a mixture problem, and an
    aggregate would hide it.
    """
    t0 = time.time()
    losses: dict[str, float] = {}
    for dom in corpus.names:
        batches = val_batches(corpus, dom, cfg.train.batch_size, cfg.model.seq_len,
                              VAL_BATCHES_PER_DOMAIN)
        if not batches:
            continue
        tot = 0.0
        for x, y in batches:
            tot += trainer.eval_loss(x, y)
            rd.beat()  # a long val pass must not look like a stall
        losses[dom] = round(tot / len(batches), 4)
    rd.metric({"type": "val", "t": utcnow(), "session": session, "step": step,
               "tokens": tokens, "losses": losses})
    rd.log("val " + "  ".join(f"{k}={v:.4f}" for k, v in losses.items())
           + f"  ({time.time()-t0:.1f}s)")
    return losses


def run_vibe_samples(rd: RunDir, trainer, tokenizer: Tokenizer, prompts: list[str],
                     step: int, tokens: int, seed: int) -> None:
    """Generate a continuation for each fixed prompt.

    Fixed prompts + fixed seed means the only thing that changes between
    checkpoints is the model, so reading two sample blocks side by side shows
    learning directly — the cheapest, highest-signal eval there is.
    """
    t0 = time.time()
    for i, p in enumerate(prompts):
        ids = tokenizer.encode(p)
        if not ids:
            continue
        out = trainer.generate(ids, SAMPLE_MAX_NEW, SAMPLE_TEMP, SAMPLE_TOP_P, seed + i)
        rd.sample({"t": utcnow(), "step": step, "tokens": tokens,
                   "prompt": p, "text": tokenizer.decode(out)})
        rd.beat()
    rd.log(f"wrote {len(prompts)} vibe samples ({time.time()-t0:.1f}s)")


# ==========================================================================


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="pretrain the model")
    ap.add_argument("--config", default="configs/session.toml")
    ap.add_argument("--resume", action="store_true",
                    help="resume from ckpt/latest (this is the DEFAULT; flag kept "
                         "for explicitness in supervisor scripts)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing checkpoint and start from scratch")
    ap.add_argument("--decay-now", action="store_true",
                    help="mint mode: run the WSD decay phase from the current "
                         "checkpoint, then exit 0")
    ap.add_argument("--halve-lr", action="store_true",
                    help="halve peak_lr (supervisor uses this after a nan_rollback)")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop (checkpoint + exit 0) after this many hours")
    ap.add_argument("--backend", default=None, choices=["mlx", "torch"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="stop after N total steps (tests and short smoke runs)")
    ap.add_argument("--runs-dir", default=None, help="override runs/ (tests)")
    ap.add_argument("--prompts", default=str(Path(__file__).resolve().parents[1] / "prompts" / "vibes.txt"))
    ap.add_argument("--no-compile", action="store_true", help="disable mx.compile/torch.compile")
    ap.add_argument("--metrics-every-s", type=float, default=None,
                    help="seconds between aggregated train records (0 = every step; tests)")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    try:
        return train(args)
    except KeyboardInterrupt:
        # SIGINT is handled above; this only fires if it lands somewhere the
        # flag cannot be honoured.  Non-zero but not 3: the supervisor will
        # treat it as a crash and restart, which is the right thing.
        print("interrupted")
        return 1


if __name__ == "__main__":
    sys.exit(main())
