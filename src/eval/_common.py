"""Shared plumbing for the final eval suite: checkpoint loading + tokenizer.

Both `vibes.py` and `run_lm_eval.py` need to turn a checkpoint directory into a
live `src.model.Model`, so that lives here once.  Nothing in this package writes
anywhere except `runs/`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Config, DataConfig, ModelConfig, TrainConfig  # noqa: E402

DEFAULT_TOKENIZER = REPO_ROOT / "data" / "tokenized" / "tokenizer.json"
EOS_TOKEN = "<|endoftext|>"
EOS_ID = 0


# --------------------------------------------------------------------------
# Checkpoint
# --------------------------------------------------------------------------


def load_ckpt_config(ckpt: Path) -> Config:
    """Rebuild the Config dataclass from the checkpoint's config.json snapshot.

    We only trust the *file* — never the repo defaults — because the run may
    have been trained with an overridden config and the eval must describe the
    weights it actually loaded.
    """
    raw = json.loads((ckpt / "config.json").read_text())
    cfg = Config(
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
        data=DataConfig(**raw["data"]),
        run_name=raw.get("run_name", "run01"),
    )
    return cfg


def ckpt_state(ckpt: Path) -> dict:
    p = ckpt / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def resolve_ckpt(path: str | Path) -> Path:
    """Accept a checkpoint dir, a `ckpt/latest` symlink, or a run dir."""
    p = Path(path).expanduser().resolve()
    if (p / "model.safetensors").exists():
        return p
    for cand in (p / "ckpt" / "latest", p / "latest"):
        if cand.exists() and (cand.resolve() / "model.safetensors").exists():
            return cand.resolve()
    raise SystemExit(f"no model.safetensors under {path}")


def run_dir_for(ckpt: Path) -> Path:
    """`runs/<run>/ckpt/step_N` -> `runs/<run>`; falls back to the ckpt itself."""
    for parent in ckpt.parents:
        if parent.name == "ckpt":
            return parent.parent
    return ckpt


def load_mlx_model(ckpt: Path):
    """Return (model, cfg).  Mirrors MLXTrainer.load() in src/train.py."""
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    from src.model import Model

    cfg = load_ckpt_config(ckpt)
    model = Model(cfg.model)
    weights = mx.load(str(ckpt / "model.safetensors"))
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model, cfg


# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------


def load_tokenizer(path: str | Path | None = None):
    from tokenizers import Tokenizer as HFTok

    p = Path(path) if path else DEFAULT_TOKENIZER
    if not p.exists():
        raise SystemExit(f"tokenizer not found: {p}")
    return HFTok.from_file(str(p))


def read_prompts(path: Path) -> list[str]:
    """One prompt per line; `#` comments and blank lines skipped (train.py rule)."""
    if not path.exists():
        raise SystemExit(f"prompt file not found: {path}")
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out
