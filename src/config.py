"""Single source of truth for model / training / data configuration.

Loaded from a TOML file (configs/*.toml); any missing key falls back to the
dataclass default. All other modules import from here — nobody hardcodes
hyperparameters.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"


@dataclass
class ModelConfig:
    # ~303M params at these defaults (22 layers): 22 * 12.65M + 25.2M tied embedding
    d_model: int = 1024
    n_layers: int = 22
    n_heads: int = 16
    ffn_hidden: int = 2752       # SwiGLU hidden dim (~2.67x d_model, rounded to /64)
    vocab_size: int = 24576
    seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tied_embeddings: bool = True


@dataclass
class TrainConfig:
    backend: str = "mlx"          # "mlx" | "torch" — calibration decides
    batch_size: int = 64          # sequences per step; calibration sweeps this
    peak_lr: float = 3e-4
    warmup_tokens: int = 100_000_000
    total_tokens: int = 6_000_000_000   # Chinchilla target across all sessions
    decay_fraction: float = 0.10  # WSD: final decay phase length (only when minting)
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    grad_clip: float = 1.0
    checkpoint_every_s: int = 1800
    val_every_tokens: int = 50_000_000
    sample_every_tokens: int = 100_000_000
    seed: int = 1337


@dataclass
class DataConfig:
    tokenized_dir: str = str(DATA_DIR / "tokenized")
    # mix ratios must sum to 1.0; per decisions.md
    mix: dict = field(default_factory=lambda: {
        "general": 0.55,      # FineWeb-Edu
        "textbooks": 0.12,    # Cosmopedia-v2, topic-tilted
        "philosophy": 0.09,   # Gutenberg
        "scifi": 0.09,        # Gutenberg pulp era
        "psych": 0.07,        # fastText-filtered FineWeb-Edu
        "comedy": 0.07,       # Gutenberg humor
        "haiku": 0.01,
    })
    max_domain_epochs: float = 4.0  # themed slices may repeat up to this many times
    val_tokens_per_domain: int = 2_000_000


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    run_name: str = "run01"


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    raw = tomllib.loads(Path(path).read_text())
    for section_name, section_cls in (("model", ModelConfig), ("train", TrainConfig), ("data", DataConfig)):
        if section_name in raw:
            section = getattr(cfg, section_name)
            valid = {f.name for f in fields(section_cls)}
            for k, v in raw[section_name].items():
                if k not in valid:
                    raise KeyError(f"unknown key [{section_name}].{k}")
                setattr(section, k, v)
    if "run_name" in raw:
        cfg.run_name = raw["run_name"]
    return cfg


def n_params(m: ModelConfig) -> int:
    """Parameter count (tied embeddings counted once)."""
    per_layer = 4 * m.d_model * m.d_model + 3 * m.d_model * m.ffn_hidden + 2 * m.d_model
    emb = m.vocab_size * m.d_model
    head = 0 if m.tied_embeddings else emb
    return m.n_layers * per_layer + emb + head + m.d_model


if __name__ == "__main__":
    c = load_config()
    print(f"params: {n_params(c.model)/1e6:.1f}M")
