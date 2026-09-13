"""Upload 1B-token milestone checkpoints to the private HF repo as they appear.

Milestones are checkpoint dirs containing a MILESTONE marker file (written by
train.py; pruning never deletes them). Uploaded ones are remembered via a
.hf_uploaded marker inside the checkpoint dir, so this is idempotent and safe
to run in a loop alongside the supervisor.

Usage:
    uv run python scripts/backup_milestones.py --run run01 --once   # single scan
    uv run python scripts/backup_milestones.py --run run01          # watch loop (30 min)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "skeggsguy/llm-mod-checkpoints"
REPO_ROOT = Path(__file__).resolve().parent.parent


def scan_and_upload(run: str) -> int:
    api = HfApi()
    ckpt_root = REPO_ROOT / "runs" / run / "ckpt"
    if not ckpt_root.exists():
        return 0
    uploaded = 0
    for d in sorted(ckpt_root.glob("step_*")):
        if not (d / "MILESTONE").exists() or (d / ".hf_uploaded").exists():
            continue
        tokens = (d / "MILESTONE").read_text().strip()
        dest = f"{run}/{d.name}"
        print(f"[backup] uploading {d.name} ({int(tokens)/1e9:.2f}B tokens) -> {REPO_ID}/{dest}", flush=True)
        try:
            api.upload_folder(folder_path=str(d), path_in_repo=dest, repo_id=REPO_ID,
                              ignore_patterns=[".hf_uploaded"],
                              commit_message=f"{run} milestone {d.name} ({tokens} tokens)")
            (d / ".hf_uploaded").write_text(time.strftime("%F %T"))
            uploaded += 1
            print(f"[backup] done {d.name}", flush=True)
        except Exception as e:  # network blips must never kill the watcher
            print(f"[backup] FAILED {d.name}: {e} (will retry next scan)", flush=True)
    return uploaded


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run01")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=1800)
    args = ap.parse_args()

    while True:
        scan_and_upload(args.run)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
