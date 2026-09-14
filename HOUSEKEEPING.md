# Housekeeping — things to undo/rotate when the run is over

## After the pretraining run completes
- [ ] **Retire the battery thermostat** (only needed because the 96W charger can't cover full training draw):
  ```bash
  launchctl unload ~/Library/LaunchAgents/com.llmmod.battery-thermostat.plist
  rm ~/Library/LaunchAgents/com.llmmod.battery-thermostat.plist
  sudo rm /etc/sudoers.d/pmset          # removes Claude's passwordless pmset grant
  ```
- [ ] **Rotate the Hugging Face token** — the current one (`creation`) was pasted into a Claude chat in plain text: huggingface.co → Settings → Access Tokens → invalidate & create new, then `uv run hf auth login` with the new one.
- [ ] Power mode back to your preference (thermostat may leave it on High or Automatic): System Settings → Battery, or `sudo pmset -c powermode 2`.
- [ ] Re-enable **Optimized Battery Charging** habits as you like; battery hit 6% once during the run (cycle count was 7, health Normal — no lasting harm).

## Optional / whenever
- [ ] **Buy a 100–120W USB-C PD adapter** (~$60–80) — the 96W brick is 1–4W short of sustained training draw; that gap caused the whole battery saga. If you go 140W, you need a 240W-rated (5A e-marked) cable or it silently caps at 100W.
- [ ] Delete `runs/demo/` and `runs/smoke/` (dashboard/test leftovers) when no longer useful.
- [ ] `data/raw/` (~25GB) can be deleted after the run — everything is reproducible via `scripts/data_pipeline.sh`; keep `data/tokenized/` (~10GB) if you plan more sessions or fine-tunes.

## Standing infrastructure (keep while training/fine-tuning continues)
- Trainer supervisor: `scripts/run_session.sh` (pidfile `runs/run01/supervisor.pid`)
- Dashboard: `scripts/dashboard.sh start|stop|status` → http://127.0.0.1:8471
- Milestone backups: `scripts/backup_milestones.py` → private HF repo `skeggsguy/llm-mod-checkpoints` (1B-token checkpoints)
- Mint a finished model anytime: `uv run python src/train.py --config configs/session.toml --decay-now`
