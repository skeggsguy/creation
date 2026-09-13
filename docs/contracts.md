# Inter-component contracts

Every component (trainer, dashboard, supervisor, eval) codes against this file.
Change it only by agreement at the table.

## Paths

- Run directory: `runs/<run_name>/` containing `ckpt/`, `metrics.jsonl`, `samples.jsonl`, `heartbeat`, `session.log`
- Tokenized corpus: `data/tokenized/<domain>_train_####.bin` (uint16, contiguous token ids) + `data/tokenized/<domain>_val.bin` + `data/tokenized/meta.json` (tokenizer path, token counts per domain)
- Tokenizer: `data/tokenized/tokenizer.json` (HF tokenizers format)

## metrics.jsonl (append-only; one JSON object per line; dashboard tails it)

Train step record (every ~10s of steps, aggregated):
```json
{"type":"train","t":"2026-09-13T10:00:00Z","session":1,"step":1200,"tokens":78643200,"loss":4.31,"lr":0.0003,"tps":8123.5,"mem_gb":41.2}
```
Validation record:
```json
{"type":"val","t":"...","session":1,"step":1500,"tokens":98304000,"losses":{"general":4.1,"textbooks":3.9,"philosophy":4.4,"scifi":4.2,"psych":4.3,"comedy":4.5,"haiku":5.0}}
```
Event record (`kind` ∈ checkpoint_saved | resume | nan_rollback | session_start | session_end | crash_restart):
```json
{"type":"event","t":"...","session":1,"kind":"checkpoint_saved","detail":"step 1500"}
```

## samples.jsonl

```json
{"t":"...","step":1500,"tokens":98304000,"prompt":"The meaning of a good life is","text":"..."}
```
Prompts come from `prompts/vibes.txt` (one per line, `#` comments allowed), sampled with fixed seed.

## heartbeat

Trainer touches `runs/<run_name>/heartbeat` at least every 60s while training.
Supervisor / monitors treat mtime older than 5 min as a stall.

## Checkpoints

`runs/<run_name>/ckpt/step_<N>/` — model weights (safetensors), optimizer state,
RNG state, config snapshot, `state.json` (step, tokens, session). Written atomically:
build in `ckpt/.tmp/` then rename. Keep last 3 + every 1B-token milestone.
`ckpt/latest` is a symlink to the newest complete checkpoint.

## Dashboard

`dashboard/server.py` (stdlib only) serves on **http://127.0.0.1:8471**:
- `GET /` → index.html
- `GET /api/metrics?after=<N>` → `{"run", "lines": [...], "next": M}`. `after` is a
  **parsed-record index**, not a raw file line (corrupt/blank lines are skipped);
  clients must pass back the `next` they were given.
- `GET /api/samples?n=20` → `{"run", "samples": [...last n records...]}`
- `GET /api/status` → `{run_name, heartbeat_age_s, latest_ckpt_step, disk_free_gb,
  tokens_total_target}`. Unknowns are `null`, never 0.
- `tokens_total_target` is hardcoded in server.py (deliberately decoupled from src/);
  if calibration ever changes TrainConfig.total_tokens, update it there too.

## Supervisor ↔ trainer CLI surface

Supervisor (`scripts/run_session.sh`) invokes exactly:
`caffeinate -dims $TRAINER_CMD --config <path> --max-hours <float> [--halve-lr]`
- `TRAINER_CMD` env override, default `uv run python src/train.py`.
- `--max-hours` is the REMAINING budget (recomputed per restart), a float.
- `train.py` must checkpoint and exit ≤120s after SIGTERM (then it gets SIGKILL).
- `run_name` must stay a top-level double-quoted key in the session TOML (parsed by sed).
- Trainer must flush after every metrics.jsonl / samples.jsonl write.
- Supervisor owns `runs/<run>/supervisor.pid` and `session.log`; the trainer logs
  session_start/session_end as `event` records in metrics.jsonl instead.

## Exit codes (train.py)

0 = clean finish/stop requested · 3 = NaN/loss-spike rollback requested (supervisor
restarts from latest ckpt with `--halve-lr`) · 130/143 (SIGINT/SIGTERM) = stop, not a
crash · anything else = crash (supervisor restarts, max 3 times, 10s backoff).
