#!/usr/bin/env bash
# Session supervisor: runs the trainer under caffeinate, restarts it per the
# exit-code contract in docs/contracts.md, logs everything to the run dir.
#
#   scripts/run_session.sh [--hours 24] [--config configs/session.toml] [--run NAME]
#
# Exit codes from the trainer (docs/contracts.md):
#   0     clean finish / stop requested -> we stop
#   3     NaN / loss-spike rollback     -> restart immediately with --halve-lr
#   other crash                         -> restart, at most MAX_CRASH_RESTARTS times
#
# Override the trainer command for testing with the TRAINER_CMD env var, e.g.
#   TRAINER_CMD="bash /tmp/stub.sh" scripts/run_session.sh --hours 0.01

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

HOURS=24
CONFIG="configs/session.toml"
RUN_NAME=""
MAX_CRASH_RESTARTS=3
SHUTDOWN_WAIT_S=120
TRAINER_CMD="${TRAINER_CMD:-uv run python src/train.py}"
CAFFEINATE="${CAFFEINATE:-caffeinate -dims}"

usage() { sed -n '2,12p' "$0"; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --hours)  HOURS="${2:-}"; shift 2 ;;
    --config) CONFIG="${2:-}"; shift 2 ;;
    --run)    RUN_NAME="${2:-}"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

# run_name comes from the config (a bare top-level `run_name = "..."` line),
# unless overridden on the command line.
if [ -z "$RUN_NAME" ]; then
  if [ -f "$REPO_ROOT/$CONFIG" ]; then
    RUN_NAME="$(sed -n 's/^[[:space:]]*run_name[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$REPO_ROOT/$CONFIG" | head -1)"
  fi
  RUN_NAME="${RUN_NAME:-run01}"
fi

RUN_DIR="$REPO_ROOT/runs/$RUN_NAME"
PIDFILE="$RUN_DIR/supervisor.pid"
LOGFILE="$RUN_DIR/session.log"
mkdir -p "$RUN_DIR"

log() {
  printf '%s [supervisor] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$LOGFILE"
}

# ---- single-instance guard ------------------------------------------------ #
if [ -f "$PIDFILE" ]; then
  OLD_PID="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [ -n "${OLD_PID:-}" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "refusing to start: supervisor already running (pid $OLD_PID, $PIDFILE)" >&2
    exit 1
  fi
  log "clearing stale pidfile (pid ${OLD_PID:-?} not running)"
  rm -f "$PIDFILE"
fi
echo "$$" > "$PIDFILE"

# ---- child management ----------------------------------------------------- #
CHILD_PID=""
SHUTTING_DOWN=0

# All descendants of $1, deepest first.
#
# On macOS `caffeinate -dims CMD` forks a child that holds the power assertion
# and then *execs* CMD in the original process — so $CHILD_PID is usually the
# trainer itself. We do not rely on that: shutdown signals $CHILD_PID *and*
# every descendant, which is correct either way (and costs only an early drop
# of the sleep assertion we are about to stop needing anyway).
descendants() {
  local pid kid
  pid="$1"
  for kid in $(pgrep -P "$pid" 2>/dev/null); do
    descendants "$kid"
    printf '%s\n' "$kid"
  done
}

cleanup() {
  local rc=$? waited=0 kids
  SHUTTING_DOWN=1
  if [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kids="$(descendants "$CHILD_PID" | tr '\n' ' ')"
    log "EXIT trap: SIGTERM -> $CHILD_PID ${kids:+and descendants [$kids]}, waiting up to ${SHUTDOWN_WAIT_S}s for checkpoint"
    # shellcheck disable=SC2086
    kill -TERM "$CHILD_PID" $kids 2>/dev/null || true
    while kill -0 "$CHILD_PID" 2>/dev/null && [ "$waited" -lt "$SHUTDOWN_WAIT_S" ]; do
      sleep 1
      waited=$((waited + 1))
    done
    if kill -0 "$CHILD_PID" 2>/dev/null; then
      log "child did not exit after ${SHUTDOWN_WAIT_S}s: SIGKILL"
      # shellcheck disable=SC2086
      kill -KILL $(descendants "$CHILD_PID" | tr '\n' ' ') "$CHILD_PID" 2>/dev/null || true
    else
      log "child exited cleanly after ${waited}s"
    fi
  fi
  [ -f "$PIDFILE" ] && [ "$(cat "$PIDFILE" 2>/dev/null || true)" = "$$" ] && rm -f "$PIDFILE"
  log "supervisor exiting (rc=$rc)"
  exit "$rc"
}
trap cleanup EXIT
trap 'log "received SIGINT"; exit 130' INT
trap 'log "received SIGTERM"; exit 143' TERM

# ---- supervise loop ------------------------------------------------------- #
log "session start: run=$RUN_NAME config=$CONFIG max_hours=$HOURS pid=$$"
log "trainer command: $CAFFEINATE $TRAINER_CMD --config $CONFIG --max-hours $HOURS"

DEADLINE=$(awk -v n="$(date +%s)" -v h="$HOURS" 'BEGIN{printf "%d", n + h*3600}')
crashes=0
attempt=0
extra_args=""
final_rc=0

while :; do
  attempt=$((attempt + 1))
  remaining_s=$(( DEADLINE - $(date +%s) ))
  if [ "$remaining_s" -le 1 ]; then
    log "time budget of ${HOURS}h exhausted; stopping"
    break
  fi
  remaining_hours="$(awk -v s="$remaining_s" 'BEGIN{printf "%.4f", s/3600}')"

  log "starting trainer (attempt $attempt, crash restarts used $crashes/$MAX_CRASH_RESTARTS, remaining ${remaining_hours}h)${extra_args:+ args:$extra_args}"
  # shellcheck disable=SC2086
  $CAFFEINATE $TRAINER_CMD --config "$CONFIG" --max-hours "$remaining_hours" $extra_args &
  CHILD_PID=$!
  log "trainer pid $CHILD_PID"

  wait "$CHILD_PID"
  rc=$?
  CHILD_PID=""
  [ "$SHUTTING_DOWN" = "1" ] && break
  log "trainer exited rc=$rc"

  extra_args=""
  case "$rc" in
    0)
      log "clean finish (rc=0); stopping"
      final_rc=0
      break
      ;;
    3)
      log "rollback requested (rc=3); restarting immediately with --halve-lr"
      extra_args="--halve-lr"
      ;;
    130|143)
      log "trainer interrupted (rc=$rc); stopping"
      final_rc="$rc"
      break
      ;;
    *)
      if [ "$crashes" -ge "$MAX_CRASH_RESTARTS" ]; then
        log "crash (rc=$rc); crash restart limit reached ($MAX_CRASH_RESTARTS); giving up"
        final_rc="$rc"
        break
      fi
      crashes=$((crashes + 1))
      log "crash (rc=$rc); restart $crashes/$MAX_CRASH_RESTARTS in 10s"
      sleep 10
      ;;
  esac
done

log "session end: run=$RUN_NAME attempts=$attempt crash_restarts=$crashes rc=$final_rc"
exit "$final_rc"
