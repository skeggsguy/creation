#!/usr/bin/env bash
# start / stop / status for the dashboard server, detached so it survives across
# training sessions.
#
#   scripts/dashboard.sh start [--run NAME] [--port 8471]
#   scripts/dashboard.sh stop
#   scripts/dashboard.sh status
#   scripts/dashboard.sh restart [...]
#
# pidfile: runs/.dashboard.pid   log: runs/.dashboard.log

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PIDFILE="$REPO_ROOT/runs/.dashboard.pid"
LOGFILE="$REPO_ROOT/runs/.dashboard.log"
PORT=8471
RUN_ARGS=()

cmd="${1:-status}"
shift || true
while [ $# -gt 0 ]; do
  case "$1" in
    --run)  RUN_ARGS+=(--run "${2:-}"); shift 2 ;;
    --port) PORT="${2:-8471}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$REPO_ROOT/runs"

running_pid() {
  [ -f "$PIDFILE" ] || return 1
  local pid
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  printf '%s' "$pid"
}

do_start() {
  local pid
  if pid="$(running_pid)"; then
    echo "already running (pid $pid) -> http://127.0.0.1:$PORT"
    return 0
  fi
  rm -f "$PIDFILE"
  # stdlib only: plain python3 is enough, no venv needed.
  nohup python3 "$REPO_ROOT/dashboard/server.py" --port "$PORT" "${RUN_ARGS[@]+"${RUN_ARGS[@]}"}" \
    >> "$LOGFILE" 2>&1 &
  echo $! > "$PIDFILE"
  sleep 1
  if pid="$(running_pid)"; then
    echo "started (pid $pid) -> http://127.0.0.1:$PORT  log: $LOGFILE"
  else
    echo "failed to start; last log lines:" >&2
    tail -5 "$LOGFILE" >&2
    rm -f "$PIDFILE"
    return 1
  fi
}

do_stop() {
  local pid waited=0
  if ! pid="$(running_pid)"; then
    echo "not running"
    rm -f "$PIDFILE"
    return 0
  fi
  kill -TERM "$pid" 2>/dev/null || true
  while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt 10 ]; do sleep 1; waited=$((waited + 1)); done
  kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
  rm -f "$PIDFILE"
  echo "stopped (pid $pid)"
}

do_status() {
  local pid
  if pid="$(running_pid)"; then
    echo "running (pid $pid) -> http://127.0.0.1:$PORT"
  else
    echo "not running"
    return 1
  fi
}

case "$cmd" in
  start)   do_start ;;
  stop)    do_stop ;;
  restart) do_stop; do_start ;;
  status)  do_status ;;
  *) echo "usage: $0 {start|stop|restart|status} [--run NAME] [--port N]" >&2; exit 2 ;;
esac
