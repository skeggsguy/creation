#!/bin/bash
# Battery thermostat daemon: keeps a power-hungry training run alive on an
# undersized (96W) adapter by flipping macOS power modes on battery level.
#   <= LOW  % -> Automatic mode (GPU throttles ~10%, battery charges)
#   >= HIGH % -> High Power mode (full speed, battery slowly drains)
# Requires passwordless sudo for pmset (/etc/sudoers.d/pmset).
# Managed by launchd (com.llmmod.battery-thermostat) — KeepAlive restarts it,
# RunAtLoad starts it on boot/login. Logs to runs/thermostat.log.
set -u
LOW=${LOW:-20}
HIGH=${HIGH:-80}
INTERVAL=${INTERVAL:-60}
LOG="$(cd "$(dirname "$0")/.." && pwd)/runs/thermostat.log"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# TEST_DIR set => harness mode: battery/mode are files, no pmset, no sudo.
# The decision logic below is identical in both modes.
if [ -n "${TEST_DIR:-}" ]; then
  get_pct()  { cat "$TEST_DIR/pct" 2>/dev/null; }
  get_mode() { cat "$TEST_DIR/mode" 2>/dev/null; }
  set_mode() { echo "$1" > "$TEST_DIR/mode"; log "switched AC powermode -> $1 ($2)"; }
else
  get_pct()  { pmset -g batt | grep -o '[0-9]*%' | tr -d '%'; }
  get_mode() { pmset -g | awk '/powermode/ {print $2}'; }   # 0=auto 1=low 2=high
  set_mode() { # $1 = 0|2, $2 = reason
    if sudo -n pmset -c powermode "$1" 2>>"$LOG"; then
      log "switched AC powermode -> $1 ($2)"
    else
      log "ERROR: pmset failed (sudoers grant missing?) wanted mode $1 ($2)"
    fi
  }
fi

log "thermostat started (LOW=$LOW HIGH=$HIGH interval=${INTERVAL}s) pid=$$"
while true; do
  pct=$(get_pct); mode=$(get_mode)
  if [ -n "${pct:-}" ] && [ -n "${mode:-}" ]; then
    if [ "$pct" -le "$LOW" ] && [ "$mode" != "0" ]; then
      set_mode 0 "battery ${pct}% <= ${LOW}%"
    elif [ "$pct" -ge "$HIGH" ] && [ "$mode" != "2" ]; then
      set_mode 2 "battery ${pct}% >= ${HIGH}%"
    fi
  fi
  sleep "$INTERVAL"
done
