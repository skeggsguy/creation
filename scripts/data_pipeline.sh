#!/bin/bash
# Full-size corpus build: downloads -> filters -> mix. Order matters
# (clean_general must precede topic_filter apply: it records the FineWeb
# offset that keeps the psych slice disjoint from the general slice).
# Logs to data/pipeline.log; each stage is resumable/idempotent.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."

log() { echo "[$(date '+%F %T')] $*"; }

run() {
  log "START $*"
  if uv run python -m "$@"; then log "OK    $*"; else log "FAIL  $* (rc=$?)"; exit 1; fi
}

log "=== data pipeline start ==="
run src.data.download --sources gutenberg,scifi,haiku_statworx,haiku_dugward,haiku_reddit
run src.data.download --sources fineweb_edu
run src.data.gutenberg_filter --domains philosophy,comedy,scifi
run src.data.clean_general --domains general,haiku
run src.data.topic_filter train --scan-docs 300000
run src.data.topic_filter apply
run src.data.topic_filter tilt-cosmopedia
run src.data.mix
log "=== data pipeline complete ==="
cat data/clean/stats.md
