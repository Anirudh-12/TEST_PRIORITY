#!/usr/bin/env bash
# =============================================================================
# run_dataset_collection.sh
# Phase 4: Run the BugsInPy processing pipeline for all target bugs.
#
# Usage (from Windows PowerShell):
#   uv run bash scripts/run_dataset_collection.sh
#
# Or from WSL (where the script will invoke the pipeline through uv):
#   bash /mnt/c/Users/aksha/OneDrive/Documents/TEST_PRIORITY/scripts/run_dataset_collection.sh
#
# Results are written to:
#   data/processed/<project>/<bug_id>.json
#   data/dataset_status.jsonl
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SUMMARY_LOG="$PROJECT_ROOT/data/collection_run_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$PROJECT_ROOT/data"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SUMMARY_LOG"; }

log "========================================================"
log "Phase 4: Dataset Collection"
log "Project root: $PROJECT_ROOT"
log "Log: $SUMMARY_LOG"
log "========================================================"

# --- Bug list ---
# Format: "project bug_id"
# Ordered from smallest to largest suite for fast feedback
BUGS=(
    "cookiecutter 1"
    "cookiecutter 2"
    "cookiecutter 3"
    "cookiecutter 4"
    "httpie 1"
    "httpie 2"
    "httpie 3"
    "httpie 4"
    "black 1"
    "black 2"
    "black 3"
    "black 4"
    "thefuck 2"
    "thefuck 3"
    "thefuck 4"
    "thefuck 5"
    "thefuck 6"
    "thefuck 7"
    "thefuck 8"
    "thefuck 9"
    "thefuck 10"
    "youtube-dl 1"
    "youtube-dl 2"
    "youtube-dl 3"
)

SUCCESS=0
FAILED=0
SKIPPED=0
TOTAL=${#BUGS[@]}
IDX=0

for entry in "${BUGS[@]}"; do
    IDX=$((IDX + 1))
    project=$(echo "$entry" | awk '{print $1}')
    bug_id=$(echo "$entry" | awk '{print $2}')

    log ""
    log "[$IDX/$TOTAL] Processing $project/$bug_id ..."

    # Check if already processed successfully
    out_file="$PROJECT_ROOT/data/processed/$project/$bug_id.json"
    if [ -f "$out_file" ]; then
        status=$(python3 -c "
import json
with open('$out_file') as f:
    d = json.load(f)
print(d.get('dataset_status', 'UNKNOWN'))
" 2>/dev/null || echo "UNKNOWN")
        if [ "$status" = "SUCCESSFULLY_PROCESSED" ]; then
            log "  [SKIP] Already SUCCESSFULLY_PROCESSED — use --force to reprocess"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
    fi

    # Run the pipeline
    if cd "$PROJECT_ROOT" && uv run python -m bugsinpy.process \
        --project "$project" \
        --bug "$bug_id" \
        --skip-coverage \
        --timeout 30 \
        --parallel 8 \
        >> "$SUMMARY_LOG" 2>&1; then
        log "  [OK] $project/$bug_id complete"
        SUCCESS=$((SUCCESS + 1))
    else
        log "  [WARN] $project/$bug_id returned non-zero exit (may still have partial data)"
        FAILED=$((FAILED + 1))
    fi

    # Brief pause between bugs to avoid WSL filesystem contention
    sleep 2
done

log ""
log "========================================================"
log "Collection complete."
log "  Successful: $SUCCESS"
log "  Failed/Partial: $FAILED"
log "  Skipped (already done): $SKIPPED"
log "  Total: $TOTAL"
log "========================================================"
log ""
log "Run 'uv run python scripts/dataset_audit.py' to see full summary."
