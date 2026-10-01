#!/usr/bin/env bash
# Score a checkpoint on the official evaluation split.
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"

: "${MODEL:?set MODEL to a checkpoint path or served model name}"

TASKS="${TASKS:-official_test}"
OUT="${OUT:-runs/eval_$(basename "$MODEL")}"
LIMIT="${LIMIT:-200}"
SAMPLES="${SAMPLES:-1}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
LABEL="${LABEL:-$(basename "$MODEL")}"

"$PYTHON" -m shoprl.cli eval \
  --tasks "$TASKS" \
  --out "$OUT" \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --label "$LABEL" \
  --limit "$LIMIT" \
  --samples "$SAMPLES" \
  --concurrency "${CONCURRENCY:-4}" \
  --temperature "${TEMPERATURE:-0.0}"

echo "Metrics: $OUT/metrics.json   Rollouts: $OUT/rollouts.jsonl"

