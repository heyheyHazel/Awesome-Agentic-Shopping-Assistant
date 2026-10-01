#!/usr/bin/env bash
# Collect teacher trajectories for SFT.
#
# The teacher only has to finish the task: trajectories that end without a purchase
# are recorded with their reason and dropped from SFT.
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"

: "${TEACHER_MODEL:?set TEACHER_MODEL, e.g. deepseek-chat}"
: "${TEACHER_BASE_URL:?set TEACHER_BASE_URL, e.g. https://api.deepseek.com/v1}"
: "${TEACHER_API_KEY:?set TEACHER_API_KEY}"

TASKS="${TASKS:-sft}"
OUT="${OUT:-runs/teacher_${TASKS}}"
CONCURRENCY="${CONCURRENCY:-4}"
SAMPLES="${SAMPLES:-1}"

"$PYTHON" -m shoprl.cli collect \
  --tasks "$TASKS" \
  --out "$OUT" \
  --model "$TEACHER_MODEL" \
  --base-url "$TEACHER_BASE_URL" \
  --api-key "$TEACHER_API_KEY" \
  --samples-per-task "$SAMPLES" \
  --concurrency "$CONCURRENCY" \
  --temperature 0.7 \
  --max-turns 30

echo "Raw trajectories in $OUT/raw; summary in $OUT/summary.json"

