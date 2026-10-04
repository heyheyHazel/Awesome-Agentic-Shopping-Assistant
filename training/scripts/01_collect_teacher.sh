#!/usr/bin/env bash
# Collect teacher trajectories for SFT.
#
# The teacher only has to finish the task: trajectories that end without a
# purchase are recorded with their reason and dropped from SFT.
#
# Run it offline first. DRY_RUN=1 swaps the API for a scripted teacher and walks
# the whole path — task pool, environment, harness, trajectory writing — with no
# network call, so the plumbing is known to work before any credit is spent.
#
#   DRY_RUN=1 TASKS=dev LIMIT=8 bash training/scripts/01_collect_teacher.sh
#   TEACHER_MODEL=deepseek-chat TEACHER_BASE_URL=... TEACHER_API_KEY=... \
#       bash training/scripts/01_collect_teacher.sh
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"

TASKS="${TASKS:-sft}"
OUT="${OUT:-runs/teacher_${TASKS}}"
CONCURRENCY="${CONCURRENCY:-4}"
SAMPLES="${SAMPLES:-1}"
LIMIT="${LIMIT:-0}"

ARGS=(
  --tasks "$TASKS"
  --out "$OUT"
  --samples-per-task "$SAMPLES"
  --concurrency "$CONCURRENCY"
  --temperature 0.7
  --max-turns 30
)
[ "$LIMIT" != "0" ] && ARGS+=(--limit "$LIMIT")

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "dry run: scripted teacher, no API call, no credit spent"
  "$PYTHON" -m shoprl.cli collect "${ARGS[@]}" --teacher scripted
else
  : "${TEACHER_MODEL:?set TEACHER_MODEL, e.g. deepseek-chat}"
  : "${TEACHER_BASE_URL:?set TEACHER_BASE_URL, e.g. https://api.deepseek.com/v1}"
  : "${TEACHER_API_KEY:?set TEACHER_API_KEY}"
  "$PYTHON" -m shoprl.cli collect "${ARGS[@]}" \
    --teacher openai \
    --model "$TEACHER_MODEL" \
    --base-url "$TEACHER_BASE_URL" \
    --api-key "$TEACHER_API_KEY"
fi

echo "Raw trajectories in $OUT/raw; summary in $OUT/summary.json"
