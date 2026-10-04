#!/usr/bin/env bash
# Score a policy on the official evaluation split.
#
# DRY_RUN=1 scores the scripted reference policy instead of a checkpoint: it
# plays the page it was shown, needs no model and no API call, and gives the
# number a real checkpoint has to beat on the same sample.
#
#   DRY_RUN=1 LIMIT=20 bash training/scripts/05_eval.sh
#   MODEL=runs/qwen3-1.7b-grpo/step-100 LIMIT=200 bash training/scripts/05_eval.sh
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"

TASKS="${TASKS:-official_test}"
LIMIT="${LIMIT:-200}"
SAMPLES="${SAMPLES:-1}"
CONCURRENCY="${CONCURRENCY:-4}"
TEMPERATURE="${TEMPERATURE:-0.0}"

ARGS=(
  --tasks "$TASKS"
  --limit "$LIMIT"
  --samples "$SAMPLES"
  --concurrency "$CONCURRENCY"
  --temperature "$TEMPERATURE"
)

if [ "${DRY_RUN:-0}" = "1" ]; then
  OUT="${OUT:-runs/eval_scripted}"
  LABEL="${LABEL:-scripted}"
  echo "dry run: scripted reference policy, no API call, no credit spent"
  "$PYTHON" -m shoprl.cli eval "${ARGS[@]}" --out "$OUT" --label "$LABEL" --teacher scripted
else
  : "${MODEL:?set MODEL to a checkpoint path or served model name}"
  OUT="${OUT:-runs/eval_$(basename "$MODEL")}"
  LABEL="${LABEL:-$(basename "$MODEL")}"
  "$PYTHON" -m shoprl.cli eval "${ARGS[@]}" \
    --out "$OUT" \
    --model "$MODEL" \
    --base-url "${BASE_URL:-http://127.0.0.1:8000/v1}" \
    --label "$LABEL" \
    --teacher openai
fi

echo "Metrics: $OUT/metrics.json   Rollouts: $OUT/rollouts.jsonl"
