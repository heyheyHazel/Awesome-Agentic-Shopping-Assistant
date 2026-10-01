#!/usr/bin/env bash
# GRPO on verifiable rewards, optionally with an on-policy distillation term.
#
#   ./04_train_grpo.sh                       # RLVR only
#   DISTILL=1 ./04_train_grpo.sh             # RLVR + OPD (needs TEACHER_PATH)
#   SELF_TEACHER=1 DISTILL=1 ./04_train_grpo.sh   # OPSD: same weights, privileged prompt
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"

CONFIG="${CONFIG:-configs/grpo.json}"
STEPS="${STEPS:-100}"
GROUP_SIZE="${GROUP_SIZE:-4}"
TASKS_PER_STEP="${TASKS_PER_STEP:-5}"

ARGS=(--config "$CONFIG" --steps "$STEPS" --group-size "$GROUP_SIZE" --tasks-per-step "$TASKS_PER_STEP")

if [[ "${DISTILL:-0}" == "1" ]]; then
  ARGS+=(--distill-beta "${DISTILL_BETA:-0.1}" --distill-mode "${DISTILL_MODE:-forward_kl}")
  if [[ "${SELF_TEACHER:-0}" == "1" ]]; then
    # OPSD: the teacher is the same model, so no second checkpoint is loaded.
    ARGS+=(--self-teacher)
  else
    : "${TEACHER_PATH:?set TEACHER_PATH to a stronger checkpoint for OPD}"
    ARGS+=(--teacher-path "$TEACHER_PATH")
  fi
fi

"$PYTHON" -m shoprl.cli grpo "${ARGS[@]}" "$@"

# What to watch in train_log.jsonl:
#   zero_variance_groups  high for the first steps is normal; if it stays high,
#                         the task pool is too easy or too hard for the policy.
#   done_rate             should climb before r_loose does.
#   turn_samples          a rollout that dies early contributes no gradient.

