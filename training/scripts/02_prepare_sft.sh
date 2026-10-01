#!/usr/bin/env bash
# Convert accepted trajectories into turn-level SFT examples with a real loss mask.
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
INPUT="${INPUT:-runs/teacher_sft}"
OUT="${OUT:-runs/sft_data}"
MAX_TOKENS="${MAX_TOKENS:-12288}"

"$PYTHON" -m shoprl.cli prepare-sft \
  --input "$INPUT" \
  --out "$OUT" \
  --model "$MODEL" \
  --max-tokens "$MAX_TOKENS"

echo "Turn-level examples: $OUT/turn_examples.jsonl"
echo "Check summary.json: token_p95 must fit MAX_TOKENS or examples are being dropped."

