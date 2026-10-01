#!/usr/bin/env bash
# Supervised fine-tuning. Default preset targets one 24 GB card with a 1.7B model.
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"

CONFIG="${CONFIG:-configs/sft.json}"
PRESET="${PRESET:-4090}"

case "$PRESET" in
  4090)
    # 1.7B bf16, full fine-tune, 8-bit Adam, gradient checkpointing.
    EXTRA=(--batch-size 1 --grad-accum 8 --max-seq-len 12288)
    ;;
  4090-lora)
    # 4B or 8B on 24 GB: freeze the base, train adapters only.
    EXTRA=(--lora-rank 16 --batch-size 1 --grad-accum 8 --max-seq-len 8192)
    ;;
  rtx6000)
    # 48-96 GB: longer sequences, bigger effective batch.
    EXTRA=(--batch-size 2 --grad-accum 8 --max-seq-len 16384)
    ;;
  smoke)
    # Thirty seconds of work to prove the wiring on any card.
    EXTRA=(--batch-size 1 --grad-accum 1 --max-seq-len 2048)
    ;;
  *) echo "unknown PRESET=$PRESET" >&2; exit 1 ;;
esac

"$PYTHON" -m shoprl.cli sft --config "$CONFIG" "${EXTRA[@]}" "$@"

