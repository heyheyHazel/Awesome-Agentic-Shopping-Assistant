#!/usr/bin/env bash
# Answer one question: is this box ready to start training?
#
# Runs only CPU-side checks and never starts a job. Every line is either PASS or
# FAIL; the exit code is non-zero if anything required is missing. A missing GPU
# is reported but does not fail the check, because the card may simply not be
# allocated yet.
#
# Usage:  bash scripts/check_ready.sh
set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${PYTHON:-$PROJECT_DIR/.venv/bin/python}"
MODEL="${MODEL:-Qwen3-1.7B}"
FAILED=0

pass() { printf 'PASS  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; FAILED=$((FAILED + 1)); }
info() { printf 'INFO  %s\n' "$1"; }

cd "$PROJECT_DIR"
echo "== environment =="
if [ -x "$PYTHON" ]; then pass "venv at $PYTHON"; else fail "no venv at $PYTHON (run scripts/setup_devbox.sh)"; fi

if "$PYTHON" - <<'PY'
import sys

import torch

assert torch is not None
sys.exit(0)
PY
then
  info "$("$PYTHON" -c 'import torch;print("torch", torch.__version__, "cuda-build", torch.version.cuda)')"
  if "$PYTHON" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    pass "CUDA visible: $("$PYTHON" -c 'import torch;print(torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/2**30), "GiB")')"
  else
    info "CUDA not visible yet (no GPU allocated); training cannot start until it is"
  fi
else
  fail "torch not importable from the venv"
fi

echo "== data =="
if "$PYTHON" - <<'PY' 2>/dev/null
from shoprl.env.catalog import load_catalog

catalogue = load_catalog()
assert len(catalogue) == 23421, len(catalogue)
PY
then
  pass "catalogue built: 23,421 products"
else
  fail "catalogue missing or wrong size; run: python -m shoprl.cli catalogue"
fi

if "$PYTHON" - <<'PY' 2>/dev/null
from shoprl.env.tasks import load_task_pool

expected = {"official_test": 1459, "dev": 150, "sft": 512, "rl": 500}
for name, size in expected.items():
    assert len(load_task_pool(name)) == size, (name, len(load_task_pool(name)))
PY
then
  pass "task pools built and sized as documented"
else
  fail "task pools missing or wrong size; run: python -m shoprl.cli tasks"
fi

echo "== checkpoints =="
if [ -f "$PROJECT_DIR/models/$MODEL/config.json" ]; then
  pass "student checkpoint present: models/$MODEL"
else
  fail "models/$MODEL is absent; run: python scripts/download_models.py"
fi
if [ -f "$PROJECT_DIR/models/bge-small-zh-v1.5/onnx/model_quantized.onnx" ]; then
  pass "local embedding model present (serving path)"
else
  info "embedding model absent; the storefront falls back to keyword recall"
fi

echo "== raw data =="
if "$PYTHON" - <<'PY' 2>/dev/null
import json
from pathlib import Path

raw = Path("data/raw")
verified = json.loads((raw / ".verified.json").read_text())
for name, size in verified.items():
    assert (raw / name).stat().st_size == size, name
side_car = raw / "fine_items_train_persona.jsonl"
rows = [json.loads(line) for line in side_car.open(encoding="utf-8") if line.strip()]
assert len(rows) == 3323, len(rows)
PY
then
  pass "raw files match .verified.json; persona side-car complete (3,323 records)"
else
  fail "raw data incomplete; run: python scripts/fetch_data.py --source hf --download-only"
fi

echo "== code =="
if "$PYTHON" -m pytest -q >/tmp/pytest-ready.log 2>&1; then
  pass "test suite green: $(tail -1 /tmp/pytest-ready.log | tr -d '\n')"
else
  fail "test suite red; see /tmp/pytest-ready.log"
fi

echo "== disk =="
info "$(df -h "$PROJECT_DIR" | tail -1 | awk '{print $4" free on "$6}')"

echo
if [ "$FAILED" -eq 0 ]; then
  echo "READY: everything except a GPU allocation is in place."
else
  echo "NOT READY: $FAILED check(s) failed."
fi
exit "$FAILED"
