#!/usr/bin/env bash
# Provision a fresh AutoDL box for this project.
#
# Deliberately does NOT install torch. The image ships torch built against its
# own CUDA runtime (2.12.1+cu130 here); reinstalling it costs several GB of
# download on a single-core box and can silently swap the CUDA build. The venv
# inherits the image interpreter with --system-site-packages instead.
#
# Usage:  bash scripts/setup_devbox.sh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python3}"
VENV="${VENV:-$PROJECT_DIR/.venv}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"   # huggingface.co is unreachable here

cd "$PROJECT_DIR"
echo "project: $PROJECT_DIR"

echo "== uv =="
command -v uv >/dev/null 2>&1 || "$PYTHON" -m pip install -q uv
UV="$(command -v uv || echo /root/miniconda3/bin/uv)"

echo "== virtualenv inheriting the image torch =="
[ -d "$VENV" ] || "$UV" venv --system-site-packages "$VENV" --python "$PYTHON"
"$VENV/bin/python" - <<'PROBE'
import torch
print("inherited torch", torch.__version__, "| cuda build", torch.version.cuda, "|", torch.__file__)
PROBE

# pip, not uv, from here on: pip treats the inherited site-packages as installed
# and leaves the image's torch alone, while uv's resolver would fetch its own
# ~2.5 GB CUDA build of the same version.
echo "== pip inside the venv =="
"$UV" pip install --python "$VENV/bin/python" pip >/dev/null

echo "== project and serving dependencies =="
"$VENV/bin/python" -m pip install -q -e . pytest

echo "== training extras =="
"$VENV/bin/python" -m pip install -q transformers peft accelerate datasets

echo "== optional: 8-bit optimiser for full fine-tuning on 24 GB =="
"$VENV/bin/python" -m pip install -q bitsandbytes || echo "  bitsandbytes unavailable here; use --optim adamw_torch"

echo "== smoke =="
"$VENV/bin/python" - <<'PROBE'
import peft
import torch
import transformers

import shoprl
from shoprl.env.catalog import load_catalog

print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
print("catalogue", len(load_catalog()), "products")
PROBE

echo "== ready: $VENV/bin/python =="
