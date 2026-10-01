#!/usr/bin/env bash
# Build the product cache and the task pools from whatever ShopSimulator data is on disk.
#
# Nothing here touches the network: it reads data/raw/fine_items_eval_train_all.json.gz
# and writes data/generated/shop_products.jsonl.gz plus data/tasks/*.jsonl.
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"

if [[ ! -f data/raw/fine_items_eval_train_all.json.gz ]]; then
  echo "Missing data/raw/fine_items_eval_train_all.json.gz." >&2
  echo "Fetch it first:  python scripts/setup.py --catalog" >&2
  exit 1
fi

"$PYTHON" -m shoprl.cli catalogue
"$PYTHON" -m shoprl.cli tasks --dev 150 --sft 512 --rl 500

echo
echo "Pools written to data/tasks/. Splits are disjoint and seeded, so this is reproducible."

