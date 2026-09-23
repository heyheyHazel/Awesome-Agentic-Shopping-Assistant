"""Embed the catalog so recall can search it semantically.

Run after `fetch_data.py`; re-run whenever the catalog changes (the index is keyed
by product id, so a changed catalog makes an old index unusable and recall falls
back to keyword-only until this runs again).

    python scripts/build_index.py
    python scripts/build_index.py --limit 500     # quick smoke test

Outputs (gitignored):
    data/generated/embeddings.npy     float32, one L2-normalised row per product
    data/generated/embedding_ids.json product ids in row order
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data import PRODUCTS  # noqa: E402
from services.embeddings import ensure_model, get_encoder  # noqa: E402
from services.vector_index import IDS_FILE, VECTORS_FILE, VectorIndex, product_text  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="embed only the first N products")
    parser.add_argument("--batch-size", type=int, default=32, help="encoder batch size")
    args = parser.parse_args()

    products = PRODUCTS[: args.limit] if args.limit else PRODUCTS
    if not products:
        print("No products loaded — run scripts/fetch_data.py first.", file=sys.stderr)
        return 1

    print(f"Encoding {len(products)} products …")
    started = time.perf_counter()
    ensure_model()
    encoder = get_encoder()
    vectors = encoder.encode([product_text(product) for product in products])
    elapsed = time.perf_counter() - started

    VectorIndex([product.product_id for product in products], vectors).save()
    print(
        f"{len(products)} products → {vectors.shape[1]}-d vectors in {elapsed:.1f}s "
        f"({len(products) / max(elapsed, 1e-6):.0f} products/s)"
    )
    print(f"→ {VECTORS_FILE.name}, {IDS_FILE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
