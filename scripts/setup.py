"""Prepare everything the app needs, in one command.

    python scripts/setup.py              # catalog + semantic index
    python scripts/setup.py --check      # report what is ready, download nothing
    python scripts/setup.py --catalog    # only the product catalogue
    python scripts/setup.py --index      # only the semantic recall index

Both steps download from mirrors that are unreachable from some networks, so they
resume interrupted files and try several hosts. A failure of the index step is not
fatal: recall falls back to keyword-only and the app still runs, so this exits 0
with a clear warning rather than leaving a half-configured checkout.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from shopping_assistant.catalog import PRODUCTS  # noqa: E402
from shopping_assistant.settings import get_settings  # noqa: E402
from shopping_assistant.retrieval.embeddings import MODEL_DIR, model_is_present  # noqa: E402
from shopping_assistant.retrieval.index import load as load_index  # noqa: E402

SCRIPTS = REPO_ROOT / "scripts"

# Below this, the generated catalogue is a fragment rather than the real dataset.
MIN_PLAUSIBLE_PRODUCTS = 10_000


def status() -> dict:
    """What is actually usable on disk, without touching the network.

    Checked by reading the artefacts rather than by testing that files exist: a
    half-written download once produced a 399-product catalogue that passed an
    existence check and was reported as ready.
    """
    product_ids = [product.product_id for product in PRODUCTS]
    return {
        "products": len(product_ids),
        "catalog_ok": len(product_ids) >= MIN_PLAUSIBLE_PRODUCTS,
        "index": load_index(product_ids) is not None,
        "model": model_is_present(MODEL_DIR),
    }


def report(ready: dict) -> None:
    catalog = f"{ready['products']:,} products" if ready["catalog_ok"] else "32 built-in demo products"
    if ready["products"] and not ready["catalog_ok"]:
        catalog = f"only {ready['products']:,} products — looks incomplete, re-run setup"
    recall = "hybrid keyword + semantic" if ready["index"] else "keyword only"
    print(f"\n  data dir  {get_settings().data_dir}")
    print(f"  catalog   {catalog}")
    print(f"  recall    {recall}")


def run(script: str, label: str, *flags: str) -> bool:
    """Run one step with its output streaming through, so it never looks hung."""
    print(f"\n=== {label} ===\n")
    command = [sys.executable, str(SCRIPTS / script), *flags]
    return subprocess.run(command).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="report what is ready and exit")
    parser.add_argument("--catalog", action="store_true", help="run only the catalogue step")
    parser.add_argument("--index", action="store_true", help="run only the index step")
    parser.add_argument("--force", action="store_true", help="re-run steps that are already done")
    args = parser.parse_args()

    ready = status()
    if args.check:
        report(ready)
        return 0

    only_one = args.catalog or args.index
    do_catalog = args.catalog or not only_one
    do_index = args.index or not only_one

    if do_catalog and (args.force or not ready["catalog_ok"]):
        if not run("fetch_data.py", "Step 1/2  download and convert the product catalogue", *(["--force"] if args.force else [])):
            print("\nThe catalogue could not be prepared, so the app would run on the")
            print("32-product demo data. Retry when the network recovers — partial")
            print("downloads are kept and resumed, not restarted.")
            return 1
    elif do_catalog:
        print("catalogue already present (use --force to refetch)")

    if do_index and (args.force or not ready["index"]):
        if not run("build_index.py", "Step 2/2  build the semantic recall index"):
            print("\nThe semantic index could not be built. Everything else works:")
            print("recall falls back to keyword-only, so the app is usable now.")
            print("Re-run `python scripts/setup.py --index` to add semantic recall.")
            report(status())
            return 0
    elif do_index:
        print("semantic index already present (use --force to rebuild)")

    report(status())
    print("\nReady. Start the app with:  python backend/main.py   → http://localhost:8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
