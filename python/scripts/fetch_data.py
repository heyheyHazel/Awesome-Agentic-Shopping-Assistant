"""Download the ShopSimulator catalog and convert it into app-ready JSON.

The raw dataset is NOT committed to the repository (no upstream license), so a
fresh clone runs on the built-in mock catalog until this script is executed.

Usage:
    python scripts/fetch_data.py                    # full catalog (~24 MB), auto mirror
    python scripts/fetch_data.py --source hf         # force the HF mirror
    python scripts/fetch_data.py --limit 500         # cap records read
    python scripts/fetch_data.py --force             # re-download raw files
    python scripts/fetch_data.py --source local --files records.jsonl --raw-dir /path/to/dir

Outputs (all gitignored):
    data/raw/*.jsonl                  untouched upstream records
    data/generated/catalog.json       Product[] ready for the app
    data/generated/shoppers.json      DemoUser[] ready for the app
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import http.client
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

HF_ENDPOINT = "https://hf-mirror.com"  # huggingface.co is unreachable from some networks
DATASET = "wpei/ShopSimulator"
# The environment repo bundles a single gzipped file covering every split;
# jsDelivr is often faster and stays up when the HF mirror does not.
CDN_ENDPOINT = "https://cdn.jsdelivr.net/gh/ShopAgent-Team/ShopSimulator@main/shop_env/data"
USER_AGENT = "shopping-assistant-fetch/1.0"

# `hf` splits products and personas across two shards, `cdn` ships one gzipped file.
# The same catalog is packaged two ways. The gzipped array is ~24 MB against the
# mirror's ~104 MB JSON Lines, so it is preferred; both hold all 23,421 products,
# 4,666 of which carry a user_persona.
MIRRORS = (
    ("cdn", CDN_ENDPOINT, "fine_items_eval_train_all.json.gz"),
    ("hf", f"{HF_ENDPOINT}/datasets/{DATASET}/resolve/main", "fine_items_eval_train_all.jsonl"),
)
DEFAULT_MIRROR = MIRRORS[0]

RAW_DIR = REPO_ROOT / "data" / "raw"
OUT_DIR = REPO_ROOT / "data" / "generated"

# Membership tiers collapse into the two levels the rest of the app understands.
VIP_TIERS = {"钻石会员", "铂金会员"}

# Preference strength ordering for the persona's category preferences.
PREFERENCE_ORDER = {"高": 0, "中": 1, "低": 2}

# Products with a non-positive price carry no signal for a recommender.
MIN_PRICE, MAX_PRICE = 1.0, 200_000.0


# ── helpers ───────────────────────────────────────────────────────────

def _hash_int(*parts: str) -> int:
    """Stable 32-bit hash so synthetic fields are reproducible across runs."""
    digest = hashlib.md5(":".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _clean(value: Any) -> str:
    """Upstream strings sometimes carry trailing separators and whitespace."""
    if value is None:
        return ""
    return str(value).strip().strip(",").strip()


def _first_number(values: Any) -> float | None:
    """Return the first parseable number in a scalar-or-list field."""
    if isinstance(values, (int, float)):
        return float(values)
    if isinstance(values, list):
        for item in values:
            if isinstance(item, (int, float)):
                return float(item)
    return None


# ── synthetic fields ──────────────────────────────────────────────────
# ShopSimulator carries no stock, rating or review count. These fields are
# SYNTHETIC and deterministic: the same asin always yields the same values so
# caches, tests and A/B buckets stay stable. Distributions are chosen to
# exercise every branch of InventoryAgent and the product-cards badges.

def synth_stock(asin: str) -> int:
    roll = _hash_int(asin, "stock") % 100
    if roll < 3:                       # ~3% sold out
        return 0
    if roll < 10:                      # ~7% low stock (below the agent's threshold of 10)
        return 1 + roll % 9
    return 20 + _hash_int(asin, "stock2") % 280


def synth_rating(asin: str) -> float:
    return round(3.5 + (_hash_int(asin, "rating") % 15) / 10, 1)


def synth_rating_count(asin: str) -> int:
    return 20 + _hash_int(asin, "reviews") % 980


# ── record mapping ────────────────────────────────────────────────────

def to_product(record: dict[str, Any]) -> dict[str, Any] | None:
    """Map one ShopSimulator record onto the app's Product shape (CNY)."""
    asin = _clean(record.get("asin"))
    title = _clean(record.get("title"))
    price = _first_number(record.get("pricing"))
    if not asin or not title or price is None:
        return None
    if not (MIN_PRICE <= price <= MAX_PRICE):
        return None

    attributes = [a for a in (_clean(x) for x in record.get("attribute") or []) if a]
    return {
        "product_id": asin,
        "name": title,
        "category": _clean(record.get("domain_zh")) or "未分类",
        "price": round(price, 2),
        "brand": _clean(record.get("shop_name")),
        "stock": synth_stock(asin),
        "rating": synth_rating(asin),
        "rating_count": synth_rating_count(asin),
        "tags": attributes[:5],
        # Kept for reference / future image support; ignored by the Product model.
        "sub_category": _clean(record.get("category")),
        "image_url": (record.get("images") or [""])[0],
    }


def to_shopper(record: dict[str, Any]) -> dict[str, Any] | None:
    """Map one persona onto the app's DemoUser shape (CNY).

    `recency_days` is DERIVED, not observed: the dataset has no "days since last
    purchase". It is a monotone function of the repurchase rate so that loyal
    shoppers look recent.
    """
    persona = record.get("user_persona") or {}
    user_id = _clean(persona.get("用户ID"))
    if not user_id:
        return None

    trading = persona.get("交易特征") or {}
    interest = persona.get("兴趣偏好") or {}
    attributes = interest.get("商品属性偏好") or {}

    orders = int(_first_number(trading.get("近90天订单数")) or 0)
    spend = _first_number(trading.get("近30天消费金额")) or 0.0
    repurchase = _first_number(trading.get("复购率"))

    price_band = attributes.get("价格区间") or {}
    low = _first_number(price_band.get("最小值")) or 0.0
    high = _first_number(price_band.get("最大值")) or 0.0

    preferences = interest.get("类目偏好") or {}
    ranked = sorted(
        ((PREFERENCE_ORDER.get(_clean(v), 3), _clean(k)) for k, v in preferences.items()),
        key=lambda item: item[0],
    )

    return {
        "user_id": user_id,
        "tier": "VIP" if _clean((persona.get("人口属性") or {}).get("会员等级")) in VIP_TIERS else "Standard",
        "orders": orders,
        "lifetime_value": round(spend, 2),
        "preferred_categories": [name for _, name in ranked if name][:3],
        "price_range": [round(low, 2), round(max(high, low), 2)],
        "repurchase_rate": repurchase if repurchase is not None else 0.5,
    }


def derive_recency(shoppers: list[dict[str, Any]]) -> None:
    """Fill `recency_days` in place, normalised across the observed repurchase range.

    Low repurchase rate -> long recency (the shopper has drifted away).
    Written as 1..89 days so it stays inside the 90-day RFM window.
    """
    rates = [s["repurchase_rate"] for s in shoppers]
    if not rates:
        return
    lowest, highest = min(rates), max(rates)
    span = highest - lowest or 1.0
    for shopper in shoppers:
        # 3 days (most loyal) .. 80 days (most lapsed)
        days = 3 + (highest - shopper["repurchase_rate"]) / span * 77
        shopper["recency_days"] = int(min(89, max(1, round(days))))
        shopper["name"] = f"用户{shopper['user_id'][-6:]}"
        shopper["email"] = f"{shopper['user_id']}@example.com"
        del shopper["repurchase_rate"]


# ── download / IO ─────────────────────────────────────────────────────

def _remote_size(url: str) -> int | None:
    """Content-Length of a remote file, or None when it cannot be determined."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError, ValueError):
        return None


def download(name: str, host: str, *, force: bool) -> Path:
    """Fetch one upstream file into data/raw/, resuming an interrupted download.

    The mirrors are slow and drop long connections, so bytes are streamed to disk
    in chunks and a partial file is resumed with a Range request rather than
    restarted (both mirrors answer Range requests with 206). A cached file is
    only trusted once its size matches the remote one, otherwise an interrupted
    download from an earlier run would be read as if it were complete.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / name
    if force and target.exists():
        target.unlink()

    url = f"{host}/{name}"
    if target.exists():
        expected = _remote_size(url)
        actual = target.stat().st_size
        if expected is None or expected == actual:
            print(f"  cached   {name}")
            return target
        print(f"  partial  {name}  {actual / 1e6:.1f} / {expected / 1e6:.1f} MB, resuming")

    for attempt in range(1, 6):
        offset = target.stat().st_size if target.exists() else 0
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                # A server that ignores Range restarts at 0, so never blind-append.
                resuming = offset > 0 and response.status == 206
                written = offset if resuming else 0
                with open(target, "ab" if resuming else "wb") as sink:
                    while chunk := response.read(1 << 16):
                        sink.write(chunk)
                        written += len(chunk)
            print(f"  download {name}  {written / 1e6:.1f} MB")
            return target
        except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as exc:
            if attempt == 5:
                raise
            print(f"  retry    {name} (attempt {attempt}) after {exc}")

    return target


def read_records(path: Path, limit: int | None) -> Iterable[dict[str, Any]]:
    """Yield records from an upstream file, transparently handling .gz.

    The mirrors disagree on format: the HF shards are JSON Lines while the file
    bundled in the environment repo is a single pretty-printed JSON array. The
    array is ~140 MB uncompressed and is parsed whole rather than streamed; that
    costs a few hundred MB of RAM once, which is fine for a setup script.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        head = handle.read(4096).lstrip()
        handle.seek(0)
        if head.startswith("["):
            for index, record in enumerate(json.load(handle)):
                if limit is not None and index >= limit:
                    return
                yield record
            return

        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                return
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ── entry point ───────────────────────────────────────────────────────

def resolve_sources(source: str, files: list[str] | None, raw_dir: Path, *, force: bool) -> list[Path]:
    """Pick the files to convert, falling through the mirrors when one is unreachable."""
    if source == "local":
        if not files:
            raise SystemExit("--source local requires --files")
        missing = [name for name in files if not (raw_dir / name).exists()]
        if missing:
            raise SystemExit(f"missing in {raw_dir}: {', '.join(missing)}")
        return [raw_dir / name for name in files]

    candidates = [mirror for mirror in MIRRORS if source in ("auto", mirror[0])]
    for name, host, default_file in candidates:
        wanted = files or [default_file]
        try:
            return [download(entry, host, force=force) for entry in wanted]
        except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as exc:
            print(f"  mirror {name} unreachable ({exc}); trying the next one", file=sys.stderr)
    raise SystemExit("no mirror served the catalog; retry when the network recovers")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", nargs="+", default=None, help="override the file list for the chosen source")
    parser.add_argument("--limit", type=int, default=None, help="cap records read per file")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    parser.add_argument("--source", choices=["auto", "cdn", "hf", "local"], default="auto",
                        help="auto = try each mirror in turn, local = read --raw-dir")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="directory holding raw jsonl files")
    args = parser.parse_args()

    sources = resolve_sources(args.source, args.files, args.raw_dir, force=args.force)

    products: dict[str, dict[str, Any]] = {}
    shoppers: dict[str, dict[str, Any]] = {}
    for path in sources:
        for record in read_records(path, args.limit):
            product = to_product(record)
            if product:
                products.setdefault(product["product_id"], product)
            shopper = to_shopper(record)
            if shopper:
                shoppers.setdefault(shopper["user_id"], shopper)

    shoppers_list = sorted(shoppers.values(), key=lambda s: s["user_id"])
    derive_recency(shoppers_list)

    if not products:
        print("No products converted — check --files / --source.", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "catalog.json").write_text(
        json.dumps(list(products.values()), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (OUT_DIR / "shoppers.json").write_text(
        json.dumps(shoppers_list, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    out_of_stock = sum(1 for p in products.values() if p["stock"] == 0)
    categories = sorted({p["category"] for p in products.values()})
    print(
        f"\ncatalog.json   {len(products):>5} products  "
        f"({out_of_stock} sold out, {len(categories)} categories: {', '.join(categories)})"
    )
    print(f"shoppers.json  {len(shoppers_list):>5} shoppers  "
          f"({sum(1 for s in shoppers_list if s['tier'] == 'VIP')} VIP)")
    print(f"→ {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
