"""Download the ShopSimulator catalog and convert it into app-ready JSON.

The raw dataset is NOT committed to the repository (no upstream license), so a
fresh clone runs on the built-in mock catalog until this script is executed.

Usage:
    python scripts/fetch_data.py                    # full catalog (~24 MB), auto mirror
    python scripts/fetch_data.py --files a.jsonl b.jsonl --download-only   # raw files only
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
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from shopping_assistant.settings import get_settings

HF_ENDPOINT = "https://hf-mirror.com"  # huggingface.co is unreachable from some networks
AIFASTHUB = "https://aifasthub.com"    # a second HF mirror; the two fail independently
DATASET = "wpei/ShopSimulator"
# The environment repo bundles a single gzipped file covering every split;
# jsDelivr is often faster and stays up when the HF mirror does not.
CDN_ENDPOINT = "https://cdn.jsdelivr.net/gh/ShopAgent-Team/ShopSimulator@main/shop_env/data"
USER_AGENT = "agentic-shopping-assistant-fetch/1.0"

# `hf` splits products and personas across two shards, `cdn` ships one gzipped file.
# The same catalog is packaged two ways. The gzipped array is ~24 MB against the
# mirror's ~104 MB JSON Lines, so it is preferred; both hold all 23,421 products,
# 4,666 of which carry a user_persona.
MIRRORS = (
    # ~24 MB gzipped array: a quarter of the JSON Lines size, but jsDelivr
    # intermittently answers 404, so it is retried like any other mirror.
    ("cdn", CDN_ENDPOINT, "fine_items_eval_train_all.json.gz"),
    # ~104 MB JSON Lines on two Hugging Face mirrors. aifasthub is tried first
    # because hf-mirror has been observed unreachable for long stretches.
    ("aifasthub", f"{AIFASTHUB}/datasets/{DATASET}/resolve/main", "fine_items_eval_train_all.jsonl"),
    ("hf", f"{HF_ENDPOINT}/datasets/{DATASET}/resolve/main", "fine_items_eval_train_all.jsonl"),
)
DEFAULT_MIRROR = MIRRORS[0]

# One source of truth for artifact locations, shared with the API (ECOM_DATA_DIR).
RAW_DIR = get_settings().data_dir / "raw"
OUT_DIR = get_settings().data_dir / "generated"

# Membership tiers collapse into the two levels the rest of the app understands.
VIP_TIERS = {"钻石会员", "铂金会员"}

# Preference strength ordering for the persona's category preferences.
PREFERENCE_ORDER = {"高": 0, "中": 1, "低": 2}

# Products with a non-positive price carry no signal for a recommender.
MIN_PRICE, MAX_PRICE = 1.0, 200_000.0

# Enough retries to survive a flaky mirror, few enough to move on to the next one.
DOWNLOAD_ATTEMPTS = 6

# The full catalogue holds ~23,000 products. Anything far below that means the
# source was incomplete, and converting it would produce a silently tiny catalogue.
MIN_PLAUSIBLE_PRODUCTS = 10_000


class MirrorUnavailable(RuntimeError):
    """A mirror could not deliver a complete file; try the next one."""


# Records the byte size of every file that passed a Content-Length check. Asking a
# host costs ~28 s on a cold CDN, so a recorded size is what makes a re-run instant
# while still catching a truncated file.
VERIFIED_MANIFEST = RAW_DIR / ".verified.json"


def _verified_sizes() -> dict[str, int]:
    try:
        return json.loads(VERIFIED_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _is_complete(path: Path) -> bool:
    """Local integrity check, for when no size can be obtained from the host.

    A truncated gzip fails its own CRC while being read, and a truncated JSON Lines
    file ends mid-record, so both can be detected offline. This matters because a
    host that answers no HEAD request would otherwise make every partial file look
    finished.
    """
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rb") as handle:
                while handle.read(1 << 20):
                    pass
            return True

        with open(path, "rb") as handle:
            handle.seek(max(0, path.stat().st_size - 65536))
            tail = handle.read().decode("utf-8", "ignore").strip()
        if not tail:
            return False
        if tail.startswith("["):  # a single pretty-printed JSON array
            return tail.endswith("]")
        return _parses(tail.splitlines()[-1])
    except Exception:  # noqa: BLE001
        return False


def _parses(line: str) -> bool:
    try:
        json.loads(line)
        return True
    except ValueError:
        return False


def _record_verified(name: str, size: int) -> None:
    sizes = _verified_sizes()
    sizes[name] = size
    VERIFIED_MANIFEST.write_text(json.dumps(sizes, indent=1, sort_keys=True), encoding="utf-8")


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
    """Fetch one upstream file into the raw dir, resuming an interrupted download.

    The mirrors are slow, drop long connections and occasionally answer with a
    transient 404, so a transfer is retried and always resumed rather than restarted.

    A transfer is only accepted once the file size matches the host's Content-Length.
    Without that check a connection closing mid-body looks like a clean finish to
    urllib, and a 2.8 MB fragment of a 104 MB file would be converted into a
    truncated catalogue that nothing downstream can detect.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / name
    if force and target.exists():
        target.unlink()

    url = f"{host}/{name}"

    # A size we recorded ourselves is as trustworthy as the host's, and costs no request.
    expected = _remote_size(url) or _verified_sizes().get(name)

    if target.exists():
        actual = target.stat().st_size
        if actual == expected or (expected is None and _is_complete(target)):
            _record_verified(name, actual)
            print(f"  cached   {name}")
            return target
        short = f"/{expected / 1e6:.1f} MB" if expected else " (no size to check against)"
        print(f"  refetch  {name}  {actual / 1e6:.1f} MB{short}")

    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        offset = target.stat().st_size if target.exists() else 0
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                # A server that ignores Range restarts at 0, so never blind-append.
                resuming = offset > 0 and response.status == 206
                with open(target, "ab" if resuming else "wb") as sink:
                    while chunk := response.read(1 << 16):
                        sink.write(chunk)
        except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as exc:
            if attempt == DOWNLOAD_ATTEMPTS:
                raise MirrorUnavailable(f"{name}: {exc}") from exc
            print(f"  retry    {name} (attempt {attempt}) after {exc}")
            continue

        actual = target.stat().st_size if target.exists() else 0
        complete = actual == expected if expected is not None else _is_complete(target)
        if complete:
            _record_verified(name, actual)
            print(f"  download {name}  {actual / 1e6:.1f} MB")
            return target
        if attempt == DOWNLOAD_ATTEMPTS:
            raise MirrorUnavailable(
                f"{name} stopped at {actual / 1e6:.1f} MB and did not verify"
                + (f" against the {expected / 1e6:.1f} MB the host reports" if expected else "")
            )
        short = f"/{expected / 1e6:.1f} MB" if expected else " (unverified)"
        print(f"  partial  {name}  {actual / 1e6:.1f} MB{short}, resuming")

    raise MirrorUnavailable(f"could not download {name}")


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
        except MirrorUnavailable as exc:
            print(f"  mirror {name} failed ({exc}); trying the next one", file=sys.stderr)
    raise SystemExit("no mirror served the catalog; retry when the network recovers")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", nargs="+", default=None, help="override the file list for the chosen source")
    parser.add_argument("--limit", type=int, default=None, help="cap records read per file")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    parser.add_argument("--download-only", action="store_true",
                        help="fetch the raw files but do not convert them or touch the catalogue")
    parser.add_argument("--source", choices=["auto", "cdn", "hf", "local"], default="auto",
                        help="auto = try each mirror in turn, local = read --raw-dir")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="directory holding raw jsonl files")
    args = parser.parse_args()

    sources = resolve_sources(args.source, args.files, args.raw_dir, force=args.force)

    if args.download_only:
        print(f"\nFetched {len(sources)} file(s) into {RAW_DIR}; nothing was converted.")
        return 0

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

    if args.limit is None and len(products) < MIN_PLAUSIBLE_PRODUCTS:
        print(
            f"\nOnly {len(products)} products came out of the source files, but the full "
            f"catalogue holds about 23,000. The download was most likely incomplete.\n"
            f"Delete {RAW_DIR} and run this script again.",
            file=sys.stderr,
        )
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
