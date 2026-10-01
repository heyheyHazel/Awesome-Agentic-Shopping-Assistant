"""Product catalogue for the ShopSimulator environment.

Upstream ships one record per task in ``fine_items_eval_train_all.json.gz`` and
the row order *is* the task id space, so this module never reorders or drops a
record. Products the web UI filters out (price <= 1, price > 200k) stay here:
seven evaluation tasks target them, and a catalogue that silently lost them
would score those tasks as unwinnable.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shoprl.settings import get_settings

RAW_NAME = "fine_items_eval_train_all.json.gz"
CACHE_NAME = "shop_products.jsonl.gz"
MANIFEST_NAME = "shop_products.manifest.json"


@dataclass(slots=True)
class ProductOption:
    value: str
    price: float | None = None
    image: str = ""


@dataclass(slots=True)
class Product:
    """One catalogue row, task id preserved as ``index``."""

    index: int
    asin: str
    title: str
    shop_name: str
    category: str
    domain: str
    attributes: list[str] = field(default_factory=list)
    description: str = ""
    images: list[str] = field(default_factory=list)
    pricing: list[float] = field(default_factory=list)
    options: dict[str, list[ProductOption]] = field(default_factory=dict)
    query: str = ""
    tag: str = ""
    instruction: str = ""
    instruction_simple: str = ""
    instruction_attributes: list[str] = field(default_factory=list)
    instruction_options: list[str] = field(default_factory=list)
    persona: dict[str, Any] | None = None

    @property
    def price(self) -> float:
        return self.pricing[0] if self.pricing else 0.0

    @property
    def price_tag(self) -> str:
        if len(self.pricing) > 1:
            return f"{self.pricing[0]} to {self.pricing[1]}"
        return f"{self.price}"

    def option_price(self, value: str) -> float | None:
        for entries in self.options.values():
            for option in entries:
                if option.value == value and option.price is not None:
                    return option.price
        return None

    @property
    def has_persona(self) -> bool:
        return bool(self.persona)

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "asin": self.asin,
            "title": self.title,
            "shop_name": self.shop_name,
            "category": self.category,
            "domain": self.domain,
            "attributes": self.attributes,
            "description": self.description,
            "images": self.images,
            "pricing": self.pricing,
            "options": {
                name: [
                    {"value": option.value, "price": option.price, "image": option.image}
                    for option in entries
                ]
                for name, entries in self.options.items()
            },
            "query": self.query,
            "tag": self.tag,
            "instruction": self.instruction,
            "instruction_simple": self.instruction_simple,
            "instruction_attributes": self.instruction_attributes,
            "instruction_options": self.instruction_options,
            "persona": self.persona,
        }

    @classmethod
    def from_json(cls, record: dict[str, Any]) -> Product:
        return cls(
            index=int(record["index"]),
            asin=str(record["asin"]),
            title=str(record.get("title") or ""),
            shop_name=_clean(record.get("shop_name")),
            category=_clean(record.get("category")),
            domain=_clean(record.get("domain")),
            attributes=[a for a in (_clean(x) for x in record.get("attributes") or []) if a],
            description=_clean(record.get("description")),
            images=list(record.get("images") or []),
            pricing=[float(p) for p in record.get("pricing") or []],
            options={
                str(name): [
                    ProductOption(
                        value=_clean(option.get("value")),
                        price=option.get("price"),
                        image=option.get("image") or "",
                    )
                    for option in entries
                ]
                for name, entries in (record.get("options") or {}).items()
            },
            query=str(record.get("query") or ""),
            tag=str(record.get("tag") or ""),
            instruction=str(record.get("instruction") or ""),
            instruction_simple=str(record.get("instruction_simple") or ""),
            instruction_attributes=list(record.get("instruction_attributes") or []),
            instruction_options=list(record.get("instruction_options") or []),
            persona=record.get("persona") or None,
        )


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().strip(",").split())


def _first_number(values: Any) -> float | None:
    if isinstance(values, (int, float)):
        return float(values)
    if isinstance(values, list):
        for item in values:
            if isinstance(item, (int, float)):
                return float(item)
    return None


def record_to_product(index: int, record: dict[str, Any]) -> Product:
    """Map one upstream record onto a catalogue entry, keeping every option."""
    instruction = (record.get("instructions") or [{}])[0]
    options: dict[str, list[ProductOption]] = {}
    for name, entries in (record.get("customization_options") or {}).items():
        values = [
            ProductOption(
                value=_clean(entry.get("value")).replace("/", " | "),
                price=_first_number(entry.get("price")),
                image=entry.get("image") or "",
            )
            for entry in entries or []
        ]
        values = [value for value in values if value.value]
        if values:
            options[_clean(name)] = values

    pricing = [float(p) for p in (record.get("pricing") or []) if isinstance(p, (int, float))]
    return Product(
        index=index,
        asin=_clean(record.get("asin")),
        title=_clean(record.get("title")),
        shop_name=_clean(record.get("shop_name")),
        category=_clean(record.get("category")),
        domain=_clean(record.get("domain_zh")) or _clean(record.get("domain_en_long")),
        attributes=[a for a in (_clean(x) for x in record.get("attribute") or []) if a],
        description=_clean(record.get("full_description")),
        images=list(record.get("images") or []),
        pricing=pricing,
        options=options,
        query="",
        tag=_clean(record.get("tag")),
        instruction=_clean(instruction.get("instruction")),
        instruction_simple=_clean(instruction.get("instruction_simple")),
        instruction_attributes=[_clean(x) for x in instruction.get("attributes") or []],
        instruction_options=[_clean(x) for x in instruction.get("options") or []],
        persona=record.get("user_persona") or None,
    )


class Catalog:
    """Indexed product list; ``by_asin`` and the index space both stay intact."""

    def __init__(self, products: list[Product]):
        self.products = products
        self.by_asin = {product.asin: product for product in products}
        self.by_index = {product.index: product for product in products}

    def __len__(self) -> int:
        return len(self.products)

    def __iter__(self) -> Iterator[Product]:
        return iter(self.products)

    def get(self, asin: str) -> Product | None:
        return self.by_asin.get(asin)

    def at(self, index: int) -> Product | None:
        return self.by_index.get(index)

    def manifest(self) -> dict[str, Any]:
        digest = hashlib.sha256()
        for product in self.products:
            digest.update(product.asin.encode())
        return {
            "schema_version": 1,
            "products": len(self.products),
            "tasks": len({product.index for product in self.products}),
            "eval_tasks": sum(1 for product in self.products if product.tag == "eval"),
            "train_tasks": sum(1 for product in self.products if product.tag == "train"),
            "persona_tasks": sum(1 for product in self.products if product.has_persona),
            "asin_digest": digest.hexdigest(),
        }


def build_catalog(
    raw_path: Path | None = None,
    cache_path: Path | None = None,
) -> dict[str, Any]:
    """Convert the upstream release into ``shop_products.jsonl.gz`` once."""
    data_dir = get_settings().data_dir
    raw_path = raw_path or data_dir / "raw" / RAW_NAME
    cache_path = cache_path or data_dir / "generated" / CACHE_NAME
    if not raw_path.exists():
        raise FileNotFoundError(f"missing upstream data: {raw_path}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if raw_path.suffix == ".gz" else open
    count = 0
    # The raw file is a single pretty-printed array (~140 MB expanded); reading it
    # whole costs a few hundred MB once, which is cheaper than a streaming parser.
    with opener(raw_path, "rt", encoding="utf-8") as handle:
        records = json.load(handle)

    with gzip.open(cache_path, "wt", encoding="utf-8") as sink:
        for index, record in enumerate(records):
            product = record_to_product(index, record)
            if not product.asin or not product.title:
                continue
            sink.write(json.dumps(product.to_json(), ensure_ascii=False) + "\n")
            count += 1

    manifest = {
        "source": str(raw_path),
        "cache": str(cache_path),
        "records": len(records),
        "products": count,
    }
    (cache_path.parent / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


_CACHED: Catalog | None = None


def load_catalog(cache_path: Path | None = None, *, rebuild: bool = False) -> Catalog:
    """Load the converted catalogue, converting it first when absent."""
    global _CACHED
    cache_path = cache_path or get_settings().data_dir / "generated" / CACHE_NAME
    if _CACHED is not None and not rebuild:
        return _CACHED
    if rebuild or not cache_path.exists():
        build_catalog(cache_path=cache_path)
    with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
        products = [Product.from_json(json.loads(line)) for line in handle if line.strip()]
    _CACHED = Catalog(products)
    return _CACHED
