"""Stock availability, low-stock alerts and purchase limits."""

from __future__ import annotations

from shopping_assistant.domain.models import InventoryItem, InventoryResult, Product

LOW_STOCK_THRESHOLD = 10
CRITICAL_STOCK_LIMIT = 1


def check_stock(products: list[Product] | None = None) -> InventoryResult:
    """Classify each product by stock level and derive its purchase limit."""
    items: list[InventoryItem] = []
    for product in products or []:
        if product.stock <= 0:
            items.append(
                InventoryItem(
                    product_id=product.product_id,
                    name=product.name,
                    stock=0,
                    status="out_of_stock",
                )
            )
            continue

        if product.stock <= LOW_STOCK_THRESHOLD:
            status, limit = "low_stock", CRITICAL_STOCK_LIMIT
        else:
            status, limit = "in_stock", None
        items.append(
            InventoryItem(
                product_id=product.product_id,
                name=product.name,
                stock=product.stock,
                status=status,
                purchase_limit=limit,
            )
        )

    available = [item.product_id for item in items if item.status != "out_of_stock"]
    return InventoryResult(items=items, available_ids=available)
