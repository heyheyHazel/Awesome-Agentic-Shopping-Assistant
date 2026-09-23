"""Inventory agent: availability check, low-stock alerts, and purchase limits."""

from __future__ import annotations

from typing import Any

from config import get_settings
from models.schemas import InventoryItem, InventoryResult, Product

from .base_agent import BaseAgent

LOW_STOCK_THRESHOLD = 10
LOW_STOCK_LIMIT = 2
CRITICAL_STOCK_LIMIT = 1


class InventoryAgent(BaseAgent):
    """Turns a product shortlist into an inventory report for the UI."""

    def __init__(self):
        super().__init__(name="inventory", timeout=get_settings().agent_timeout_default)

    async def _execute(self, products: list[Product] | None = None, **_: Any) -> InventoryResult:
        """Classify each product by stock level and derive purchase limits."""
        items: list[InventoryItem] = []
        for product in products or []:
            if product.stock <= 0:
                items.append(InventoryItem(
                    product_id=product.product_id, name=product.name,
                    stock=0, status="out_of_stock",
                ))
                continue

            if product.stock <= LOW_STOCK_THRESHOLD:
                status, limit = "low_stock", CRITICAL_STOCK_LIMIT
            else:
                status, limit = "in_stock", None
            items.append(InventoryItem(
                product_id=product.product_id, name=product.name,
                stock=product.stock, status=status, purchase_limit=limit,
            ))

        available = [item.product_id for item in items if item.status != "out_of_stock"]
        return InventoryResult(items=items, available_ids=available)
