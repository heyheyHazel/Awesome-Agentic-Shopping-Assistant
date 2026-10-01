"""ShopSimulator reference environment: catalogue, session, reward, task pools."""

from shoprl.env.catalog import Catalog, Product, ProductOption, build_catalog
from shoprl.env.local import EnvPool, LocalShopEnv
from shoprl.env.reward import SUB_SCORES, score_purchase
from shoprl.env.tasks import Task, build_task_pools, load_task_pool

__all__ = [
    "Catalog",
    "EnvPool",
    "LocalShopEnv",
    "Product",
    "ProductOption",
    "SUB_SCORES",
    "Task",
    "build_catalog",
    "build_task_pools",
    "load_task_pool",
    "score_purchase",
]

