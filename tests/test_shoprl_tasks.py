"""Task pools: disjoint, sized, deterministic."""

from __future__ import annotations

from shoprl.env.catalog import Catalog, Product
from shoprl.env.tasks import build_task_pools, load_task_pool


def make_product(index: int, tag: str, category: str, options: list[str]) -> Product:
    return Product(
        index=index,
        asin=f"A{index:05d}",
        title=f"商品 {index}",
        shop_name="店铺",
        category=category,
        domain="测试",
        attributes=["属性"],
        pricing=[10.0 + index],
        instruction=f"买商品 {index}",
        instruction_attributes=["属性"],
        instruction_options=options,
        tag=tag,
    )


def build_catalog(count: int = 400) -> Catalog:
    categories = ["服饰›上衣›卫衣", "家居›水具›保温杯", "食品›零食›坚果"]
    products = []
    for index in range(count):
        tag = "eval" if index < 40 else "train"
        products.append(
            make_product(index, tag, categories[index % len(categories)], ["黑色"][: 1 + index % 3])
        )
    return Catalog(products)


def test_pools_are_disjoint_and_cover_the_requested_sizes(tmp_path):
    manifest = build_task_pools(
        build_catalog(), out_dir=tmp_path, sizes={"dev": 10, "sft": 20, "rl": 15}
    )

    assert manifest["pools"] == {
        "official_test": 40,
        "official_test_persona": 0,
        "dev": 10,
        "sft": 20,
        "rl": 15,
    }
    ids = {name: {task.task_id for task in load_task_pool(name, tmp_path / f"{name}.jsonl")}
           for name in manifest["pools"]}
    assert ids["official_test"].isdisjoint(ids["dev"] | ids["sft"] | ids["rl"])
    assert ids["dev"].isdisjoint(ids["sft"] | ids["rl"])
    assert ids["sft"].isdisjoint(ids["rl"])
    assert all(task_id >= 40 for task_id in ids["sft"] | ids["rl"] | ids["dev"])


def test_pools_are_reproducible_and_cover_several_categories(tmp_path):
    first = build_task_pools(build_catalog(), out_dir=tmp_path / "a", sizes={"dev": 10, "sft": 20, "rl": 15})
    second = build_task_pools(build_catalog(), out_dir=tmp_path / "b", sizes={"dev": 10, "sft": 20, "rl": 15})

    assert first["pools"] == second["pools"]
    left = [task.to_json() for task in load_task_pool("sft", tmp_path / "a" / "sft.jsonl")]
    right = [task.to_json() for task in load_task_pool("sft", tmp_path / "b" / "sft.jsonl")]
    assert left == right
    assert first["categories"]["sft"] > 1
