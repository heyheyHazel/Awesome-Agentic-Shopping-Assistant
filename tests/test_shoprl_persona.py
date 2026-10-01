"""Persona rendering: a compact block, and an honest one for malformed input."""

from __future__ import annotations

from shoprl.env.catalog import Catalog, Product
from shoprl.env.local import EnvPool
from shoprl.env.persona import summarise
from shoprl.env.search import Bm25Index
from shoprl.env.tasks import build_task_pools, load_task_pool
from shoprl.env.tools import ShopToolkit

RAW = {
    "用户ID": "U1, ",
    "地区信息": {"省份": "浙江省", "城市": "杭州市", "区县": "滨江区"},
    "人口属性": {"性别": "女", "年龄段": "35-44, ", "消费等级": "中", "会员等级": "黄金会员"},
    "交易特征": {"近90天订单数": 28, "复购率": 0.62, "是否促销敏感": True},
    "兴趣偏好": {
        "类目偏好": {"母婴用品": "高", "家居生活": "中", "运动户外": "低"},
        "品牌偏好": [{"品牌名称": "梦洁宝贝", "偏好程度": "高"}, {"品牌名称": "杂牌", "偏好程度": "低"}],
        "商品属性偏好": {
            "价格区间": {"最小值": 0, "最大值": 1200},
            "材质": ["天然乳胶"],
            "风格": ["北欧简约"],
        },
    },
    "用户标签": ["育儿刚需", "夜间活跃"],
}


def test_the_block_keeps_decisions_and_drops_the_rest():
    text = summarise(RAW)

    assert "黄金会员" in text and "杭州市" in text
    assert "母婴用品" in text and "梦洁宝贝" in text
    assert "常购价格区间: 0-1200" in text
    assert "育儿刚需" in text
    # Low-strength preferences and raw internals are noise.
    assert "运动户外" not in text
    assert "杂牌" not in text
    assert "U1" not in text


def test_a_missing_or_empty_persona_renders_nothing():
    assert summarise(None) == ""
    assert summarise({}) == ""
    assert summarise({"用户ID": "U1"}) == ""


def test_malformed_persona_fields_do_not_raise():
    assert summarise({"人口属性": "not a dict", "兴趣偏好": {"类目偏好": ["高"]}}) == ""


def build_pool(with_persona: bool) -> EnvPool:
    product = Product(
        index=0,
        asin="A0001",
        title="商品",
        shop_name="店铺",
        category="服饰›上衣›卫衣",
        domain="服饰",
        pricing=[100.0],
        instruction="买一件卫衣",
        instruction_options=["黑色"],
        persona=RAW if with_persona else None,
    )
    catalog = Catalog([product])
    return EnvPool(catalog, Bm25Index.build(catalog), capacity=1)


def test_reset_reports_the_persona_and_the_tool_decides_whether_to_show_it():
    pool = build_pool(with_persona=True)
    lease = pool.acquire(0, "s1")

    payload = lease.env.reset(0, "s1")
    assert payload["user_persona"]["人口属性"]["会员等级"] == "黄金会员"

    shown = ShopToolkit(lease.env, 0, "s1", show_persona=True)
    hidden = ShopToolkit(lease.env, 0, "s1", show_persona=False)
    assert "黄金会员" in shown.registry()["shop_reset"].handler({})
    assert "黄金会员" not in hidden.registry()["shop_reset"].handler({})


def test_task_pools_record_which_tasks_carry_a_persona(tmp_path):
    from shoprl.env.catalog import Product

    products = [
        Product(
            index=index,
            asin=f"A{index:04d}",
            title=f"商品 {index}",
            shop_name="店铺",
            category="服饰›上衣›卫衣",
            domain="服饰",
            pricing=[10.0 + index],
            instruction=f"买商品 {index}",
            instruction_options=["黑色"],
            tag="eval" if index < 4 else "train",
            persona=RAW if index % 2 == 0 else None,
        )
        for index in range(20)
    ]
    manifest = build_task_pools(
        Catalog(products), out_dir=tmp_path, sizes={"dev": 2, "sft": 4, "rl": 4}
    )

    assert manifest["persona_tasks"]["official_test"] == 2
    persona_pool = load_task_pool("official_test_persona", tmp_path / "official_test_persona.jsonl")
    assert [task.task_id for task in persona_pool] == [0, 2]
    assert all(task.has_persona for task in persona_pool)

