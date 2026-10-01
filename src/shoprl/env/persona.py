"""Render a ShopSimulator persona document as the prompt block the agent reads.

The release ships a deeply nested Chinese profile per shopper: demographics,
region, recent behaviour, transaction history, declared preferences and free-text
tags. Handing all of it to the model costs thousands of tokens per turn and
mostly repeats the same "低" preference level, so this keeps the fields that can
change a decision and drops the rest.

Only 4,666 of 23,421 tasks carry a persona, so the block is optional everywhere
and never assumed by the environment.
"""

from __future__ import annotations

from typing import Any

# Preference strength, strongest first, mirroring upstream's ordering.
STRENGTH = {"高": 0, "中": 1, "低": 2}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().strip(",").split())


def _mapping(value: Any) -> dict[str, Any]:
    """Upstream persona fields are sometimes a string where a dict is expected."""
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _preferred(mapping: Any, limit: int = 3) -> list[str]:
    if not isinstance(mapping, dict):
        return []
    ranked = sorted(
        ((STRENGTH.get(_clean(level), 3), _clean(name)) for name, level in mapping.items()),
        key=lambda item: (item[0], item[1]),
    )
    return [name for level, name in ranked if name and level <= 1][:limit]


def _brands(entries: Any, limit: int = 3) -> list[str]:
    if not isinstance(entries, list):
        return []
    ranked = sorted(
        ((STRENGTH.get(_clean(item.get("偏好程度")), 3), _clean(item.get("品牌名称"))) for item in entries if isinstance(item, dict)),
        key=lambda item: (item[0], item[1]),
    )
    return [name for level, name in ranked if name and level <= 1][:limit]


def summarise(persona: dict[str, Any] | None) -> str:
    """A short profile block, or "" when the task carries no persona."""
    if not persona:
        return ""

    demographics = _mapping(persona.get("人口属性"))
    region = _mapping(persona.get("地区信息"))
    trading = _mapping(persona.get("交易特征"))
    interest = _mapping(persona.get("兴趣偏好"))
    attributes = _mapping(interest.get("商品属性偏好"))
    band = _mapping(attributes.get("价格区间"))

    facts: list[tuple[str, str]] = [
        ("会员等级", _clean(demographics.get("会员等级"))),
        ("性别", _clean(demographics.get("性别"))),
        ("年龄段", _clean(demographics.get("年龄段"))),
        ("消费等级", _clean(demographics.get("消费等级"))),
        ("地区", "".join(filter(None, [_clean(region.get("省份")), _clean(region.get("城市"))]))),
        ("近90天订单数", _clean(trading.get("近90天订单数"))),
        ("复购率", _clean(trading.get("复购率"))),
        ("是否促销敏感", "是" if trading.get("是否促销敏感") else ""),
    ]
    lines = [f"{label}: {value}" for label, value in facts if value]

    preferences = [
        ("偏好类目", "、".join(_preferred(interest.get("类目偏好")))),
        ("偏好品牌", "、".join(_brands(interest.get("品牌偏好")))),
        ("偏好风格", "、".join(_clean(x) for x in _list(attributes.get("风格")))),
        ("偏好颜色", "、".join(_clean(x) for x in _list(attributes.get("颜色")))),
        ("偏好材质", "、".join(_clean(x) for x in _list(attributes.get("材质")))),
    ]
    lines += [f"{label}: {value}" for label, value in preferences if value]

    low, high = band.get("最小值"), band.get("最大值")
    if low is not None and high is not None:
        lines.append(f"常购价格区间: {_clean(low)}-{_clean(high)}")

    tags = [_clean(tag) for tag in _list(persona.get("用户标签")) if _clean(tag)]
    if tags:
        lines.append("用户标签: " + "、".join(tags))

    if not lines:
        return ""
    return "本单顾客的画像：\n" + "\n".join(f"- {line}" for line in lines)
