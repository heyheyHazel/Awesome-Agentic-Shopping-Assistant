# 架构设计文档

本文档描述 v2 架构：LangGraph 动态调度 + 并行 Agent 流水线 + SSE 实时流。

## 1. 运行时拓扑

```
浏览器 (React 19)
  │  POST /api/v1/chat (SSE)
  ▼
FastAPI (python/main.py)
  │  graph.astream(stream_mode=["custom", "updates"])
  ▼
LangGraph 状态图 (python/orchestrator/graph.py)
  ├── SupervisorAgent   LLM 结构化输出 → 意图 + 检索参数
  ├── UserProfileAgent  确定性分位 RFM 计算
  ├── ProductRecAgent   目录召回 + LLM 排序与文案
  ├── InventoryAgent    库存规则
  └── ChatAgent         create_agent 工具调用
      │
      ▼
OpenAI 兼容 LLM API（DeepSeek / OpenAI / …）
```

## 2. 图结构

```
START → supervisor                                    ← LLM ①
          ├─ general        → assistant → END
          └─ product_search → profile → recall → inventory → filter
                                                              │
                                ┌─────────────────────────────┴──────────┐
                      候选非空  │                                        │ 无匹配
                                ▼                                        ▼
                      recommend → respond → END                  respond → END
                       ↑ LLM ②
```

关键设计：

- **每轮两次 LLM 调用**：查询理解（`supervisor`）与排序 + 文案（`recommend`）。其余节点是确定性代码，实测每项 0.0–1.4 ms
- **单链而非扇出**：确定性节点串成一条直线——它们都是毫秒级，并行不会更快。更重要的是，LangGraph 中一个有多条不同层级入边的节点会**每个超步执行一次**；把 `profile → filter` 与 `inventory → filter` 分开连（`inventory` 比 `profile` 晚一个超步完成）会让下游 LLM 调用静默翻倍（`tests/test_graph.py` 的调用计数测试守住这一点）
- **单调收窄**：`candidates` 从召回开始逐级收窄（库存过滤 → 候选上限），排序前就剔除缺货商品，不让模型为买不到的东西花一次调用
- **条件跳过**：`filter` 后候选为空时直接进入 `respond`，省掉整次排序调用

## 3. 图状态（GraphState）

| 字段 | 类型 | 说明 |
|------|------|------|
| `messages` | `Annotated[list[AnyMessage], add_messages]` | 对话历史，随 Checkpointer 持久化 |
| `user_id` / `query` | `str` | 当前请求上下文 |
| `plan` | `dict` | SupervisorPlan 序列化结果 |
| `profile` | `UserProfile \| None` | 画像节点输出 |
| `candidates` | `list[Product]` | 召回候选，经库存过滤与候选上限逐级收窄 |
| `inventory` / `available_ids` | 列表 | 库存节点输出 |
| `pitches` | `list[CopyItem]` | 排序调用同时产出的推荐语 |
| `final_products` | `list[Product]` | 聚合后的展示商品 |
| `reply` | `str` | 本轮回复文本 |
| `timings` | `Annotated[dict, merge_dicts]` | 各节点耗时，使用自定义 reducer 合并并行写入 |

`supervisor_node` 每轮会重置所有瞬态字段（profile / candidates / …），避免上一轮的残留数据污染本轮。

## 4. 节点职责与事件

| 节点 | LLM | 发出事件 | 失败降级 |
|------|-----|----------|----------|
| `supervisor` | 结构化计划 | `agent`(×2), `plan`, `experiment` | 计划回退为 general + 请用户重述 |
| `assistant` | 工具调用流式 | `agent`(×2), `token` | 节点内捕获异常，输出错误提示 |
| `profile` | 确定性 | `agent`(×2), `profile` | 无画像继续（排序按通用规则） |
| `recall` | 确定性 | `agent`(running) | 约束全不命中时返回空列表，**不**放宽预算/类目 |
| `inventory` | 确定性 | `agent`(×2) | 输出超时则跳过库存过滤 |
| `filter` | 确定性 | — | 候选为空则不进入排序 |
| `recommend` | 结构化：排序 + 文案 | `agent`(×2) | 返回空结果，`respond` 回退到候选顺序 |
| `respond` | — | `products`, `inventory`, `agent`(×2), `token` | 无商品时不发 token |

`respond` 不发 LLM 调用：它在发出 `products` 与 `inventory` 后，把 `recommend` 已经写好的推荐语分块作为 `token` 事件推给前端，保证聊天内消息顺序为：商品卡片 → 库存 → 逐字推荐语。

## 5. 计划与路由

`SupervisorAgent` 用 `JsonStructured` 让 LLM 输出 `SupervisorPlan`：

```json
{
  "intent": "product_search | general",
  "reply": "一句即时确认，如 Got it! Here are the best running shoes for you.",
  "search": { "keywords": [], "category": "", "brand": "", "min_price": null, "max_price": 120 }
}
```

- `route_supervisor`：`intent=general` → `assistant`，否则走 `profile` 起头的确定性流水线
- `route_filter`：候选非空 → `recommend`，否则直接 `respond`
- 计划里不再有 `agents` 枚举：哪些阶段能省由代码决定（候选为空就跳过排序），不需要模型判断
- `catalog_language_rule()` 依据**实际加载的商品类目**推断检索词语言（ShopSimulator 中文目录 vs 内置英文演示目录），不再写死英文

## 6. 流式协议

FastAPI 将 LangGraph 的两种流映射为 SSE：

```python
async for mode, chunk in graph.astream(inputs, config=config, stream_mode=["custom", "updates"]):
    if mode == "custom":   yield _sse(chunk["type"], chunk)     # 业务事件
    else:                  merge timings from chunk              # 状态增量
yield _sse("done", {"latency_ms": ..., "timings": {...}})
```

- 节点通过 `get_stream_writer()` 发出 custom 事件（`emit()` 封装，非流式 `ainvoke` 下自动静默）
- token 只有两个来源：`assistant`（工具调用 Agent 的真实流式输出）与 `respond`（分块推送已生成的推荐语）；结构化调用的中间 token 不会泄漏到前端

前端 `useAgentStream` 的顺序处理：

```
session → (agent 状态更新 | plan → Supervisor 气泡) → profile(右栏)
        → products(卡片) → inventory(气泡)
        → token(逐字追加到 Assistant 气泡) → done(耗时)
```

## 7. 稳定性设计

| 机制 | 实现 | 参数 |
|------|------|------|
| 超时熔断 | `BaseAgent` 中 `asyncio.wait_for` 包裹 `_execute` | 默认 8s，LLM 节点 25s，对话 45s |
| 指数退避重试 | `tenacity`（0.5s 起步，上限 4s） | 2 次尝试 |
| 降级 | 每个 Agent 覆写 `_fallback()` | 见第 4 节表格 |
| 请求级兜底 | `/api/v1/chat` 捕获异常发 `error` 事件 | 保证 SSE 正常结束 |

## 8. 会话记忆

- `build_graph()` 默认挂载 `InMemorySaver`（可注入其他 Checkpointer）
- 会话由请求里的 `thread_id` 标识；前端首次请求拿到 `session` 事件后复用，点击「New chat」即丢弃
- 多轮上下文（如「cheaper ones」）由 Supervisor 读取 `messages` 历史解析

## 9. A/B 测试

- **分桶**：`md5(user_id + experiment_id) % n_variants`，同一用户永远在同一实验组（`ABTestEngine.assign`）
- **动态调权**：`sample()` 从各组 Beta 后验采样取最大（Thompson Sampling）
- **记录结果**：`POST /api/v1/experiments/outcome` 更新后验
- 演示数据预置 A(313/186)、B(370/128)，显示转化率 62.7% / 74.3%

## 10. RFM 与客群规则

分数是**相对客群的分位排名**，不是绝对分数——20 单在「中位数 4 单」的客群里是高频客户，在「中位数 40 单」的客群里只是普通客户，所以绝对阈值一旦换数据集就会退化（实测在 4009 个真实用户上，旧公式把 78% 的人塞进同一个客群，且 `New` 恒为 0）。

```
RFMScale 从当前用户总体预计算三组排序数组
recency   = 1 - percentile(recency_days)      # 天越少越好，所以要取反
frequency = percentile(orders)
monetary  = percentile(lifetime_value)
overall   = 0.3·recency + 0.3·frequency + 0.4·monetary
```

客群判定（自上而下命中即返回）：

1. `frequency ≤ 0.30` → **New**
2. `recency ≥ 0.50 且 frequency ≥ 0.66 且 monetary ≥ 0.66` → **Champions**
3. `frequency ≥ 0.66 且 monetary ≥ 0.50` → **Loyal**
4. `recency ≤ 0.33` → **At Risk**
5. 其余 → **Potential**

顺序很重要：正向信号优先于流失判定，否则一个刚沉默不久的高频客户会被判成 At Risk。

## 11. 多语言（i18n）

- 前端 `src/i18n.ts` 保存全部界面文案词典与类目/标签/客群映射；`i18n-provider.tsx` 提供 `useI18n()` hook
- 语言优先级：URL `?lang=zh` > `localStorage` > 默认 `en`；切换时写入 `localStorage` 与 `<html lang>`
- 每个聊天请求携带 `language` 字段：
  - 调度 / 回复 / 通用问答的 system prompt 追加 `language_directive()`
  - 调度 Agent 额外收到 `catalog_language_rule()`，它按实际加载类目是否含中文来要求 `search.keywords` / `search.category` 使用对应语言（ShopSimulator 中文目录 vs 内置英文演示目录）
- 商品型号保留英文，类目、标签、库存状态、A/B 文案等均由前端词典翻译

## 12. 前端数据流

```
useAgentStream(userId)
  ├── feed          → ChatPanel（user / agent / products / inventory 四种块）
  ├── agentStates   → AgentPanel（idle / running / done 状态灯）
  ├── profile       → ProfilePanel 画像 + RFM 聚类
  ├── experiment    → ProfilePanel A/B 面板
  └── latencyMs/timings → ProfilePanel 响应耗时卡片
```

用户切换时重新拉取画像与实验数据并重置会话；所有流式写入都在单个 `send()` 的事件回调中完成。

## 13. 扩展点

- **新增 Agent**：实现 `BaseAgent` 子类 → 在图中注册节点与边。如果它需要 LLM，先想清楚能否并入 `supervisor` 或 `recommend`——每多一个 LLM 节点就多一次秒级往返
- **新增工具**：在 `chat_agent.py` 用 `@tool` 装饰函数并加入 `create_agent` 的 tools 列表
- **持久化记忆**：把 `build_graph()` 的 checkpointer 换成 `SqliteSaver` / Postgres 实现
- **换数据源**：`ECOM_DATA_SOURCE=auto|real|mock`（见 `data/store.py`）；拉取脚本见 `scripts/fetch_data.py`
- **可观测性**：设置 `LANGSMITH_TRACING=true` 即获得全链路追踪

## 14. 召回与语义索引

```
用户查询 ──┬─► 硬过滤（预算 / 类目 / 品牌）→ eligible
           │
           ├─► 关键词召回：product_text 子串命中数 × 2 + 评分      → 全量排序
           ├─► 语义召回：bge-small-zh-v1.5 512 维余弦，掩码在 eligible 内 → top 50
           │
           └─► RRF 融合（k=60）→ 出口再校验一次硬过滤 → top 12
```

**为什么用 RRF 而不是加权求和**：需要给「关键词命中数」和「余弦相似度」标定相对权重，而两者量纲完全无关。RRF 只用名次，天然免标定，且两路都找到的商品会自动上浮。

**为什么掩码而不是先检索再过滤**：先检索的话，一次超预算查询的 top-50 可能全部落在预算外，过滤后就没结果了；掩码保证语义排序只在用户已经认可的集合里挑最相关的。

**索引失效策略**：索引以 `product_id` 为行的连接键。加载时校验 id 集合与当前目录是否一致，不一致就视为不存在（回落关键词召回）。这样换数据集不会静默排错行，代价是必须重跑 `scripts/build_index.py`。

**没有相似度阈值**：实测在真实目录上不可分（应命中 top-1 最低 0.600，应不命中最高 0.602；换成 z 分数同样重叠）。因此「没有合适商品」的判定交给 `recommend` 的结构化输出——返回空 `product_ids` 即为刻意判空，此时 `respond` 不发商品事件也不发库存事件，前端显示「没有符合条件的商品」。注意这与调用失败不同：`_fallback` 的 `success=False` 仍会回落到候选顺序。
