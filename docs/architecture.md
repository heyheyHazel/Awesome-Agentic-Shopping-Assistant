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
  ├── SupervisorAgent   LLM 结构化输出 → 执行计划
  ├── UserProfileAgent  确定性 RFM 计算
  ├── ProductRecAgent   目录召回 + LLM 重排
  ├── InventoryAgent    库存规则
  ├── MarketingCopyAgent LLM 结构化文案
  └── ChatAgent         create_agent 工具调用
      │
      ▼
OpenAI 兼容 LLM API（DeepSeek / OpenAI / …）
```

## 2. 图结构

```
START → supervisor
          ├─ general        → assistant → END
          └─ product_search → [profile?, recall]
                                     ↓        ↓
                                  rerank    inventory
                                     ↓        ↓
                                 aggregate
                                     ├─ copy? → marketing
                                     └────────→ respond → END
```

关键设计：

- **条件扇出**：`route_supervisor` 返回目标节点列表（LangGraph 支持列表形式的扇出），`profile` 只在计划包含时执行，未触发的分支不会阻塞下游汇合（已用测试验证）
- **隐式汇合**：`rerank` / `inventory` 都有来自 `profile` 与 `recall` 的入边，LangGraph 保证汇合节点每个超步只执行一次，不会重复调用 LLM
- **条件跳过**：`aggregate` 之后按计划决定是否进入 `marketing`，快速事实类问题可以省掉一次 LLM 文案调用

## 3. 图状态（GraphState）

| 字段 | 类型 | 说明 |
|------|------|------|
| `messages` | `Annotated[list[AnyMessage], add_messages]` | 对话历史，随 Checkpointer 持久化 |
| `user_id` / `query` | `str` | 当前请求上下文 |
| `plan` | `dict` | SupervisorPlan 序列化结果 |
| `profile` | `UserProfile \| None` | 画像节点输出 |
| `candidates` | `list[Product]` | 召回候选 |
| `ranked` | `list[Product]` | LLM 重排结果 |
| `inventory` / `available_ids` | 列表 | 库存节点输出 |
| `copies` | `list[CopyItem]` | 文案节点输出 |
| `final_products` | `list[Product]` | 聚合后的展示商品 |
| `reply` | `str` | 本轮回复文本 |
| `timings` | `Annotated[dict, merge_dicts]` | 各节点耗时，使用自定义 reducer 合并并行写入 |

`supervisor_node` 每轮会重置所有瞬态字段（profile / candidates / …），避免上一轮的残留数据污染本轮。

## 4. 节点职责与事件

| 节点 | LLM | 发出事件 | 失败降级 |
|------|-----|----------|----------|
| `supervisor` | 结构化计划 | `agent`(×2), `plan`, `experiment` | 计划回退为 general + 请用户重述 |
| `assistant` | 工具调用流式 | `agent`(×2), `token` | 节点内捕获异常，输出错误提示 |
| `profile` | 确定性 | `agent`(×2), `profile` | 无画像继续（重排按通用规则） |
| `recall` | 确定性 | `agent`(running) | 关键词无命中时回退到全目录按评分排序 |
| `rerank` | 结构化排序 | `agent`(done) | 返回空 ranked，聚合使用召回顺序 |
| `inventory` | 确定性 | — | 输出超时则跳过库存过滤 |
| `aggregate` | — | `products` | 过滤后为空时回退到完整排名 |
| `marketing` | 结构化文案 | `agent`(×2), `marketing` | 返回空列表，跳过文案气泡 |
| `respond` | 流式总结 | `agent`(×2), `inventory`, `token` | token 失败输出兜底文本 |

`respond` 在流式输出前补发 `inventory` 事件（只含最终商品），保证前端聊天内的消息顺序为：商品 → 文案 → 库存 → 总结。

## 5. 动态调度规则

`SupervisorAgent` 用 `JsonStructured` 让 LLM 输出 `SupervisorPlan`：

```json
{
  "intent": "product_search | general",
  "reply": "一句即时确认，如 Got it! Here are the best running shoes for you.",
  "search": { "keywords": [], "category": "", "brand": "", "min_price": null, "max_price": 120 },
  "agents": ["profile", "recall", "rerank", "inventory", "copy"]
}
```

- `route_supervisor`：`intent=general` → `assistant`；否则扇出 `recall`（必需）+ `profile`（计划包含时）
- `route_aggregate`：`agents` 含 `copy` 才进入 `marketing`
- 计划提示词要求：产品检索必须含 `recall/rerank/inventory`；大多数情况加 `profile`；需要营销文案才加 `copy`；闲聊为空列表

## 6. 流式协议

FastAPI 将 LangGraph 的两种流映射为 SSE：

```python
async for mode, chunk in graph.astream(inputs, config=config, stream_mode=["custom", "updates"]):
    if mode == "custom":   yield _sse(chunk["type"], chunk)     # 业务事件
    else:                  merge timings from chunk              # 状态增量
yield _sse("done", {"latency_ms": ..., "timings": {...}})
```

- 节点通过 `get_stream_writer()` 发出 custom 事件（`emit()` 封装，非流式 `ainvoke` 下自动静默）
- token 由 `assistant` / `respond` 两个节点显式转发，因此监督者/重排等结构化调用的中间 token 不会泄漏到前端

前端 `useAgentStream` 的顺序处理：

```
session → (agent 状态更新 | plan → Supervisor 气泡) → profile(右栏)
        → products(卡片) → marketing(气泡) → inventory(气泡)
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

```
recency   = max(0, 1 - 距上次购买天数 / 90)
frequency = min(1, 购买次数 / 12)
monetary  = min(1, 累计消费 / 1500)
overall   = 0.3·recency + 0.3·frequency + 0.4·monetary
```

客群判定（自上而下命中即返回）：

1. `orders ≤ 2` → **New**
2. `recency_days > 60` → **At Risk**
3. `orders ≥ 10 且 消费 ≥ 1000 且 recency ≤ 14` → **Champions**
4. `orders ≥ 5 且 recency ≤ 45` → **Loyal**
5. 其余 → **Potential**

## 11. 前端数据流

```
useAgentStream(userId)
  ├── feed          → ChatPanel（user / agent / products / inventory 四种块）
  ├── agentStates   → AgentPanel（idle / running / done 状态灯）
  ├── profile       → ProfilePanel 画像 + RFM 聚类
  ├── experiment    → ProfilePanel A/B 面板
  └── latencyMs/timings → ProfilePanel 响应耗时卡片
```

用户切换时重新拉取画像与实验数据并重置会话；所有流式写入都在单个 `send()` 的事件回调中完成。

## 12. 扩展点

- **新增 Agent**：实现 `BaseAgent` 子类 → 在图中注册节点与边 → 把名字加入计划提示词的 `agents` 枚举
- **新增工具**：在 `chat_agent.py` 用 `@tool` 装饰函数并加入 `create_agent` 的 tools 列表
- **持久化记忆**：把 `build_graph()` 的 checkpointer 换成 `SqliteSaver` / Postgres 实现
- **真实数据**：替换 `data/products.py` 与 `data/users.py`，或在画像节点接入数据库/特征服务
- **可观测性**：设置 `LANGSMITH_TRACING=true` 即获得全链路追踪
