# Agentic Shopping Assistant · 架构设计

## 1. 系统总览

一个 tool-calling agent + 四个确定性工具。

```
用户消息
  │
  ▼
FastAPI (src/shopping_assistant/api/app.py)
  │  会话 id / A/B 分桶 / SSE 帧封装 / 按工具分解耗时
  ▼
ShoppingAgent (src/shopping_assistant/agent/shopping_agent.py)          ← 唯一的 LLM 循环
  │
  ├── get_shopper_profile(user_id)      分位 RFM 画像（确定性）
  ├── search_catalog(query, …)           混合召回 + 预算/类目硬过滤（确定性）
  ├── check_inventory(product_ids)       库存 + 限购（确定性）
  └── present_recommendation(ids)        发出商品卡片（确定性）
        │
        ▼
  每轮把 language + user_id 拼进 system prompt
        │
        ▼
  OpenAI 兼容 LLM API（DeepSeek / OpenAI / …）
```

**意图判断、query 改写、选品、写文案都由这个循环完成**——前两者体现在它填工具参数，后两者体现在它写最终回答。工具只负责模型没法凭空知道的事。

## 2. 一次购物咨询的事件流

```
session → experiment
        → tool(assistant, running)
        → tool(search_catalog, running/done)
        → tool(get_shopper_profile, running/done)   ← 顺序与是否调用由模型决定
        → tool(check_inventory, running/done)
        → tool(present_recommendation, running/done)
        → products                                   ← 卡片数据来自这个工具的返回
        → inventory
        → token × N                                  ← 最终回答
        → done {latency_ms, timings}
```

### 关键设计：卡片只来自工具的返回

`products` 事件的载荷是 `present_recommendation` 的真实返回，**不是从模型措辞里解析出来的**。如果把模型文案里的商品渲染成卡片，用户说「200 元以内」就可能在卡片上看到超预算的东西，或者模型说「四款」而界面只有三张卡。工具是唯一的真相来源，模型只负责叙述。

同理，售罄商品在 `search_catalog` 内部就被剔除；预算与类目作为工具参数传入后由 `retrieval/recall.py` 强制执行，并在出口再校验一次。

### 助手文本为什么要缓冲

一次购物咨询会产生多个 model 消息：调用工具前的「工作笔记」和最后的回答。直接把所有 token 流推给用户，会看到「I'll look up the shopper's profile and search the catalog…」这种内心独白。所以 `ShoppingAgent` 只在**最后一个没有工具调用的 assistant 消息**上放 token 事件：前言随工具调用一起被丢掉。事件顺序由 `AgentLoop(emit=...)` 的调用顺序决定，落盘的 `Trajectory` 里仍然保留完整记录，包括那些没给用户看的前言。

## 3. 工具契约

```python
def search_catalog(query: str, keywords: list[str] | None = None, category: str = "",
                   min_price: float | None = None, max_price: float | None = None,
                   limit: int = 8) -> str

# 声明给模型的那一份是 JSON Schema，写在 build_tool_registry() 里
```

- 返回给模型的是紧凑的文本行（`id | 名称 | 类目 | 价格 | 评分 | 店铺 | 标签`），模型能直接读
- 工具自己通过 `services/events.py` 的 `emit()` 发事件；`get_stream_writer()` 在非图执行环境（单测、脚本）下自动静默，所以工具仍可直接调用与测试
- `present_recommendation` 的返回值会明确告诉模型「实际展示了哪几件」，若被上限截断也会说明——否则模型会继续描述用户看不见的商品

## 4. LLM 调用次数

| 场景 | 调用次数 |
|------|----------|
| 通用问答（不调工具） | **1** |
| 购物咨询（搜索 → 呈现） | 2–3 |
| 购物咨询（+ 画像 / 库存查验） | 3–4 |

实测（DeepSeek）：通用问答 1.2–1.6s，购物咨询 3.5–4.3s，模型多搜几次时可达 8s。**旧的流水线架构固定 4 次链式调用、11.1s**；现在次数由任务复杂度决定。

## 5. 为什么是一个循环，而不是一条流水线

| 方案 | 购物咨询 LLM 调用 | 通用问答 | 确定性保证 | 代码面 |
|------|------------------|----------|-----------|--------|
| 多 Agent 流水线（旧） | 4（固定） | 2 | 强 | 5 个 Agent + LangGraph 状态图 |
| 一个循环 + 确定性工具（现） | 2–4（自适应） | **1** | 强（约束在工具里） | 1 个循环 + 4 个工具 |

这不是「单 Agent 更好」的立场，而是实测结论：

1. **没有可并行的东西**。旧版把 `rerank ∥ inventory` 并行，但 inventory 是纯内存遍历（0.9ms），一次 LLM 调用是 1–2s——并行一毫秒也省不到。
2. **任务本身简单**。不需要多 Agent 间的协商与 handoff，只需要「查数据 → 选品 → 说话」。
3. **旧的文案 Agent 是净负债**。它和最终总结在写同一批商品的同一段话，用户看到两个气泡说同一件事。

## 6. 稳定性设计

| 机制 | 实现 | 参数 |
|------|------|------|
| 超时熔断 | `ECOM_LLM_REQUEST_TIMEOUT` 传给 httpx 客户端；工具本身是纯函数，不会挂住 | 默认 60s |
| 指数退避重试 | `tenacity`（0.5s 起步，上限 4s） | 2 次尝试 |
| 降级 | 每个 Agent 覆写 `_fallback()` | 返回失败结果，调用方继续 |
| 召回降级 | 索引缺失 / 目录变化 / 模型未下载 | 自动回落纯关键词召回 |
| 请求级兜底 | `/api/v1/chat` 捕获异常发 `error` 事件 | 保证 SSE 正常结束 |
| 工具参数校验 | `search_catalog` 的 limit 夹紧；`present_recommendation` 丢弃不存在的 id | 模型编造的 id 不会进卡片 |
## 7. A/B 测试

- **分桶**：`md5(user_id + experiment_id) % n_variants`，同一用户永远在同一实验组（`ABTestEngine.assign`）
- **动态调权**：`sample()` 从各组 Beta 后验采样取最大（Thompson Sampling）
- **记录结果**：`POST /api/v1/experiments/outcome` 更新后验
- 演示数据预置 A(313/186)、B(370/128)，显示转化率 62.7% / 74.3%

## 8. RFM 与客群规则

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

## 9. 多语言（i18n）

- 前端 `src/i18n.ts` 保存全部界面文案词典与类目/标签/客群映射；`i18n-provider.tsx` 提供 `useI18n()` hook
- 语言优先级：URL `?lang=zh` > `localStorage` > 默认 `en`；切换时写入 `localStorage` 与 `<html lang>`
- 每个聊天请求携带 `language` 字段：
  - `ShoppingAgent` 每轮把这些拼进 system prompt（`language_directive()` 与当前 shopper id），两者都属于**请求**而非 Agent，所以同一份配置能同时服务所有用户与语言
  - `catalog_language_rule()` 写进 Agent 的固定 system prompt，它按实际加载类目是否含中文来判断该用哪种语言写 `query` 与 `keywords`（ShopSimulator 中文目录 vs 内置英文演示目录）
- 商品型号保留英文，类目、标签、库存状态、A/B 文案等均由前端词典翻译

## 10. 前端数据流

```
useAgentStream(userId)
  ├── feed        → ChatPanel（user / agent / products / inventory 四种块）
  ├── toolStates  → AgentPanel（每个工具的 idle / running / done 状态灯）
  ├── profile     → ProfilePanel 画像 + RFM
  ├── experiment  → ProfilePanel A/B 面板
  └── latencyMs/timings → ProfilePanel 响应耗时卡片（按工具分解）
```

`tool` 事件用工具名作为 key（`search_catalog` / `get_shopper_profile` / `check_inventory` / `present_recommendation` / `assistant`），左栏卡片就从真实的工具调用点亮，没有脚本化的假动画。

## 11. 扩展点

- **新增工具**：在 `agent/tools.py` 写一个普通函数，在 `build_tool_registry()` 里加一条 `ToolSpec`（名字 + 描述 + JSON Schema），然后在 `shopping_agent.py` 的 system prompt 里说明什么时候用它。写进 UI 的载荷一律通过 `emit()` 从工具的真实返回发出。
- **新增确定性能力**：写一个纯函数并注册成工具，**不要为了它新增 LLM 节点**——那会多一次秒级往返。
- **持久化记忆**：把 `ShoppingAgent` 的 `_sessions` 换成 Redis/SQLite 实现；需要落盘的只是 `Session.messages` 与 `Session.memory`。
- **换数据源**：`ECOM_DATA_SOURCE=auto|real|mock`（见 `catalog/store.py`）；拉取脚本见 `scripts/fetch_data.py`。
- **换召回策略**：`retrieval/recall.py` 的 `recall_products` 是唯一入口，关键词与语义两路各自排序后用 RRF 融合；加第三路只需要多传一个 ranking 给它。
- **可观测性**：每一轮都产出 `Trajectory`，里面含每个 step 发给模型的**完整 prompt**、工具返回值、耗时与终止原因；训练侧的采集脚本直接消费同一结构。

## 12. 召回与语义索引

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

## 13. 会话记忆

- 会话由请求里的 `thread_id` 标识；前端首次请求拿到 `session` 事件后复用，点击「New chat」即丢弃
- `ShoppingAgent` 为每个 `thread_id` 保存一个 `Session`：对话记录（不含 system prompt）+ 记忆笔记 + 一把锁（同一会话的并发请求串行执行）
- 多轮上下文（如「cheaper ones」）由 agent 自己读取历史解析，不再有单独的「计划」步骤
- **上下文裁剪**由 `ContextPolicy` 负责：旧的工具结果替换成占位符（保留最近 3 条），整步裁剪只在超预算时发生，且永远保留最新一步。裁剪只作用于**发给模型的副本**，落盘的记录始终完整
- **记忆笔记**是 `Memory`：模型可通过 `save_note` 写下约束（「预算 500」「必须黑色」），它以 pinned system block 的形式插在 system prompt 之后——剪枝剪不到它
- 语言与 shopper id 每轮拼进 system prompt，不写进对话记录，所以同一份配置能同时服务所有用户与语言
