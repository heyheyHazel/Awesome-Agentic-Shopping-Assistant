# Harness design, and where Pi fits

## What Pi is in the reference project

`Piucente/pi-slime-shopsimulator` is three things stacked:

| Layer | What it is |
|---|---|
| **Pi** | `@earendil-works/pi-coding-agent`, a Node.js *coding agent* CLI, not a library |
| **Slime** | a vendored fork of `THUDM/slime` (Megatron-LM training + SGLang rollout + Ray) |
| **ShopSimulator patch** | a 391 KB diff that makes upstream's Flask shop thread-safe and poolable |

How they connect, concretely, from the repo's own sources:

1. `pi_harness.py` launches `pi` as a subprocess in `--mode json`, with
   `PI_CODING_AGENT_DIR` pointing at a throwaway config so no user setting leaks
   in, `--no-builtin-tools --no-skills --no-extensions` and a single `--extension
   shop_extension.ts`.
2. `shop_extension.ts` registers exactly two tools, `shop_reset` and `shop_act`,
   which POST to the Flask environment. It also installs a `context` hook that
   rewrites the *copy* of the transcript sent to the model: old `shop_act`
   results are replaced with `[旧的 shop_act 工具结果已裁剪；done=false]`, keeping
   the last `SHOP_CONTEXT_KEEP_ACT_RESULTS=3`, and thinking blocks are stripped.
   Every rewritten prompt is appended to a context trace file.
3. Pi's model provider is Slime's SGLang adapter, so the "coding agent" is
   actually being driven by the policy under training.
4. Slime's `generate.py` collects the JSONL event stream, reads the reward out of
   `tool_execution_end` details, and turns one rollout into `Sample` objects with
   a loss mask.

That is a large machine. The reference README's own sizing note says the
published run used a single 84 GB Pro 6000D with `MAX_TOKENS_PER_GPU=12288`, and
that smaller-memory configurations "尚未验证".

## Why this repository does not vendor it

Not because the design is wrong — most of it is right, and the next section
adopts it. Because of what it costs on one card:

* **A conda/micromamba environment that compiles CUDA extensions** for SGLang and
  Megatron-LM, pinned to specific SGLang/Megatron revisions by `build_conda.sh`.
  This is a multi-hour build on a good connection and it is the *prerequisite*,
  not the experiment.
* **A Node process per rollout.** Fine at 20-way concurrency on a datacentre
  card; on a 24 GB card the GPU is the bottleneck, so the process is pure
  overhead — and it hides the prompt, which you then have to reconstruct by
  parsing an event stream to build training data.
* **An environment that needs Java 21 and a Lucene index that upstream does not
  ship** (see `docs/data-audit.md` §5).
* **Megatron for a 1.7B model on one GPU.** Megatron's value is tensor/pipeline
  parallelism across many devices. On one device you are paying its
  configuration cost for FSDP-equivalent behaviour that `transformers` + PEFT
  already give you.

## What was taken from Pi, item by item

The design ideas are the durable part, so they were kept and re-expressed in
Python where the loop is inspectable:

| Pi / Slime mechanism | Where it lives here |
|---|---|
| Two tools, `shop_reset` + `shop_act`, native action DSL | `shoprl/env/tools.py` |
| Deterministic request-time context pruning of old tool results | `shoprl/harness/context.py`, `ContextPolicy` |
| Context trace per model turn | every `Step` stores the exact `context` it was called with |
| `rollout_session_id` and a pooled environment | `shoprl/env/local.py`, `EnvPool` + lease per episode |
| `over` / `done` / turn budget | `Trajectory.termination`, in `shoprl/harness/types.py` |
| JSONL event stream for observability | `AgentLoop(emit=...)`, forwarded to SSE when serving |
| Loss-mask correctness against the chat template | `shoprl/data/sft.py` verifies prefix equality and drops the example when it fails |
| Group validation before normalising rewards | `GrpoTrainer._advantages`, reports `zero_variance_groups` |
| Deterministic prices and price ceilings | `shoprl/env/reward.py`, seeded from the ASIN |

Two of those deserve the emphasis the reference gave them, because both were
bugs upstream had to patch and both re-appear if the code is rewritten naively:

**Session isolation.** One environment per in-flight episode, never a shared
one. `EnvPool.acquire` hands out a slot, `ShopToolkit.close` always returns it,
and `interact` refuses a session id it did not issue. Concurrency over shared
state leaks one rollout into another and corrupts the reward silently.

**Deterministic reward inputs.** Prices and price ceilings are functions of the
ASIN and the instruction. If they come from process-level `random`, the same task
scores differently on two rollouts of the same group, the group advantage becomes
noise, and RL optimises the noise.

## What is different here

### The loop is the artefact, not a subprocess

`shoprl/harness/loop.py` is a hundred lines of Python that call a model, run the
tools it asked for, and record what happened. The consequences that matter for
training:

* every step carries the **exact prompt** the model saw, so an SFT turn is
  byte-identical to the turn that was rolled out;
* sampled token ids and log-probabilities ride along on the assistant message, so
  RL never re-tokenises text to recover what the policy produced;
* a terminal observation is a marker string, so the loop stops without reaching
  into environment internals.

### One harness, two consumers

The same `AgentLoop` serves the web app and generates training data. Serving
passes an `emit` callback that pushes SSE events; training passes nothing and
reads `Trajectory` objects. That is the whole reason the demo and the training
run cannot drift apart.

```
                      ┌─ emit=SSE writer ──► FastAPI /api/v1/chat
AgentLoop ── backends ─┤
    ▲   ▲             └─ emit=None ───────► Trajectory ──► SFT data / GRPO reward
    │   │
    │   └── tools: shop_reset, shop_act  (or the web app's search/profile/inventory tools)
    └────── context policy + memory
```

### Memory is a first-class block, not a summary

Pi handles long episodes with compaction. That is expensive, lossy and hard to
train on. Here, `Memory` is a small bounded set of notes the model writes itself
through an optional `save_note` tool, rendered into one pinned system block.
`ContextPolicy._head_end` pins that block along with the system prompt and the
task statement, so pruning the transcript can never prune the constraints the
shopper stated in turn one.

It is off by default in the RL config, because adding a tool changes the tool set
the policy sees and therefore the comparability of a run. Turn it on with
`EpisodeRunner(use_memory=True)` or by passing a `Memory` to `ShopToolkit`.

## Environment fidelity

`shoprl/env/` reproduces the upstream page flow, action grammar and reward
formula. The deliberate divergences, all of them recorded in the reward module's
docstring:

| Upstream | Here | Why |
|---|---|---|
| Lucene/`pyserini` BM25, needs Java 21, index not distributed | in-process BM25 over CJK bigrams, ~1 s to build | runs anywhere; the index cannot be obtained otherwise |
| spaCy `zh_core_web_sm` noun overlap for `r_type` | CJK bigram overlap | avoids a 50 MB model download for a term that cannot move on this data anyway |
| `thefuzz` at threshold 85 | `rapidfuzz` at threshold 85, `difflib` fallback | same threshold, faster, no hard dependency |
| `random.uniform` price, `random.sample` ceiling | seeded from ASIN and instruction | the reference project patched the same thing |
| option values with `/` rewritten to a pipe only in the clickable list | kept as-is on purpose: it is a real property of the release that caps `r_hard` at ~95 % |

Set `SHOPRL_ENV_URL` to a running upstream Flask service and
`shoprl.env.http.RemoteShopEnv` is used instead, with identical method names, so
a published comparison can still be run.
