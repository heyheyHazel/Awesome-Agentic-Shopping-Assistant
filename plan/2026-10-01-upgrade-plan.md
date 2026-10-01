# Plan: turn the shopping assistant into a trainable agent project

## Goal

One repository that shows the whole loop: an interactive product, a harness that
is also the training harness, a verifiable environment, and a model pipeline that
fits on a single card.

## Where things stood

The repository was a working storefront demo over a single tool-calling agent.
The training directory held a design document and no code, for a stated reason:
the reference pipeline needs CUDA, and the development machine has none.

Two problems followed from that:

1. the data pipeline kept only the product half of ShopSimulator, so there was
   nothing to train on;
2. the agent loop was LangGraph's `create_agent`, which records no trajectories,
   exposes no token-level output and cannot be rewarded.

## Decisions

| Decision | Reason |
|---|---|
| Keep the harness framework-free, in `src/shoprl/` | the loop must record exact prompts, sampled tokens and rewards; a graph runtime hides all three |
| Reimplement the environment instead of vendoring upstream's Flask + Lucene service | the Lucene index is not distributed and needs Java 21; the action grammar and reward are portable |
| Keep an HTTP env client anyway | a published number may still need the reference environment |
| Custom GRPO rather than a framework trainer | multi-turn rollouts, group-aware credit assignment and a distillation term on the same rollouts are easier to keep honest in 200 lines than to configure around |
| SFT labels precomputed, not produced by a collator | most tokens in a sequence are tool output; masking by role is a silent way to train the model to write the pages it reads |
| Task pools stratified, disjoint and seeded | a saturated or hopeless pool tells you nothing; the mix is part of the experiment |

## Delivered in this pass

1. **Audit** — `docs/data-audit.md`: 23,421 tasks, source order is the id space, a
   persona side-car that arrived truncated and was re-fetched, 106 products lost
   to a price filter, and a measured 95.2 % oracle ceiling.
2. **Environment** — `src/shoprl/env/`: catalogue with options, in-process BM25,
   the page-flow session, the upstream reward with four sub-scores, an env pool
   with per-rollout leases, and a client for the reference service.
3. **Harness** — `src/shoprl/harness/`: loop, context policy, memory, tool
   registry, backends, rollout runner. Every step records the exact prompt it saw.
4. **Data** — `src/shoprl/data/`: trajectory collection with explicit rejection
   reasons; turn-level SFT conversion with chat-template verification.
5. **Training** — `src/shoprl/train/`: rollout engines, SFT trainer, GRPO with
   group-relative advantages, and distillation terms for OPD / OPSD / RLSD.
6. **Evaluation** — `src/shoprl/eval/`: rollout evaluation and the official metric
   set, sub-scores reported separately.
7. **Entry points** — `python -m shoprl.cli` plus six numbered shell scripts with
   hardware presets.
8. **Tests** — 22 new tests over environment semantics, harness invariants, loss
   masks and pool construction; 55 total, all CPU-only.
9. **Serving on the same loop** — the FastAPI agent now runs `shoprl.harness`
   with the four storefront tools declared as `ToolSpec`s. The SSE protocol, the
   tool panel and the typewriter reply are unchanged, the 33 original tests still
   pass, and `langchain` / `langgraph` / `langchain-openai` are gone from the
   dependency list and from the import graph.
10. **Personas wired through** — all 4,666 persona-carrying records keep their
    profile through the catalogue, the environment returns it on reset, and
    `shoprl.env.persona` renders the decisions-relevant fields into a short block
    (`--persona`; 1,343 of them form the `official_test_persona` pool).

## Not done, and why

* No training run has been executed. There is no CUDA device here. The trainers
  are implemented, unit-tested at the component level, and unproven end to end.
* The demo frontend has not been rebuilt; the compiled `frontend/dist` bundle
  still shows the previous agent's layout, which is unchanged by this work.

## Next steps, in order

1. Run the smoke preset and a 5-step RL run on the GPU host; record the first
   `run_manifest.json` and the wall-clock per step.
2. Collect 512 teacher trajectories and compare the acceptance rate with the
   reference's 80.5 %.
3. Establish three rows on `official_test`: base, SFT, RL. Nothing is comparable
   until all three exist.
4. Only then vary the method: OPD, OPSD, reward shaping, context policy.
