# Training pipeline and single-GPU feasibility

## The pipeline

```
            raw release                       a stronger model
   data/raw/fine_items_*.json.gz          (DeepSeek / Qwen / GPT via API)
                 │                                    │
        shoprl catalogue + tasks                     │
                 │                                    │
         ┌───────┴────────┐                           │
         │ task pools     │                           │
         │ official_test  │  ← evaluation, untouched  │
         │ dev / sft / rl │  ← disjoint, seeded       │
         └───────┬────────┘                           │
                 ▼                                    ▼
            teacher rollout  ─────────────────►  collect  (EpisodeRunner + shoprl.harness)
                 │                                    │
                 │                          accepted trajectories only
                 │                                    ▼
                 │                          prepare-sft  → turn-level examples
                 │                                    ▼
                 │                                  SFT
                 │                                    ▼
                 └──────────► on-policy RL ──► GRPO + RLVR (+ OPD / OPSD / RLSD)
                                        │
                                        ▼
                                  eval on official_test
```

Everything from `catalogue` to `prepare-sft` runs on CPU. Only SFT and RL need a
GPU.

### Stage 0 — data (`00_build_data.sh`)

Builds `data/generated/shop_products.jsonl.gz` (all 23,421 products, options
included, source order preserved) and four task pools:

| Pool | Size | Split | Purpose |
|---|---|---|---|
| `official_test` | 1,459 | eval | the published evaluation split, never trained on |
| `dev` | 150 | train | smoke tests and prompt iteration |
| `sft` | 512 | train | teacher collection |
| `rl` | 500 | train | on-policy rollouts |

Sampling is stratified round-robin over leaf categories, easiest first, with a
fixed seed, so pools are reproducible and cover the catalogue instead of
clustering in the biggest category. Disjointness is asserted by the test suite.

### Stage 1 — teacher collection (`01_collect_teacher.sh`)

Any OpenAI-compatible endpoint drives the real harness against the real
environment. A trajectory is kept only if the episode reports `done`; everything
else is written with its termination reason. In the reference run 412 of 512
tasks were accepted (80.5 %) — treat anything near or below that as the expected
pass rate, not as a bug.

Trajectories are written one JSON file per task/sample under `raw/`, so a rerun
after a crash skips completed work.

### Stage 2 — SFT data (`02_prepare_sft.sh`)

One trajectory becomes one example per assistant turn, where the prompt is the
context recorded at that turn. Two checks make the data trustworthy:

* the prompt is re-rendered with the student's chat template and must be a
  **prefix** of the full conversation; if not, the example is dropped, because
  the token boundaries do not line up with the template;
* examples longer than the cap are dropped, never truncated, so a turn is either
  fully trained or not trained.

Watch `token_p95` in `summary.json`. If it sits at the cap, raise the cap or drop
the long tail deliberately — the reference run's own numbers were 6,153 examples
from 412 trajectories.

### Stage 3 — SFT (`03_train_sft.sh`)

`transformers.Trainer` over precomputed `input_ids` + `labels`. The loss mask is
computed once in stage 2 rather than by a collator, because in this data most
tokens in a sequence are tool output the model must read, not write.

### Stage 4 — RL (`04_train_grpo.sh`)

GRPO with the environment's own reward. Per step:

1. sample `tasks_per_step` tasks from the `rl` pool;
2. roll each task out `group_size` times with the current policy;
3. score every candidate with the ShopSimulator verifier;
4. centre rewards **inside the group** — the advantage is group-relative, so a
   task everyone solves and a task nobody solves both carry zero gradient;
5. turn each candidate into turn-level samples and backprop
   `-min(ρA, clip(ρ)A) + λ·KL(student‖reference)`.

Three numbers in `train_log.jsonl` decide whether a run is worth continuing:

| Field | Reading |
|---|---|
| `zero_variance_groups` | if it stays near `tasks_per_step` for many steps, the pool is saturated or hopeless for this policy — change the pool, not the learning rate |
| `done_rate` | usually moves before `r_loose` does; a policy that never finishes an episode cannot be improved by any advantage |
| `turn_samples` | near zero means rollouts are dying before the model commits to anything, usually a parsing or context problem |

### Method variants

The RLVR path is the default. The distillation variants differ only in what is
laid on top of the same rollouts:

| Variant | What the teacher is | Configure |
|---|---|---|
| RLVR | none | `configs/grpo.json` |
| **OPD** on-policy distillation | a stronger checkpoint scored on the student's own tokens | `configs/grpo_rlvr_opd.json` + `--teacher-path` |
| **OPSD** self-distillation | the same weights, given privileged context the student did not have | `--self-teacher` |
| **RLSD** | RL and a distillation term on the same rollouts, summed in the loss | `--distill-beta` with either of the above |

One structural constraint decides where this code lives: the distillation target
must be computed from the **same** rollouts, before the policy update, and must
not be recomputed from updated weights mid-step. So the variant dispatch is in
the loss (`GrpoTrainer._policy_loss`), not in the rollout loop, exactly as the
reference project argues.

Whether any of these combinations help is an empirical question. Implementing
them is not evidence that they work, and `configs/grpo_rlvr_opd.json` is a
starting point, not a result.

## Feasibility on one card

Measured against the actual memory budget, not a rule of thumb. A bf16
full fine-tune needs 2 bytes/param for weights, 2 for gradients, and 8 for AdamW
in fp32 — 12 bytes/param before activations; 8-bit Adam cuts the optimiser term
to about 2, giving ~6 bytes/param.

| Run | 1.7B | 4B | 8B |
|---|---|---|---|
| SFT, LoRA r=16 | ✅ comfortable | ✅ | ✅ QLoRA |
| SFT, full FT + 8-bit Adam | ✅ 24 GB is enough | ⚠️ 48 GB, or 24 GB with `max_seq_len` ≈ 4k | ❌ |
| GRPO, LoRA + shared base as reference | ✅ | ⚠️ tight at 24 GB | ❌ |
| GRPO, full FT + vLLM colocated | ⚠️ only with a small KV budget | ❌ | ❌ |

Concretely on a 24 GB card:

* **1.7B SFT, full fine-tune** — weights 3.4 GB, gradients 3.4 GB, 8-bit Adam
  ~3.4 GB, ≈ 10 GB plus activations with gradient checkpointing. `batch_size 1`,
  `grad_accum 8`, `max_seq_len 12288` fits.
* **1.7B GRPO, LoRA** — the frozen base serves as its own reference via
  `disable_adapter()`, so no second copy is loaded and the KL term costs one
  extra forward pass instead of another checkpoint.
* **vLLM colocated** — `gpu_memory_utilization 0.3–0.4` for a 1.7B rollout engine
  leaves the rest for training. This is the setting that decides whether RL is
  minutes or hours per step; `HFEngine` is the fallback that always works.

On an RTX 6000 (48–96 GB) the same configuration runs with `max_seq_len 16384`,
`batch_size 2` and a larger KV budget; 4B full fine-tuning becomes practical.

What is **not** feasible on one card, and should not be attempted: the reference's
Megatron + SGLang + Ray stack, or any configuration that needs tensor parallel
across devices.

### Time budget, honestly

The wall clock is dominated by rollouts, not by the optimiser. A shopping episode
is 10–30 model turns, each accumulating the whole page history, so one rollout is
roughly 20 forward passes and 20 samples of a few hundred tokens. With a 1.7B
policy on a 24 GB card:

| Configuration | Rough cost per RL step (20 rollouts) |
|---|---|
| `transformers` generation, no vLLM | tens of minutes |
| vLLM colocated | minutes |

This is why `--concurrency` and the pool capacity matter more than the learning
rate, and why the first thing to do on new hardware is `PRESET=smoke
03_train_sft.sh` followed by a 5-step RL run with `group_size 2`.

## Reproducing a result

Every stage writes a `run_manifest.json` with the resolved config, seeds, torch
and CUDA versions and the GPU name. A number without that file is not a result.

```bash
export PYTHON=python

bash training/scripts/00_build_data.sh

export TEACHER_MODEL=deepseek-chat
export TEACHER_BASE_URL=https://api.deepseek.com/v1
export TEACHER_API_KEY=...
bash training/scripts/01_collect_teacher.sh

MODEL=Qwen/Qwen3-1.7B bash training/scripts/02_prepare_sft.sh
PRESET=4090 bash training/scripts/03_train_sft.sh

CONFIG=configs/grpo.json STEPS=100 bash training/scripts/04_train_grpo.sh

MODEL=runs/qwen3-1.7b-grpo/step-100 bash training/scripts/05_eval.sh
```

Compare against a base-model row measured the same way:

```bash
MODEL=Qwen/Qwen3-1.7B LABEL=base bash training/scripts/05_eval.sh
```

The oracle ceiling on this environment is 95.2 % `r_hard` and 100 % `done`
(`docs/data-audit.md` §4). Reference numbers to place a run against: base 2.0 %
positive reward, SFT 72.5 %, RL 90.5 %; strict success 0 % → 10.5 % → 31.0 %.
A run that beats 90.5 % positive reward is close to the ceiling, not to the
middle of the field.

