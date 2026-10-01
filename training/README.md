# Training subsystem

Post-training for a small open model (0.6B–4B) that drives the shopping agent,
against the ShopSimulator environment:

```
teacher collection  ->  SFT  ->  on-policy GRPO (RLVR)  ->  evaluation
                                     + optional OPD / OPSD / RLSD
```

The agent being trained is the same loop this repository serves. The harness, the
environment and the reward live in `src/shoprl/`; this directory holds the run
scripts and the configs.

## Start here

| Document | Contents |
|---|---|
| [`docs/data-audit.md`](../docs/data-audit.md) | what the ShopSimulator release actually contains, and its two defects |
| [`docs/harness.md`](../docs/harness.md) | the harness design, what Pi does in the reference project, and what was taken from it |
| [`docs/training.md`](../docs/training.md) | the pipeline, the method variants, and single-GPU feasibility with memory arithmetic |

## Quick start

```bash
export PYTHON=python            # a venv with `pip install -e ".[train]"` on the GPU host

bash training/scripts/00_build_data.sh          # CPU, ~1 minute, no network

export TEACHER_MODEL=deepseek-chat
export TEACHER_BASE_URL=https://api.deepseek.com/v1
export TEACHER_API_KEY=...
bash training/scripts/01_collect_teacher.sh     # 512 tasks -> ~400 trajectories

MODEL=Qwen/Qwen3-1.7B bash training/scripts/02_prepare_sft.sh
PRESET=4090 bash training/scripts/03_train_sft.sh
CONFIG=configs/grpo.json STEPS=100 bash training/scripts/04_train_grpo.sh
MODEL=runs/qwen3-1.7b-grpo/step-100 bash training/scripts/05_eval.sh
```

## Layout

```
training/
├── README.md            # this file
└── scripts/             # one script per stage, each with hardware presets

src/shoprl/
├── harness/             # the loop: backends, tools, context policy, memory, rollout
├── env/                 # ShopSimulator: catalogue, search, session, reward, task pools
├── data/                # trajectory collection and turn-level SFT conversion
├── train/               # rollout engines, SFT trainer, GRPO trainer, distillation
└── eval/                # rollout evaluation and the official metrics

configs/                 # JSON config, overridable field by field on the CLI
data/tasks/              # generated pools: official_test, dev, sft, rl
runs/                    # generated artefacts, gitignored, one manifest per run
```

## Status

CPU-side stages are implemented and tested: catalogue and task pools, the
environment (reward, paging, option selection, session isolation, persona
blocks), the harness (loop, context policy, memory, tool registry), trajectory
collection, turn-level SFT conversion with template verification, and the
official metrics.

GPU-side trainers are implemented but **have not been executed here**: this
repository was developed on an Apple M3 with no CUDA device, so `train_sft` and
`train_grpo` have been reviewed and their components unit-tested, not run to
convergence. The first thing to do on a GPU host is the `smoke` preset, then a
5-step RL run with `group_size 2`. No numbers in this repository come from a
training run.
