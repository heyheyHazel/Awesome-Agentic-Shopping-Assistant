# Results

No number here comes from a training run, because none has happened. The
pipeline runs end to end on CPU and both trainers have been executed as tests
(see `training/README.md`), but no checkpoint has loaded real weights and no
gradient step has run over a real episode.

This file exists so that the first real numbers land somewhere structured, and
so the one number that *is* measured has a home.

## Measured

| Policy | Split | Tasks | done | `r_loose` | `r_hard` | `r_success` | How |
|---|---:|---:|---:|---:|---:|---:|---|
| scripted reference | `official_test` | 20 | 0.80 | 0.757 | 0.750 | 0.750 | `DRY_RUN=1 LIMIT=20 bash training/scripts/05_eval.sh` |

The scripted policy plays the page it was shown and searches once with the raw
instruction. It has no language ability, so **0.750 `r_hard` is the floor a
trained model has to clear**, not a target.

It is not the ceiling either: an oracle that can re-query reaches 95.2 % on this
environment, and the 12.5 % gap between the two is search reachability, not
scoring (`docs/data-audit.md` §4).

## Scale reference, not directly comparable

The reference project — Pi + Slime + ShopSimulator, a Qwen3.5-2B student on
84 GB — published these from a 200-task sample of its own build:

| Model | positive reward | strict success |
|---|---:|---:|
| Base | 2.0 % | 0.0 % |
| SFT | 72.5 % | 10.5 % |
| RL | 90.5 % | 31.0 % |

Their environment differs from this one in ways that change the number: a Lucene
index instead of in-process BM25, spaCy noun overlap instead of CJK bigrams, and
a different catalogue build. Treat the shape — base near zero, SFT large, RL
incremental — as the signal, not the digits.

## To be filled in

Run `05_eval.sh` once per row and copy the fields from `metrics.json`:

| Policy | Checkpoint | Split | Tasks | done | `r_loose` | `r_hard` | `r_success` | Wall clock | Manifest |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| base | `models/Qwen3-1.7B` | `official_test` | 200 | | | | | | |
| SFT | `runs/qwen3-1.7b-sft` | `official_test` | 200 | | | | | | |
| RL | `runs/qwen3-1.7b-grpo/step-100` | `official_test` | 200 | | | | | | |

A row is only comparable to another row if:

- it uses the same split and the same number of tasks — `official_test` is the
  published evaluation split and is never trained on;
- the policy is sampled at temperature 0, one rollout per task;
- the line comes from `metrics.json`, not from a hand-computed average;
- the checkpoint's `run_manifest.json` is recorded next to it, because a number
  without its seed, library versions and GPU is not a result.

Policies on the `dev` pool are for smoke tests. Reporting a `dev` number as if it
were an evaluation number is the easiest way to fool yourself, since `dev` tasks
come from the training split.

## What is still unmeasured

| | Why |
|---|---|
| any training run | no GPU |
| a gradient step over a real episode turn | a sampled turn is ~2,700 tokens against a 151k vocabulary; the backward pass needs several GB |
| teacher trajectories from a real model | needs an API key, and spending it is the user's call |
| SFT and RL wall-clock per step | follows from the two above |
| whether OPD / OPSD / RLSD help | implemented in `shoprl/train/distill.py`, unrun; implementing a method is not evidence it works |
