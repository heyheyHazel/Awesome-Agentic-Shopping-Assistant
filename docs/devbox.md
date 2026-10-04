# The training box

Everything here describes the AutoDL container this project now lives on. Half of
it is not obvious from the repository, and the container is not immortal: if it
is rebuilt, these are the steps that make it work again.

## What the machine is

| | |
|---|---|
| Host | `ssh -p 48325 root@connect.westb.seetacloud.com` (alias `devbox`) |
| Project | `/root/autodl-tmp/Agentic-Shopping-Assistant` |
| OS | Ubuntu 22.04.5, kernel 5.15 |
| CPU / RAM | 1 vCPU visible, ~1 TB RAM |
| Container memory cap | **2 GB**, no swap, until a GPU is allocated |
| Disk | 50 GB on `/root/autodl-tmp` (data disk), 30 GB on `/` |
| GPU | one RTX 6000, **not yet allocated** |
| Python | 3.12.3 from `/root/miniconda3` |
| torch | 2.12.1+cu130, preinstalled by the image |

The project deliberately lives on `/root/autodl-tmp` rather than `/root`: the root
filesystem is 30 GB and the checkpoints alone are about 14 GB.

## Network

huggingface.co is unreachable from this container. Two mirrors are not:

| Endpoint | Measured | Used for |
|---|---|---|
| `hf-mirror.com` | ~1 MB/s | `HF_ENDPOINT`, the fallback for weights |
| `modelscope.cn` | ~6 MB/s | the default source for weights |
| `pypi.org` | works | packages |

`scripts/download_models.py` tries ModelScope first and falls back to the Hugging
Face mirror. `scripts/fetch_data.py` already points at `hf-mirror.com` and
`aifasthub.com`.

## The Python environment

`scripts/setup_devbox.sh` builds it. Two decisions are worth keeping.

**The venv inherits the image's torch** via `--system-site-packages`. torch is
matched to the image's CUDA runtime; reinstalling it costs several GB and can swap
the CUDA build underneath a working driver.

**Everything after the venv is installed with pip, not uv.** uv's resolver does
not treat inherited site-packages as satisfying a requirement, so it re-fetches
its own ~2.5 GB CUDA build of the same torch version. pip inside the venv sees the
inherited torch and leaves it alone.

    bash scripts/setup_devbox.sh          # idempotent
    .venv/bin/python -m pytest -q         # 100+ tests, no GPU needed

## Data and weights

The ShopSimulator release was copied from the workstation rather than re-fetched,
which is why the box needs no dataset download. `data/raw/.verified.json` records
all three raw files, including the persona side-car that had to be re-fetched to
its full 3,323 records.

    python -m shoprl.cli catalogue        # 23,421 products, ~3 s
    python -m shoprl.cli tasks            # five pools, seeded and disjoint
    python scripts/download_models.py     # Qwen3-0.6B / 1.7B / 4B

## What broke, and is fixed

The box runs **transformers 5.17** and the training code was written against 4.x.
Both failures were hard errors on the first line of a run.

| Symptom | Cause | Fix |
|---|---|---|
| `from_pretrained` rejected the weight dtype | v5 renamed `torch_dtype` to `dtype` | `shoprl.train.common.precision_kwarg` picks by major version |
| `TrainingArguments` rejected `warmup_ratio` | v5 removed it | the fraction is converted to `warmup_steps` in `build_training_arguments` |
| every SFT example came out with an empty target | v5 returns a **dict** from `apply_chat_template` where 4.x returned a list, and `list(dict)` yields its keys | `shoprl.data.sft._render` normalises list, dict and tensor shapes |
| the first training step died with `unsupported operand type(s) for +: 'dict' and 'list'` | a comprehension variable shadowed the collator's parameter | rename to `rows` in `train.sft.build_collator` |

A fifth only appears on a CPU-only box: `bf16=True` is a hard error without a GPU,
so the precision flags are gated on `torch.cuda.is_available()` and `use_cpu` is
set otherwise.

The first three were only reachable with a real tokenizer and a real
`TrainingArguments`; the last two would have failed on the first step of the
first run. `tests/test_training_smoke.py`, `tests/test_pipeline_smoke.py` and one
new case in `tests/test_shoprl_data.py` cover them, and all of them pass here.

## Memory

`/sys/fs/cgroup/memory.max` is 2 GB with no swap until a GPU is allocated, which
is far below what a forward pass through a 151k-vocabulary head needs.

What actually consumes it is worth recording, because it is not the tests:

| Holder | Resident |
|---|---|
| VS Code Remote server processes | ~1.2 GB |
| jupyter-lab, tensorboard, autopanel (image services) | ~0.4 GB |
| left for whatever is being run | **~0.3–0.5 GB** |

The ceiling therefore moves with the editor session. With it closed the full
suite runs in one process in about 40 seconds; with it open, importing a
tokenizer is enough to be killed. Two consequences are built into the repository:

- `scripts/check_ready.sh` runs one test file per process and reports a file the
  kernel killed as INFO, so the gate distinguishes "the tests failed" from "this
  box could not run them";
- checkpoint-backed tests are opt-in through `SHOPRL_TEST_CHECKPOINT`, so the
  default suite never reaches for a 151k-vocabulary tokenizer.

Nothing else in the suite needs more than a few hundred MB.

## What is deliberately absent

**vLLM.** The RL rollout engine falls back to `transformers` generation, which
works everywhere. vLLM decides whether an RL step takes minutes or tens of
minutes, so install it once a GPU is allocated:

    .venv/bin/python -m pip install vllm

## Readiness

    bash scripts/check_ready.sh

Prints one PASS/FAIL line per requirement and exits non-zero if anything is
missing. A missing GPU is reported as INFO rather than a failure, because the card
may simply not be allocated yet. Everything else — venv, torch, catalogue, task
pools, checkpoints, raw data, the test suite and free disk — is checked.

## State as of the initial provisioning

    PASS  venv at .venv/bin/python
    INFO  torch 2.12.1+cu130, cuda build 13.0
    INFO  CUDA not visible yet (no GPU allocated)
    PASS  catalogue built: 23,421 products
    PASS  task pools built and sized as documented
    PASS  checkpoints: Qwen3-0.6B 1.52 GB, Qwen3-1.7B 4.08 GB, Qwen3-4B 8.06 GB
    PASS  raw files match .verified.json; persona side-car complete
    PASS  test suite green: 117 passed, 1 skipped, 1 deselected
    INFO  37 GB free
    READY: everything except a GPU allocation is in place.

The skip is the gradient test in `test_training_smoke.py`, which needs more than
the 2 GB the container is capped at before a card is allocated. The deselected
test is the slow one below.

### Both training entry points have been executed

**The interactive app.** `bash scripts/check_server.sh` starts the canned model
and the real server against the real catalogue, then checks the HTTP surface and
a whole SSE conversation over a socket:

    PASS  server up on :8077
    PASS  health
    PASS  meta: {"data_source":"shopsimulator","currency":"CNY","currency_symbol":"¥"}
    PASS  frontend shell served
    PASS  profile for U02358E8
    PASS  sse event: session / experiment / tool / products / token / done
    PASS  product cards carried real catalogue ids
    PASS  no error events
    SERVER OK: HTTP surface and SSE conversation both work.

That is the front-to-back claim in the objective, checked rather than asserted:
the real FastAPI app, the real httpx client, the real tool registry, the real
23,421-product catalogue and the real SSE framing, with only the model canned.
`scripts/stub_llm.py` is the canned model; it is also the way to develop the UI
without spending API credit. CI runs the same script.

**SFT.** `tests/test_sft_run.py` calls `train()` itself — the function the GPU run
calls — on a two-layer model built from the real checkpoint's configuration and
tokenizer, writes two synthetic examples through the real data pipeline, runs one
optimiser step, and checks that a checkpoint and its manifest land on disk. It
passes, which means everything under the model-loading step works: the optimiser
arguments, the collator, the trainer loop, the save path and the manifest.

**GRPO.** `tests/test_grpo_run.py` drives `GrpoTrainer.train()` with the real
catalogue, a real env pool and a scripted engine that solves every other
candidate. It loads a task from the seeded pool, rolls out a group, centres the
rewards, detects the group's variance, updates, logs and checkpoints. Two
candidates with different outcomes are deliberate: an all-equal group produces
zero advantages and would leave the interesting half of the loop unexecuted.

The sampled turns are around 2,700 tokens against a 151k vocabulary, so a single
backward pass needs several GB — more than this container has before a card is
allocated. `gather_token_logprobs` therefore computes
`logit - logsumexp(logits)` instead of calling `log_softmax`, which avoids a
second `[batch, tokens, vocab]` tensor. The gradient step itself is a separate
test that skips below 6 GB and runs in CI and on a host with a card.

The slow tests total about five minutes on one CPU core, so they are marked
`slow` and deselected by default:

`tests/test_sft_run.py` calls `train()` itself — the function the GPU run calls —
on a two-layer model built from the real checkpoint's configuration and
tokenizer, writes two synthetic examples through the real data pipeline, runs one
optimiser step, and checks that a checkpoint and its manifest land on disk. It
passes here, which means everything under the model-loading step is known to
work: the optimiser arguments, the collator, the trainer loop, the save path and
the manifest.

It takes about five minutes on one CPU core, so it is marked `slow` and
deselected by default:

    .venv/bin/python -m pytest -q -m slow      # the full training step
    .venv/bin/python -m pytest -q              # everything else, ~50 s

CI runs both, in the torch job.

What is left is not preparation:

1. allocate the RTX 6000 and re-run `scripts/check_ready.sh` to confirm CUDA is
   visible and the skipped test now runs;
2. collect teacher trajectories (`training/scripts/01_collect_teacher.sh`), which
   needs an API key and no GPU at all — it could be done before the card lands;
3. SFT, then GRPO, then evaluation against the three baselines.
