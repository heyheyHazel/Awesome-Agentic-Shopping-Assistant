# ShopSimulator data audit

Everything below was measured on the files in `data/`, not read off the upstream
README. Commands that reproduce each number are given inline.

## Summary

The data needed to train an agent is present. The data needed to *serve* the demo
was the only part the old pipeline kept, so the training half was effectively
missing: 23,421 tasks existed on disk and none of them reached the application.

Two real defects exist and both are in the persona side-car files, not in the
main release.

| Item | State |
|---|---|
| `fine_items_eval_train_all.json.gz` (23,421 records) | complete, size-verified |
| Product catalogue for the environment | rebuilt from the above, all 23,421 rows |
| Instructions / goals | complete: exactly one per record |
| SKU options and their prices | complete: 239,925 option values |
| Personas | 4,009 unique users, 4,666 records; rendered into a ~266-character prompt block (`shoprl.env.persona`) |
| `fine_items_eval_persona.jsonl` | complete: 1,343 records |
| `fine_items_train_persona.jsonl` | **truncated**: 1,603 usable of 3,323 |
| Upstream Lucene search index | **absent**: not distributed, needs Java 21 |
| Model weights (2B / 4B) | out of scope for this audit |

## 1. The task space

```bash
python - <<'PY'
import gzip, json
rows = json.load(gzip.open("data/raw/fine_items_eval_train_all.json.gz", "rt", encoding="utf-8"))
print(len(rows), rows[1458]["tag"], rows[1459]["tag"])
PY
```

23,421 records, one instruction each, so 23,421 tasks. **Row order is the task id
space** and it is split at a fixed boundary:

| Task ids | Split | Count |
|---|---|---|
| `0 .. 1458` | `eval` | 1,459 |
| `1459 .. 23420` | `train` | 21,962 |

This matters because the upstream environment selects a task by integer index
(`env.reset(idx=i)`) and its own runner hard-codes `range(1459)` for evaluation.
Any repacking that reorders records silently remaps every task, which is why
`shoprl.env.catalog` writes its cache in source order and stores `index` on every
product.

Persona coverage inside the same file:

| Range | Records with a persona |
|---|---|
| `0 .. 1458` (eval) | 1,343 |
| `1459 .. 23420` (train) | 3,323 |
| whole file | 4,666 records, 4,009 distinct `用户ID` |

## 2. What the old converter threw away

`scripts/fetch_data.py` maps each record to a web-app `Product`. Fields it does
not carry through:

| Dropped field | Why it matters |
|---|---|
| `instructions[0].instruction` | the task itself — without it there is nothing to train on |
| `instructions[0].options` | the requested SKU values, and `r_option` is scored against them |
| `customization_options` | the clickable option values, and the price of each |
| `pricing[1]` | the ceiling of a price range; 15,974 records have a range |
| `attribute` beyond the first five | `r_att` is averaged over *all* requested attributes |
| `user_persona` | personalisation tasks |

So `data/generated/catalog.json` is a legitimate catalogue for a storefront demo
and an unusable one for training. `shoprl.env.catalog` builds a second artefact
(`data/generated/shop_products.jsonl.gz`, 14 MB) that keeps all of it, in source
order.

## 3. Two defects worth knowing about

### `fine_items_train_persona.jsonl` is truncated

```bash
wc -l data/raw/fine_items_train_persona.jsonl     # 1616 lines
```

The file should hold the 3,323 train-split records that carry a persona. It holds
1,603 complete records plus one cut off mid-record; the remaining ~1,720 are
missing. The downloader itself flagged this: `.verified.json` records sizes for
the other two raw files and has no entry for this one, which is how the fetch
script marks a file that never passed its integrity check.

Fix: `rm data/raw/fine_items_train_persona.jsonl && python scripts/fetch_data.py
--source hf --files fine_items_train_persona.jsonl`.

Nothing in this repository depends on it. The environment and all task pools are
built from the main `.json.gz`, which is intact. Persona-conditioned tasks read
the persona from that same file.

### 106 products are dropped by the web-app price filter

`fetch_data.py` rejects prices `<= 1` (90 records) or `> 200000` (16 records),
leaving 23,315 of 23,421. Seven of the dropped records are evaluation tasks
(ids 0..1458) and 99 are train tasks.

The environment keeps them. A dropped record means a task whose target product
cannot be bought, which would score as an unwinnable task rather than as bad
model behaviour, and that is a silent way to make a benchmark look harder than
it is.

## 4. Task difficulty, measured

From the 3,421-record level statistics:

| Property | Value |
|---|---|
| Instruction length | mean 69.5 chars, p95 106, max 217 |
| Requested attributes per task | mean 4.5 |
| Requested options per task | 1 option: 17,186; 2: 5,943; 3: 276; 4: 13; 0: 3 |

And from rolling an oracle that knows the target through the **whole stack** —
harness, tools, environment, metrics — on evaluation tasks 0..119
(`python scripts/oracle_smoke.py --limit 120`):

| Measurement | Result |
|---|---|
| Finished an episode (`done`) | 105 / 120 = **87.5 %** |
| Bought the target product | 87.5 % |
| `r_hard = 1.0` | 86.7 % |
| `r_loose` | 0.872 |
| Termination of the other 15 | `turn_limit`: the target never appeared in the pages a one-shot search reached |

Two further measurements pin down where the remaining 12.5 % goes. Searching with
the raw instruction puts the target in the top 150 for 90 % of tasks (rank median
3.5, p90 56); and among tasks whose target *is* reachable, an oracle that can
re-query reaches `r_hard = 1.0` on 95.2 %. So the ceiling is a search-reachability
bound, not a scoring bound, and a policy that rewrites its query can push past
87.5 %.

| Term | Value among the 105 completed episodes |
|---|---|
| `r_type` | 1.000 |
| `r_price` | 1.000 |
| `r_option` | 1.000 on 262 of 270 tasks in the separable run below |

Two consequences that shape the whole project:

**The environment has a 95 % ceiling, and it is not the model's fault.** The
residual ~5 % are tasks whose requested option string does not match any product
option closely enough to score. Upstream rewrites `/` to ` | ` inside option
values before they become clickable but keeps the raw string in the goal, so any
task whose requested option contains `/` can be clicked correctly and still
score zero. That is a property of the release, reproducible with the oracle, and
it caps every result reported on this environment.

**`r_type` and `r_price` carry no signal on this data.** No record has a `query`
field, and upstream's `r_type` starts with `purchased.query == goal.query`; both
sides are `""`, so the term is satisfied before category or title similarity is
consulted. `r_price` compares against a ceiling drawn from a price ladder built
*above the target's own price*, so buying the target always passes. The reward
that is actually being optimised is `r_att · r_option`, which is exactly what the
reference run's numbers show (strict success 31 % after RL, loose 62.8 %).

Both are reported per sub-score rather than as a single number precisely so this
stays visible.

## 5. What upstream expects but does not ship

The ShopSimulator repository's own quick start runs `sh setup.sh`, which:

1. installs `pyserini`, Java 21 and `spacy zh_core_web_sm`;
2. builds a Lucene index from `shop_env/data/items_eval_train.json`;
3. downloads `zh_core_web_sm` for the reward's noun extraction.

Neither `search_engine/` nor `items_eval_train.json` is in the repository, and
`items_eval_train.json` is produced by a script that is also absent. Anyone
reproducing upstream from a clean clone has to reconstruct both.

This project replaces them with an in-process BM25 index over the same release
(`shoprl.env.search`, built in ~1 s) and a character-bigram stand-in for the
spaCy noun overlap. Both are deterministic, dependency-free on CPU, and
documented in `docs/harness.md`. `shoprl.env.http.RemoteShopEnv` exists for the
case where a number has to be comparable with a published one: point it at a
running upstream service and the agent code does not change.
