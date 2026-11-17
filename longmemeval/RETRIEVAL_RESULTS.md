# LongMemEval-M retrieval: SeCom + BGE-base

Run: `results/secom_bge/auto_470`  
Date: 2026-11-16  
Retriever: `BAAI/bge-base-en-v1.5`  
Memory construction: SeCom segment + LLMLingua-2 compression (`compress_rate=0.75`)  
Retrieval: top-10 memory units, deduplicated into a session-level ranking  
Subset: 470 LongMemEval-M non-abstention questions (official retrieval metrics
also exclude `_abs`)

## TL;DR

The run's `aggregate.json` reports `recall@5 = 0.7548`, but that is a
**fractional session recall** (`hits / len(gold)`), not the official
LongMemEval metric. Re-scored with the official definitions:

| metric | value |
|---|---:|
| `recall_all@5` | **0.6489** |
| `ndcg@5` | **0.6931** |
| `recall_any@5` | 0.8553 |
| `recall_all@10` | 0.7149 |
| `ndcg@10` | 0.7149 |
| `fractional_recall@5` (runner metric, diagnostic only) | 0.7548 |

The `0.9012` in `aggregate_current.json` is an **81-question snapshot**, not the
470-question result. Those 81 questions are all single-session, all with one
gold session, and 68% are `single-session-assistant`. It should not be quoted
as the benchmark score. The same applies to the `0.8902` in
`aggregate_82.json` (82 rows; one row was an error at the time).

## Metric definitions

| metric | definition | role |
|---|---|---|
| `recall_any@k` | 1 if at least one gold session is in top-k | auxiliary |
| `recall_all@k` | 1 if **every** gold session is in top-k | official primary |
| `fractional_recall@k` | `hits / len(gold)` | runner metric; diagnostic only |
| `ndcg@k` | NDCG with binary relevance | official ranking metric |

For a question with 2 gold sessions:

- retrieve 1 of 2: `recall_any=1`, `recall_all=0`, `fractional=0.5`
- retrieve both: `recall_any=1`, `recall_all=1`, `fractional=1.0`

The official evaluator
[`src/evaluation/print_retrieval_metrics.py`](https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/print_retrieval_metrics.py)
filters out abstention questions and reports `recall_all@5` / `ndcg_any@5`
(and the `@10` versions). The official definition is in
[`src/retrieval/eval_utils.py`](https://github.com/xiaowu0162/LongMemEval/blob/main/src/retrieval/eval_utils.py):

```python
recall_any = float(any(doc in recalled_docs for doc in correct_docs))
recall_all = float(all(doc in recalled_docs for doc in correct_docs))
```

## Why the runner's 0.7548 is not the official 0.6489

300 of the 470 questions (63.8%) have more than one gold session:

| gold sessions per question | questions |
|---:|---:|
| 1 | 170 |
| 2 | 229 |
| 3 | 39 |
| 4 | 18 |
| 5 | 11 |
| 6 | 3 |

The runner gives partial credit for those questions. `recall_all` does not.
This is the entire gap between 0.7548 and 0.6489.

## Results by question type

| type | n | `recall_any@5` | `recall_all@5` | `fractional@5` | `ndcg@5` |
|---|---:|---:|---:|---:|---:|
| single-session-assistant | 56 | 1.0000 | 1.0000 | 1.0000 | 0.9713 |
| single-session-user | 64 | 0.7500 | 0.7500 | 0.7500 | 0.5735 |
| single-session-preference | 30 | 0.5667 | 0.5667 | 0.5667 | 0.4514 |
| knowledge-update | 72 | 0.8889 | 0.7083 | 0.7986 | 0.7530 |
| multi-session | 121 | 0.8926 | 0.5124 | 0.7088 | 0.6825 |
| temporal-reasoning | 127 | 0.8583 | 0.5591 | 0.7126 | 0.6641 |
| **overall** | **470** | **0.8553** | **0.6489** | **0.7548** | **0.6931** |

The single-session types have exactly one gold session, so all three recall
columns are identical. The gap appears only in multi-gold types:
`multi-session` is 0.709 fractional but 0.512 all-gold; `temporal-reasoning`
is 0.713 fractional but 0.559 all-gold.

## Comparison with the LongMemEval paper

LongMemEval Table 2 reports a session-level baseline (`Value=Session, K=V`) of
`Recall@5 = 0.706` and `NDCG@5 = 0.617` with a **Stella V5 1.5B** retriever
([paper](https://arxiv.org/abs/2410.10813)). This run uses **BGE-base 109M**.
The official-compatible `recall_all@5 = 0.6489` is in the same ballpark, not a
demonstrated improvement over that baseline. To claim a SeCom gain, run an
ablation with the same retriever, same metric, and same 470 questions:

- raw sessions + BGE
- round-level memory + BGE
- SeCom segments + BGE

## Caveats

1. **Hybrid run.** The first 81 questions use full SeCom segmentation. The
   remaining 389 use `--prefilter-topn 75 --prefilter-keep-raw` and are
   retrieval-only. The 470 result is therefore not a single configuration.
2. **Abstention.** The official retrieval metric excludes `_abs` questions, so
   removing the 30 abstention questions does not change the retrieval average.
   The `_abs` questions do have `answer_session_ids` (e.g.
   `answer_c6fd8ebd_abs`), but official retrieval does not score them.
3. **No obvious leakage.** `noans_` normalization had zero occurrences in this
   run. The prefilter's raw documents never appeared in the top-10 (0/389
   questions), so the top-10 ranking is driven by SeCom segments. Gold sessions
   are present in each question's haystack by benchmark design.
4. **Session-level ranking from unit retrieval.** Top-10 units deduplicate to
   an average of 7.27 unique sessions; `@5` uses the first 5 unique sessions.
   This is a session-level metric, not a unit-level one. The official
   `evaluate_retrieval_turn2session` also expands `k` to obtain `k` unique
   sessions.
5. **Rounding coincidence.** `ndcg@10` and `recall_all@10` both display 0.7149;
   the exact values are 0.714853 and 0.714894.

## Reproduce

```bash
python eval_retrieval_official.py artifacts/auto_470_predictions.jsonl \
  --out artifacts/auto_470_retrieval_official.json
```

The script computes `recall_any`, `recall_all`, `fractional_recall`, and `ndcg`
for `@5` and `@10`, overall and by question type. It excludes `_abs` by
default, matching the official retrieval evaluator.

## Artifacts

- `artifacts/auto_470_retrieval_official.json` — metrics above, machine-readable
- `artifacts/auto_470_predictions.jsonl` — 470 prediction rows with gold and
  retrieved session ids (questions/answers are from the public LongMemEval
  dataset)
- `artifacts/auto_470_aggregate.json` — runner aggregate for all 470 rows
  (fractional recall, retrieval-only)
- `artifacts/auto_470_aggregate_current.json` — stale 81-question snapshot
  (`recall@5 = 0.9012`); kept only to document the selection effect
- `artifacts/auto_470_aggregate_82.json` — stale 82-question snapshot
  (`recall@5 = 0.8902`)
