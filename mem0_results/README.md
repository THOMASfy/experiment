# mem0 on LongMemEval — published results

Two runs are published here. Only the first is a usable benchmark result.

| run | dataset | status | questions | primary metric |
|---|---|---|---:|---|
| [`mem0_lme_s/`](mem0_lme_s) | LongMemEval-S | **complete** | 470/470 ok | `recall_all@5 = 0.9234` |
| [`mem0_lme_m_partial/`](mem0_lme_m_partial) | LongMemEval-M | **incomplete, do not cite** | 98/470 ok | — |

Both use mem0 `2.2.1` (commit `f1791da`), `deepseek-flash` as the extraction LLM,
`BAAI/bge-base-en-v1.5` (768-d) embeddings, dense retrieval, `topk=10`.

---

## `mem0_lme_s/` — LongMemEval-S, 470/470

Run dir: `results/mem0_lme_s/mem0lme_20261116-090147` (not in git, see
[Files](#files)). Finished 2026-11-16 18:24.

### Configuration

| key | value |
|---|---|
| data | `longmemeval_s_cleaned.json` |
| profile | `lme` (mem0-lme; fact-style `rich`, date-mode `resolve`, prompt `lme`) |
| LLM | `deepseek-flash`, max-tokens 8192, thinking disabled |
| embedder | `BAAI/bge-base-en-v1.5`, 768 dims |
| retrieval | dense, `topk=10`, `oversample=200` |
| query prefix | `Represent this sentence for searching relevant passages: ` |
| workers | 24 |
| delete-after-question | true (memory rebuilt per question) |
| extract cache | shared sqlite, 1961 hits / 2067 misses |
| abstention | excluded (`--skip-abs`) |

### Retrieval metrics

The runner's own `aggregate.json` reports a *fractional* session recall
(`hits / len(gold)`), which is **not** the official LongMemEval metric: 300 of
470 S questions have more than one gold session. `aggregate_official.json`
re-scores the same predictions with the official definitions
(`recall_all@k` / `ndcg@k`, abstention excluded).

| metric | @1 | @3 | @5 | @10 |
|---|---:|---:|---:|---:|
| `recall_all` (official primary) | 0.2936 | 0.8426 | **0.9234** | 0.9489 |
| `ndcg` (official) | 0.9064 | 0.9197 | 0.9279 | 0.9315 |
| `recall_any` (auxiliary) | 0.9064 | 0.9511 | 0.9574 | 0.9574 |
| `fractional_recall` (runner, diagnostic) | 0.5690 | 0.9096 | 0.9465 | 0.9541 |

Per question type (`recall_all` / `ndcg`):

| type | n | R@5 | nDCG@5 | R@10 |
|---|---:|---:|---:|---:|
| knowledge-update | 72 | 1.0000 | 0.9935 | 1.0000 |
| multi-session | 121 | 0.9504 | 0.9755 | 0.9917 |
| single-session-assistant | 56 | **0.6607** | **0.6607** | 0.6607 |
| single-session-preference | 30 | 1.0000 | 0.8672 | 1.0000 |
| single-session-user | 64 | 1.0000 | 0.9885 | 1.0000 |
| temporal-reasoning | 127 | 0.9134 | 0.9471 | 0.9685 |

Gold-session counts: 170 questions with 1 gold, 229 with 2, 39 with 3, 18 with
4, 11 with 5, 3 with 6.

`single-session-assistant` is the weak spot and caps the headroom: 19 of 56
questions miss the gold session entirely at every cutoff, i.e. the failure is in
memory construction / embedding, not ranking depth.

### Cost

2067 LLM calls, 6.43M prompt tokens (41.5% cache hit) + 0.39M completion
tokens, no retries. Estimated **$0.81 off-peak / $1.62 peak** (`llm_usage.json`).

### What is *not* measured

This run evaluates **retrieval only**. There is no LLM-judge / answer-accuracy
number for mem0 on S or M in this repo yet, so the table above must not be
compared against papers that report answer accuracy.

---

## `mem0_lme_m_partial/` — LongMemEval-M, incomplete

Same configuration as above, but on `longmemeval_m_cleaned.json`. The run is
resumable and was stopped mid-way:

- 470 questions attempted, **98 ok / 372 failed**
- every failure is `APIStatusError('402 - Insufficient Balance')`, starting
  2026-11-18 03:59:39 (DeepSeek account ran out of credit)
- all completed questions are `single-session-user` (42/64) and
  `single-session-assistant` (56/56); nothing else was reached
- no `aggregate.json` was produced

`predictions.jsonl` holds one row per question (latest attempt, deduplicated
from 2820 attempt rows) so the failure pattern is auditable. **No metric should
be computed from this file.**

---

## Reproducing

```bash
# S run (from repo root, after topping up the LLM account)
python run_mem0_lme.py \
  --data longmemeval_s_cleaned.json --limit 470 --topk 10 --workers 24 \
  --retrieval dense --model deepseek-flash --max-tokens 8192 \
  --llm-thinking disabled --question-retries 3 \
  --embedder BAAI/bge-base-en-v1.5 --embedding-dims 768 \
  --query-prefix "Represent this sentence for searching relevant passages: " \
  --oversample 200 --delete-after-question --skip-abs \
  --profile lme --lme-cache-key auto --lme-date-mode resolve \
  --lme-prompt lme --lme-fact-style rich --lme-batch-sessions 1

# official re-scoring of any mem0 predictions file
python longmemeval/eval_retrieval_official.py \
  mem0_results/mem0_lme_s/predictions.jsonl \
  --gold-key answer_session_ids --retrieved-key retrieved_sessions \
  --ks 1 3 5 10 --out mem0_results/mem0_lme_s/aggregate_official.json
```

## Files

| file | contents |
|---|---|
| `*/meta.json` | full run config as recorded by the runner |
| `*/predictions.jsonl` | per-question gold/retrieved **session ids** only — no conversation text, no answers |
| `mem0_lme_s/aggregate.json` | runner metrics (fractional recall), diagnostic |
| `mem0_lme_s/aggregate_official.json` | official `recall_all` / `ndcg` re-scoring |
| `mem0_lme_s/llm_usage.json` | token counts and cost estimate |

Raw run directories (`results/`) stay local: they contain the Qdrant store,
`history.db` and multi-GB datasets. Only the artifacts above are tracked.
