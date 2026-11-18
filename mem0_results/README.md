# mem0 on LongMemEval — published results

All runs below use mem0 `2.2.1` (commit `f1791da`), `deepseek-flash` as the
extraction LLM, `BAAI/bge-base-en-v1.5` (768-d) embeddings, dense retrieval,
`topk=10`. The 2026-11-18 runs reuse the extraction cache and therefore made
**zero LLM calls** (`llm_usage.json`: `calls=0`).

| run | dataset | caliber | n | recall_all@5 | nDCG@5 | status |
|---|---|---|---:|---:|---:|---|
| [`mem0_s_session_470/`](mem0_s_session_470) | S | session (raw fallback) | 470 | 0.9404 | 0.9453 | **inflated, do not cite** (see warning) |
| [`mem0_s_session_470_strict/`](mem0_s_session_470_strict) | S | session, strict | 470 | 0.9064 | 0.9086 | complete, clean ablation |
| [`mem0_lme_s/`](mem0_lme_s) | S | fact | 470 | **0.9234** | **0.9279** | complete, **main-table number** |
| [`mem0_s_151_session/`](mem0_s_151_session) | S | session | 151 | 0.9536 | 0.9503 | complete (subset) |
| [`mem0_s_151_fact/`](mem0_s_151_fact) | S | fact | 151 | 0.9007 | 0.9158 | complete (subset) |
| [`mem0_s_151_fact_os2000/`](mem0_s_151_fact_os2000) | S | fact, `oversample=2000` | 151 | 0.9007 | 0.9158 | complete; identical to os200, rules out truncation |
| [`mem0_lme_m_partial/`](mem0_lme_m_partial) | M | fact | 470 | - | - | **incomplete, do not cite** (98/470 ok) |

> WARNING: the session-level runs currently mix in a **raw-text fallback**. In the
> extraction cache 26.3% of sessions have an empty fact list (5,327 / 20,227);
> `--session-level` indexes those sessions with the raw conversation text, while the
> fact-level path has no vector for them at all. So `0.9404` is a 74% fact / 26% raw
> hybrid and must **not** be put in the same table as BGE / ThinkQE / SeCom
> session-level retrieval. The second suspected confounder, fact-level truncation at
> `oversample=200`, was tested and **ruled out**: with `oversample=2000` the 151-question
> fact run returns byte-identical retrieval lists (0/151 differ) and identical metrics.
> With the fallback removed (`--session-level-strict`, 470/470, `calls=0`) session-level
> drops to 0.9064 / 0.9086, i.e. **below** fact-level, and `single-session-assistant`
> falls from 1.0000 to exactly 0.6607 (the fact-level value). The fallback accounts for
> the entire gap; fact-level is the better granularity here.
> Evidence, per-type breakdown and the fix are in
> [`ANALYSIS_session_vs_fact.md`](ANALYSIS_session_vs_fact.md).

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

### Extraction cache

`mem0_lme_s/cache/extract_cache_s_facts.sqlite.gz` (5.1MB) is the LLM
fact-extraction output for every S session the run touched: **20,227 session
keys** (10.8MB of `key -> JSON facts` rows), keyed exactly like
`lme_profile.session_cache_key(msgs, date, "auto")`. Restoring it skips the
2,067 extraction calls entirely.

```bash
python mem0_results/restore_extract_cache.py            # fresh cache
python mem0_results/restore_extract_cache.py --merge    # add S rows to an existing cache
```

BGE vectors are **not** included: 80,994 rows / 249MB of float32 for S alone
(the full shared cache is 892MB, 626MB of which is vectors). They are
deterministic from the fact text, so the runner re-embeds whatever is missing
from the `vectors` table. `cache/manifest.json` records the sha256 of the
source cache, the dataset, and the archive; `restore_extract_cache.py` verifies
the archive checksum before unpacking, and `export_extract_cache.py` regenerates
the slice from a full cache.

The **full** S cache *with* vectors (350MB unzipped, 243MB gzip, 80,994 rows) is
published as a release asset:

> https://github.com/THOMASfy/experiment/releases/tag/mem0-s-cache-v1

```bash
python mem0_results/restore_extract_cache.py \
  --archive extract_cache_s_full.sqlite.gz --out results/extract_cache/cache_resolve_auto.sqlite
```

Use it when you want to skip re-embedding; the 5.1MB in-repo slice is enough
for correctness. Git itself must stay vectors-free: a 243MB binary in history
would bloat every clone forever.

854 of the 21,081 S session keys are absent: 30 abstention questions were
excluded (`--skip-abs`), so their sessions were never extracted.

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
| `mem0_lme_s/cache/extract_cache_s_facts.sqlite.gz` | S slice of the LLM extraction cache (facts only) + `manifest.json` |
| `mem0_s_session_470/`, `mem0_s_session_470_strict/`, `mem0_s_151_session/`, `mem0_s_151_fact/`, `mem0_s_151_fact_os2000/` | 2026-11-18 runs: `meta.json`, `predictions.jsonl`, runner + official aggregates, `llm_usage.json`, `evidence.log` |
| `ANALYSIS_session_vs_fact.md` | why session-level scores higher than fact-level, and why it is not a clean main-table number |
| release `mem0-s-cache-v1` | same slice **with** BGE vectors, 243MB gzip (not in git history) |
| `restore_extract_cache.py` | unpack / merge the cache archive |
| `export_extract_cache.py` | regenerate the archive from a full local cache |

Raw run directories (`results/`) stay local: they contain the Qdrant store,
`history.db` and multi-GB datasets. Only the artifacts above are tracked.
