# LongMemEval memory experiments

Experiments on long-term conversational memory for the
[LongMemEval](https://github.com/xiaowu0162/LongMemEval) benchmark: mem0 as the
memory layer, plus a SeCom + BGE retrieval baseline.

## Results

| what | where | headline |
|---|---|---|
| mem0 on LongMemEval-**S**, 470/470 | [`mem0_results/`](mem0_results) | `recall_all@5 = 0.9234`, `ndcg@5 = 0.9279` |
| SeCom + BGE on LongMemEval-**M**, 470/470 | [`longmemeval/RETRIEVAL_RESULTS.md`](longmemeval/RETRIEVAL_RESULTS.md) | `recall_all@5 = 0.6489`, `ndcg@5 = 0.6931` |

Both are **retrieval** metrics re-scored with the official definitions
(`recall_all@k` / `ndcg@k`, abstention questions excluded). The runners' own
`aggregate.json` uses a fractional session recall and is not comparable.

Caveats: no LLM-judge answer-accuracy numbers exist yet, and the mem0-M run is
incomplete (LLM credit ran out; see `mem0_results/README.md`).

## Layout

| path | contents |
|---|---|
| `run_mem0_lme.py` | mem0 runner for LongMemEval (`--profile lme` or `stock`), resumable |
| `lme_profile.py` | prompt / fact-extraction variants for the `lme` profile |
| `schedule_runner.py`, `start_scheduler.bat` | off-peak window scheduler (resume + kill outside window) |
| `raw_bge_eval.py` | retrieval-only BGE baseline |
| `eval_predictions.py`, `compare_runs.py`, `inspect_run.py`, `estimate_cost.py` | analysis helpers |
| `probe_*.py`, `smoke_*.py` | one-off probes: concurrency, prefilter ceiling, profile smoke tests |
| `subsets/` | small fixed question subsets used by the smoke tests |
| `Secom/区/secom_lme/` | SeCom + LLMLingua-2 baseline runner |
| `longmemeval/` | official-metric re-scoring + published M artifacts |
| `mem0_results/` | published mem0 result artifacts |
| `mem0/` | upstream mem0 checkout (git-ignored, not vendored here) |

Datasets (`longmemeval_*.json`, `longmemeval.zip`) and raw run trees (`results/`)
are git-ignored; only small per-question artifacts are committed.
