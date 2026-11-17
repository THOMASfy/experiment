# SeCom LongMemEval M local runner

## Files

- `run_secom_lme.py` - SeCom + BGE runner, resumable, caches segmentation/compression
- `lme_metrics.py` - retrieval recall/NDCG + LongMemEval QA judge
- `adapt_secom.py` - writes `bge.yaml` and patches `SeCom/secom/utils.py`
- `run_local.cmd` - Windows CMD bootstrap
- `check_run.py` / `check_latest.cmd` - acceptance
- `resume.cmd` - resume latest run

## Run

1. Edit the top of `run_local.cmd`:
   - `WORK` - work directory, default `D:\longmemeval`
   - `SUBSET_SRC` - full path to `subset_m_500_ids.json`; only used if the file is not already in `WORK`
   - `OPENAI_API_KEY`
   - `OPENAI_API_BASE`
   - `SEGMENT_MODEL`

2. Run:

```bat
cd /d D:\区\secom_lme
run_local.cmd
```

`run_local.cmd` runs a 2-question smoke test, pauses, then runs the full set.

## skip_abs

With `--skip-abs`, abstention questions are removed before the run.
LongMemEval M has 500 questions and 30 abstention questions, so:

- `n_planned = 470`
- `topk = 10`
- `skip_abs = true`

Metrics are computed on 470 questions only. `check_run.py` expects 470 when
`skip_abs` is true, and 500 otherwise.
