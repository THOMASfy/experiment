"""Recompute the official metrics for every published run and compare to the stored file.

Usage:
    python mem0_results/verify_published.py

Each ``mem0_results/*/`` directory that contains both ``predictions.jsonl`` and
``aggregate_official.json`` is rescored from scratch with the official definitions
(skip ``_abs``, gold = ``answer_session_ids``, retrieved = ``retrieved_sessions``)
and every metric is compared against the committed numbers. Exit code is non-zero
on any mismatch, so this can be used in CI.

No LLM, no embeddings, no network: predictions are session-id lists only.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVAL_PATH = ROOT / "longmemeval" / "eval_retrieval_official.py"


def load_eval_module():
    spec = importlib.util.spec_from_file_location("eval_retrieval_official", EVAL_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {EVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ev = load_eval_module()
    failures = 0
    header = f"{'run':<36} {'n':>4} {'recall@5':>10} {'stored':>10} {'maxdiff':>9}"
    print(header)
    print("-" * len(header))

    for run_dir in sorted(p for p in HERE.iterdir() if p.is_dir()):
        pred_path = run_dir / "predictions.jsonl"
        agg_path = run_dir / "aggregate_official.json"
        if not pred_path.exists() or not agg_path.exists():
            continue

        rows = ev.load_predictions(str(pred_path))
        recomputed = ev.aggregate(
            rows, [5, 10], "answer_session_ids", "retrieved_sessions"
        )
        metrics = recomputed["metrics"]
        stored = json.loads(agg_path.read_text(encoding="utf-8"))["metrics"]

        missing = [k for k in metrics if k not in stored]
        diff = max((abs(metrics[k] - stored[k]) for k in metrics if k in stored), default=0.0)
        status = "ok" if not missing and diff < 1e-9 else "MISMATCH"
        if status != "ok":
            failures += 1
        print(
            f"{run_dir.name:<36} {recomputed['n']:>4} "
            f"{metrics['recall_all@5']:>10.4f} {stored.get('recall_all@5', float('nan')):>10.4f} "
            f"{diff:>9.1e}  {status}"
        )
        if missing:
            print(f"    missing keys in stored aggregate: {', '.join(missing)}")

    print()
    if failures:
        print(f"FAIL: {failures} run(s) do not reproduce")
        return 1
    print("PASS: all published runs reproduce their committed official metrics")
    return 0


if __name__ == "__main__":
    sys.exit(main())
