#!/usr/bin/env python3
"""Compute LongMemEval-compatible retrieval metrics from a predictions.jsonl.

Why this script exists
----------------------
The runner's aggregate uses a *fractional* session recall:

    hits / len(gold_session_ids)

For questions with more than one gold session, retrieving 1 of 2 gold sessions
scores 0.5.  The official LongMemEval retrieval evaluator instead reports
`recall_all@k` (all gold sessions must be in top-k) and `ndcg_any@k`.  The
official `src/evaluation/print_retrieval_metrics.py` also excludes abstention
questions (`question_id` ending in `_abs`).

This script computes all of the following so results cannot be mixed up:

    recall_any@k        at least one gold session in top-k
    recall_all@k        every gold session in top-k          (official primary)
    fractional_recall@k hits / len(gold)                     (runner's metric)
    ndcg@k              binary-relevance NDCG               (official ranking metric)

Usage
-----
    python eval_retrieval_official.py predictions.jsonl
    python eval_retrieval_official.py predictions.jsonl --out metrics.json
    python eval_retrieval_official.py predictions.jsonl --include-abs

Other runners use different field names (e.g. mem0-lme writes `answer_session_ids`
/ `retrieved_sessions`), so the gold/retrieved keys can be overridden:

    python eval_retrieval_official.py predictions.jsonl \\
        --gold-key answer_session_ids --retrieved-key retrieved_sessions

Official references
-------------------
- https://github.com/xiaowu0162/LongMemEval/blob/main/src/retrieval/eval_utils.py
- https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/print_retrieval_metrics.py
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def normalize_session_id(value: Any) -> str:
    """Match the runner's noans_ -> answer_ normalization.

    LongMemEval uses `noans_` for sessions that are not evidence.  The runner
    normalizes both gold and retrieved ids.  This is a no-op when no such ids
    are present.
    """
    return str(value).replace("noans_", "answer_")


def dedupe(values: Iterable[Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        key = normalize_session_id(value)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def dcg(relevances: Sequence[int]) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances))


def retrieval_scores(
    retrieved: Sequence[Any],
    gold: Sequence[Any],
    ks: Sequence[int],
) -> Dict[str, float]:
    gold_set = set(normalize_session_id(x) for x in gold)
    ranked = dedupe(retrieved)
    out: Dict[str, float] = {
        "n_gold": float(len(gold_set)),
        "n_retrieved": float(len(ranked)),
    }
    if not gold_set:
        return out

    for k in ks:
        top = ranked[:k]
        hits = sum(1 for sid in top if sid in gold_set)

        out[f"recall_any@{k}"] = 1.0 if hits > 0 else 0.0
        out[f"recall_all@{k}"] = 1.0 if hits == len(gold_set) else 0.0
        out[f"fractional_recall@{k}"] = hits / len(gold_set)

        # Official definition: ideal DCG has min(len(gold), k) relevant items.
        rels = [1 if sid in gold_set else 0 for sid in top]
        ideal = [1] * min(len(gold_set), k)
        idcg = dcg(ideal)
        out[f"ndcg@{k}"] = (dcg(rels) / idcg) if idcg else 0.0
    return out


def mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def load_predictions(path: str) -> List[Dict[str, Any]]:
    by_qid: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("question_id", ""))
            if qid:
                by_qid[qid] = row
    return list(by_qid.values())


def is_abstention(row: Dict[str, Any]) -> bool:
    return bool(row.get("is_abstention")) or str(row.get("question_id", "")).endswith("_abs")


def aggregate(
    rows: Sequence[Dict[str, Any]],
    ks: Sequence[int],
    gold_key: str = "gold_session_ids",
    retrieved_key: str = "retrieved_session_ids",
) -> Dict[str, Any]:
    scored: List[Tuple[Dict[str, Any], Dict[str, float]]] = []
    for row in rows:
        gold = row.get(gold_key) or []
        if not gold:
            continue
        scores = retrieval_scores(row.get(retrieved_key) or [], gold, ks)
        if scores.get("n_gold", 0) <= 0:
            continue
        scored.append((row, scores))

    def summarize(items: Sequence[Tuple[Dict[str, Any], Dict[str, float]]]) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {"n": len(items)}
        for k in ks:
            for name in ("recall_any", "recall_all", "fractional_recall", "ndcg"):
                key = f"{name}@{k}"
                metrics[key] = mean([scores.get(key, 0.0) for _, scores in items])
        return metrics

    by_type: Dict[str, List[Tuple[Dict[str, Any], Dict[str, float]]]] = defaultdict(list)
    for row, scores in scored:
        by_type[str(row.get("question_type") or "unknown")].append((row, scores))

    gold_dist = Counter(int(scores["n_gold"]) for _, scores in scored)
    return {
        "n": len(scored),
        "ks": list(ks),
        "metrics": summarize(scored),
        "by_type": {
            qtype: summarize(items) for qtype, items in sorted(by_type.items())
        },
        "n_gold_distribution": {str(k): v for k, v in sorted(gold_dist.items())},
        "n_multi_gold": sum(1 for _, scores in scored if scores["n_gold"] > 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", help="predictions.jsonl from run_secom_lme.py")
    parser.add_argument("--out", default=None, help="optional output JSON path")
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[5, 10],
        help="cutoffs to report (default: 5 10)",
    )
    parser.add_argument(
        "--include-abs",
        action="store_true",
        help="include abstention questions (official retrieval metrics exclude them)",
    )
    parser.add_argument(
        "--gold-key",
        default="gold_session_ids",
        help="field holding gold session ids (default: gold_session_ids)",
    )
    parser.add_argument(
        "--retrieved-key",
        default="retrieved_session_ids",
        help="field holding ranked retrieved session ids (default: retrieved_session_ids)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_predictions(args.predictions)
    if not args.include_abs:
        rows = [row for row in rows if not is_abstention(row)]

    result = aggregate(rows, args.ks, args.gold_key, args.retrieved_key)
    result["predictions"] = args.predictions
    result["include_abs"] = bool(args.include_abs)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
