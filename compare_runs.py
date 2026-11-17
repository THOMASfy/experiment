#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在**同一批题**上对比多个 run 的检索指标 (recall@K / nDCG@K)。

用法:
    python compare_runs.py --subset subsets/s_val_120.json \
        --run "stock-mem0=results/mem0_bge_s/mem0_20261112-143726/predictions.jsonl" \
        --run "lme-resolve=results/mem0_lme_s_val/mem0lme_20261112-171058/predictions.jsonl" \
        --run "raw-bge=results/raw_bge_s_user/raw_xxx/predictions.jsonl"
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def load(path: str):
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[str(rec["question_id"])] = rec
    return out


def calc(pool, ks):
    n = len(pool)
    res = {"n": n}
    for k in ks:
        r_sum = 0.0
        ndcg_sum = 0.0
        for p in pool:
            gold = set(p.get("answer_session_ids") or [])
            ret = (p.get("retrieved_sessions") or [])[:k]
            r_sum += sum(1 for s in ret if s in gold) / len(gold)
            dcg = sum(1.0 / math.log2(i + 2) for i, s in enumerate(ret) if s in gold)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
            ndcg_sum += (dcg / idcg) if idcg > 0 else 0.0
        res[f"recall@{k}"] = round(r_sum / n, 6) if n else 0.0
        res[f"ndcg@{k}"] = round(ndcg_sum / n, 6) if n else 0.0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=path")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--ks", default="1,3,5,10")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    subset = None
    if args.subset:
        subset = set(json.load(open(args.subset, encoding="utf-8")))

    runs = {}
    for spec in args.run:
        label, _, path = spec.partition("=")
        runs[label] = load(path)

    # 共同题集: 有 gold 的题
    common = None
    for label, preds in runs.items():
        ids = {q for q, r in preds.items() if r.get("answer_session_ids")}
        if subset is not None:
            ids &= subset
        common = ids if common is None else (common & ids)
    common = sorted(common or [])

    print(f"共同题数 = {len(common)}")
    if not common:
        print("没有共同题, 换个 subset 或等 run 跑完")
        return

    out = {"n_common": len(common), "ks": ks, "overall": {}, "by_question_type": {}}
    type_of = {}
    for preds in runs.values():
        for q, r in preds.items():
            type_of[q] = r.get("question_type") or "unknown"

    header = f"{'run':28s} {'n':>4s}  " + "  ".join(f"r@{k:<4d}" for k in ks)
    print(header)
    print("-" * len(header))
    for label, preds in runs.items():
        pool = [preds[q] for q in common]
        m = calc(pool, ks)
        out["overall"][label] = m
        print(
            f"{label:28s} {m['n']:4d}  "
            + "  ".join(f"{m[f'recall@{k}']:.4f}" for k in ks)
        )

    print("\n按题型 recall@10:")
    types = sorted({type_of[q] for q in common})
    print(f"{'type':26s} {'n':>4s}  " + "  ".join(f"{lab[:12]:>12s}" for lab in runs))
    for qt in types:
        qids = [q for q in common if type_of[q] == qt]
        row = f"{qt:26s} {len(qids):4d}  "
        vals = []
        for label, preds in runs.items():
            m = calc([preds[q] for q in qids], ks)
            out["by_question_type"].setdefault(qt, {})[label] = m
            vals.append(f"{m[f'recall@{max(ks)}']:.4f}")
        print(row + "  ".join(f"{v:>12s}" for v in vals))

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nsaved -> {args.json_out}")


if __name__ == "__main__":
    main()
