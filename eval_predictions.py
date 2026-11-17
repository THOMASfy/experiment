#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 predictions.jsonl 离线计算 recall@K / nDCG@K (只读, 不影响正在跑的评测)。

用法:
    python eval_predictions.py results/mem0_bge_s/mem0_20261112-143726/predictions.jsonl
    python eval_predictions.py <predictions.jsonl> --ks 1,3,5,10,20

口径与 run_mem0_lme.py 的 evaluate_and_save 一致:
  - 分母 = 有 answer_session_ids 的题;
  - failed 题按 0 分计入 overall, 另给 overall_excluding_failed;
  - K 不能超过该文件实际保存的检索深度 (topk), 超过的 K 会被忽略并提示。
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
            ndcg_sum += dcg / idcg if idcg > 0 else 0.0
        res[f"recall@{k}"] = round(r_sum / n, 6) if n else 0.0
        res[f"ndcg@{k}"] = round(ndcg_sum / n, 6) if n else 0.0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=str)
    ap.add_argument("--ks", type=str, default="1,3,5,10,20")
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    preds = [
        json.loads(line)
        for line in Path(args.predictions).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not preds:
        raise SystemExit("predictions.jsonl 为空")

    max_ret = max(len(p.get("retrieved_sessions") or []) for p in preds)
    requested = sorted({int(x) for x in args.ks.split(",") if x.strip()})
    dropped = [k for k in requested if k > max_ret]
    ks = sorted({k for k in requested if k <= max_ret} | {max_ret})
    if dropped:
        print(f"[warn] 文件最多只存了 {max_ret} 条检索结果, 忽略 K={dropped}")

    pool = [p for p in preds if p.get("answer_session_ids")]
    ok_pool = [p for p in pool if p.get("status") == "ok"]

    out = {
        "file": str(Path(args.predictions).resolve()),
        "n_rows": len(preds),
        "n_ok": sum(1 for p in preds if p.get("status") == "ok"),
        "n_failed": sum(1 for p in preds if p.get("status") != "ok"),
        "max_retrieved": max_ret,
        "ks": ks,
        "overall": calc(pool, ks),
        "overall_excluding_failed": calc(ok_pool, ks),
        "by_question_type": {},
    }
    by_type = defaultdict(list)
    for p in pool:
        by_type[p.get("question_type") or "unknown"].append(p)
    for qt, items in sorted(by_type.items()):
        out["by_question_type"][qt] = calc(items, ks)

    print(f"rows={out['n_rows']}  ok={out['n_ok']}  failed={out['n_failed']}  "
          f"with_gold={len(pool)}  max_retrieved={max_ret}")
    print("overall: " + json.dumps(out["overall"], ensure_ascii=False))
    for qt, m in out["by_question_type"].items():
        print(f"  {qt:24s} n={m['n']:3d}  " + "  ".join(
            f"r@{k}={m[f'recall@{k}']:.4f}" for k in ks
        ))

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"saved -> {args.json_out}")


if __name__ == "__main__":
    main()
