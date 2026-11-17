#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成一个按题型比例抽样的 question_id 子集 (给 mem0-lme 适配版做验证跑)。

用法:
    python make_subset.py --data longmemeval_s_cleaned.json --out subsets/s_val_120.json --n 120 --seed 42
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="longmemeval_s_cleaned.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-per-type", type=int, default=10)
    args = ap.parse_args()

    data = json.load(open(args.data, encoding="utf-8"))
    data = [x for x in data if not str(x["question_id"]).endswith("_abs")]
    by_type = defaultdict(list)
    for x in data:
        by_type[x["question_type"]].append(str(x["question_id"]))

    rng = random.Random(args.seed)
    total = len(data)
    picked: list[str] = []
    for qt, ids in sorted(by_type.items()):
        quota = max(args.min_per_type, round(args.n * len(ids) / total))
        quota = min(quota, len(ids))
        picked.extend(rng.sample(ids, quota))
    picked = sorted(set(picked))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(picked, indent=2), encoding="utf-8")
    cnt = defaultdict(int)
    for q in picked:
        cnt[next(x["question_type"] for x in data if str(x["question_id"]) == q)] += 1
    print(f"{args.out}: {len(picked)} questions, by type = {dict(cnt)}")


if __name__ == "__main__":
    main()
