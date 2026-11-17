#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 smoke 变体结果。"""

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/smoke_lme_s")
rows = []
for d in sorted(root.glob("smoke_*")):
    agg = d / "aggregate.json"
    usage = d / "llm_usage.json"
    if not agg.exists():
        continue
    a = json.load(open(agg, encoding="utf-8"))
    o = a["overall"]
    u = json.load(open(usage, encoding="utf-8")) if usage.exists() else {}
    # 墙钟: 从 run.log 的第一行到最后一行
    log = d / "run.log"
    wall = ""
    if log.exists():
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        if lines:
            import re
            ts = []
            for ln in lines:
                m = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", ln)
                if m:
                    ts.append(m.group(1))
            if len(ts) >= 2:
                from datetime import datetime

                fmt = "%Y-%m-%d %H:%M:%S"
                wall = round(
                    (datetime.strptime(ts[-1], fmt) - datetime.strptime(ts[0], fmt)).total_seconds() / 60,
                    1,
                )
    rows.append(
        {
            "variant": d.name.replace("smoke_", ""),
            "wall_min": wall,
            "calls": u.get("calls"),
            "retried": u.get("retried_calls"),
            "prompt_tok": u.get("prompt_tokens"),
            "cache_hit": u.get("cache_hit_ratio"),
            "out_tok": u.get("completion_tokens"),
            "cost": (u.get("est_cost_usd") or {}).get("off_peak"),
            "cache": u.get("extract_cache"),
            "r1": o.get("recall@1"),
            "r3": o.get("recall@3"),
            "r5": o.get("recall@5"),
            "r10": o.get("recall@10"),
            "ndcg5": o.get("ndcg@5"),
            "failed": a.get("n_failed"),
        }
    )

if not rows:
    print("no results yet")
    raise SystemExit(0)

hdr = (
    f"{'variant':12s} {'wall(m)':>8s} {'calls':>6s} {'retry':>6s} {'prompt_tok':>11s} "
    f"{'hit%':>6s} {'out_tok':>8s} {'cost$':>7s} {'r@1':>7s} {'r@5':>7s} {'r@10':>7s} {'fail':>5s}"
)
print(hdr)
print("-" * len(hdr))
for r in rows:
    hit = r["cache_hit"]
    hit_s = f"{hit*100:.0f}" if isinstance(hit, (int, float)) else "-"
    print(
        f"{r['variant']:12s} {str(r['wall_min']):>8s} {str(r['calls']):>6s} {str(r['retried']):>6s} "
        f"{str(r['prompt_tok']):>11s} {hit_s:>6s} {str(r['out_tok']):>8s} {str(r['cost']):>7s} "
        f"{r['r1']:>7.4f} {r['r5']:>7.4f} {r['r10']:>7.4f} {str(r['failed']):>5s}"
    )
    if r["cache"]:
        print(f"             extract_cache: {r['cache']}")
