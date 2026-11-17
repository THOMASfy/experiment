#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mem0-lme 轻量化变体的 smoke 对比。

在**同一批题**上跑若干变体, 量化: LLM 调用数 / token / 成本 / 墙钟 / recall@K。

变体
----
baseline   : 现在的配置 (rich facts, 无上限, 1 session/调用, 无粗筛)
short5     : 短事实 + 每 session 最多 5 条
batch4     : 一次调用提炼 4 个 session
short5_b4  : 短事实 + 上限 5 + 4 session/调用
prefilter64: ⚠️ query-aware BGE 粗筛到 64 个 session (方法性质改变)

每个变体用**独立的 cache**, 避免互相污染 (不同配置产出的 fact 不同)。

用法:
  python smoke_lme_variants.py --subset subsets/s_smoke_24.json \
      --variants baseline,short5,batch4,short5_b4,prefilter64
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent

# 变体定义: label -> 额外 CLI 参数
VARIANTS = {
    "baseline": [],
    "short5": ["--lme-fact-style", "short", "--lme-max-facts", "5"],
    "batch4": ["--lme-batch-sessions", "4"],
    "short5_b4": ["--lme-fact-style", "short", "--lme-max-facts", "5",
                  "--lme-batch-sessions", "4"],
    "prefilter64": ["--lme-prefilter-topk", "64"],
}


def run_variant(label, extra, args, out_root):
    run_dir = Path(out_root) / f"smoke_{label}"
    cache = Path(out_root) / f"cache_{label}.sqlite"
    cmd = [
        sys.executable, "-u", str(ROOT / "run_mem0_lme.py"),
        "--resume", str(run_dir),
        "--data", args.data,
        "--profile", "lme",
        "--extract-cache", str(cache.resolve()),
        "--lme-cache-key", args.cache_key,
        "--lme-date-mode", args.date_mode,
        "--topk", "10",
        "--workers", str(args.workers),
        "--retrieval", "dense",
        "--model", args.model,
        "--max-tokens", "8192",
        "--llm-thinking", "disabled",
        "--question-retries", "3",
        "--embedder", "BAAI/bge-base-en-v1.5",
        "--embedding-dims", "768",
        "--skip-abs",
        "--delete-after-question",
        "--out-root", str(out_root),
    ]
    if args.subset:
        cmd += ["--subset", args.subset]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    cmd += extra

    print(f"\n=== {label} ===")
    print("  cmd:", " ".join(cmd[2:]))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    wall = time.time() - t0

    res = {"label": label, "wall_s": round(wall, 1), "rc": proc.returncode}
    agg_path = run_dir / "aggregate.json"
    usage_path = run_dir / "llm_usage.json"
    if agg_path.exists():
        agg = json.load(open(agg_path, encoding="utf-8"))
        o = agg.get("overall", {})
        res["recall"] = {k: o.get(f"recall@{k}") for k in (1, 3, 5, 10)}
        res["ndcg"] = {k: o.get(f"ndcg@{k}") for k in (1, 5, 10)}
        res["n_total"] = agg.get("n_total")
        res["n_failed"] = agg.get("n_failed")
        res["by_type"] = {
            qt: v.get("recall@10") for qt, v in agg.get("by_question_type", {}).items()
        }
    if usage_path.exists():
        u = json.load(open(usage_path, encoding="utf-8"))
        res["llm_calls"] = u.get("calls")
        res["prompt_tokens"] = u.get("prompt_tokens")
        res["completion_tokens"] = u.get("completion_tokens")
        res["cache_hit_ratio"] = u.get("cache_hit_ratio")
        res["cost_off_peak"] = (u.get("est_cost_usd") or {}).get("off_peak")
        res["extract_cache"] = u.get("extract_cache")
    res["predictions"] = str(run_dir / "predictions.jsonl")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="longmemeval_s_cleaned.json")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--cache-key", default="auto")
    ap.add_argument("--date-mode", default="resolve")
    ap.add_argument("--variants", default="baseline,short5,batch4")
    ap.add_argument("--out-root", default="results/smoke_lme")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for label in [v.strip() for v in args.variants.split(",") if v.strip()]:
        if label not in VARIANTS:
            print(f"[skip] unknown variant {label}")
            continue
        results.append(run_variant(label, VARIANTS[label], args, out_root))

    report = {"args": vars(args), "results": results}
    (out_root / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 100)
    hdr = (
        f"{'variant':14s} {'wall(min)':>10s} {'LLM calls':>10s} {'prompt_tok':>11s} "
        f"{'out_tok':>9s} {'cost$':>7s} {'r@1':>7s} {'r@5':>7s} {'r@10':>7s} {'n_fail':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        rc = r.get("recall") or {}
        print(
            f"{r['label']:14s} {r['wall_s']/60:>10.1f} {r.get('llm_calls', 0):>10d} "
            f"{r.get('prompt_tokens', 0):>11d} {r.get('completion_tokens', 0):>9d} "
            f"{r.get('cost_off_peak', 0):>7.2f} {rc.get(1, 0):>7.4f} {rc.get(5, 0):>7.4f} "
            f"{rc.get(10, 0):>7.4f} {r.get('n_failed', 0):>7d}"
        )
    print(f"\nsaved -> {out_root / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
