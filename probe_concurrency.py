#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""API 并发探针: 测 DeepSeek 在不同并发下的单次延迟和总吞吐。

目的: 判断 M 能不能靠"加 workers"白拿时间。
  - 如果并发从 16 加到 32, 吞吐几乎不变 -> API 已经打满, 只能靠少调/少生成 token
  - 如果吞吐接近线性            -> workers 提到 24/32 就能把 M 的 31h 砍掉一半

用真实 session 构造 prompt (尺寸与正式跑一致), 每档 N 个并发同时打。
成本: 每档 N 次调用, 总共 ~80 次, 约 $0.06。

用法: python probe_concurrency.py --levels 8,16,24,32 --calls-per-level 8
"""

import argparse
import json
import logging
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="longmemeval_s_cleaned.json")
    ap.add_argument("--levels", default="8,16,24,32")
    ap.add_argument("--calls-per-level", type=int, default=8)
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--out", default="results/concurrency_probe.json")
    args = ap.parse_args()

    logger = logging.getLogger("probe")
    logger.setLevel(logging.WARNING)
    logger.handlers.clear()

    import run_mem0_lme as R
    import lme_profile as L

    R.patch_openai_for_mem0(
        logger, max_retries=2, max_tokens=8192, timeout=180.0, thinking="disabled"
    )

    data = json.load(open(args.data, encoding="utf-8"))
    data = [x for x in data if not str(x["question_id"]).endswith("_abs")]

    # 取一批真实 session 当 prompt (尺寸和正式跑一致)
    prompts = []
    for item in data[:40]:
        for i, turns in enumerate(item["haystack_sessions"]):
            msgs = L.clean_turns_local(turns)
            if not msgs:
                continue
            dt = item["haystack_dates"][i] if i < len(item["haystack_dates"]) else ""
            messages = L.build_messages(msgs, str(dt), 0, "lme", "resolve")
            prompts.append(messages)
            if len(prompts) >= args.calls_per_level * 4:
                break
        if len(prompts) >= args.calls_per_level * 4:
            break

    from openai import OpenAI

    client = OpenAI()

    def one(messages):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_tokens=8192,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )
            txt = resp.choices[0].message.content or ""
            u = resp.usage
            return {
                "latency": time.time() - t0,
                "out_tokens": int(getattr(u, "completion_tokens", 0) or 0),
                "in_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
                "cache_hit": int(getattr(u, "prompt_cache_hit_tokens", 0) or 0),
                "ok": bool(L.parse_facts(txt)),
            }
        except Exception as e:
            return {"latency": time.time() - t0, "ok": False, "error": repr(e)[:120]}

    results = []
    print(f"{'concurrency':>11s} {'calls':>6s} {'wall(s)':>9s} {'calls/s':>9s} "
          f"{'p50 lat':>9s} {'p90 lat':>9s} {'out tok':>8s}")
    for level in [int(x) for x in args.levels.split(",") if x.strip()]:
        batch = prompts[: args.calls_per_level]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=level) as pool:
            res = list(pool.map(one, batch))
        wall = time.time() - t0
        lats = sorted(r["latency"] for r in res)
        ok = sum(1 for r in res if r.get("ok"))
        out_tok = sum(r.get("out_tokens", 0) for r in res)
        p50 = statistics.median(lats)
        p90 = lats[int(len(lats) * 0.9) - 1] if len(lats) > 1 else lats[0]
        print(
            f"{level:>11d} {len(res):>6d} {wall:>9.1f} {len(res)/wall:>9.3f} "
            f"{p50:>9.1f} {p90:>9.1f} {out_tok:>8d}"
        )
        results.append(
            {
                "concurrency": level,
                "calls": len(res),
                "wall_s": round(wall, 2),
                "calls_per_s": round(len(res) / wall, 4),
                "latency_p50": round(p50, 2),
                "latency_p90": round(p90, 2),
                "ok": ok,
                "out_tokens": out_tok,
            }
        )

    base = results[0]["calls_per_s"] if results else 0
    for r in results:
        r["throughput_vs_lowest"] = round(r["calls_per_s"] / base, 3) if base else 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")
    print("读法: calls/s 随 concurrency 线性涨 => 加 workers 就能省时间; "
          "几乎不涨 => API 已饱和, 只能少调/少生成 token")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
