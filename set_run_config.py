#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""更新某个 run 目录 meta.json 里的 lme 配置 (调度器据此重建 resume 命令)。"""

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--fact-style", choices=["rich", "short"], default=None)
    ap.add_argument("--max-facts", type=int, default=None)
    ap.add_argument("--batch-sessions", type=int, default=None)
    ap.add_argument("--prefilter-topk", type=int, default=None)
    ap.add_argument("--cache-key", choices=["auto", "content", "content_date"], default=None)
    ap.add_argument("--date-mode", choices=["resolve", "verbatim"], default=None)
    args = ap.parse_args()

    p = Path(args.run_dir) / "meta.json"
    meta = json.load(open(p, encoding="utf-8"))
    changed = {}
    if args.workers is not None:
        meta["workers"] = args.workers
        changed["workers"] = args.workers
    if args.fact_style is not None:
        meta["lme_fact_style"] = args.fact_style
        changed["lme_fact_style"] = args.fact_style
    if args.max_facts is not None:
        meta["lme_max_facts"] = args.max_facts
        changed["lme_max_facts"] = args.max_facts
    if args.batch_sessions is not None:
        meta["lme_batch_sessions"] = args.batch_sessions
        changed["lme_batch_sessions"] = args.batch_sessions
    if args.prefilter_topk is not None:
        meta["lme_prefilter_topk"] = args.prefilter_topk
        changed["lme_prefilter_topk"] = args.prefilter_topk
    if args.cache_key is not None:
        meta["lme_cache_key"] = args.cache_key
        changed["lme_cache_key"] = args.cache_key
    if args.date_mode is not None:
        meta["lme_date_mode"] = args.date_mode
        changed["lme_date_mode"] = args.date_mode
    json.dump(meta, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"updated {p}: {json.dumps(changed, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
