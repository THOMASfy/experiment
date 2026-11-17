#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""创建一个 mem0-lme run 目录 + meta.json (给 schedule_runner 用)。"""

import argparse
import json
from datetime import datetime
from pathlib import Path

BASE = {
    "method": "mem0-lme",
    "mem0_commit": "f1791daa86d469249c9b002f62644d58a1eb96a0",
    "mem0_version": "2.2.1",
    "subset": None,
    "topk": 10,
    "skip_abs": True,
    "n_planned": 470,
    "workers": 16,
    "llm_model": "deepseek-flash",
    "llm_max_tokens": 8192,
    "llm_thinking": "disabled",
    "llm_reasoning_effort": None,
    "llm_base_url": "https://api.deepseek.com",
    "embedder": "BAAI/bge-base-en-v1.5",
    "embedding_dims": 768,
    "query_prefix": "Represent this sentence for searching relevant passages: ",
    "query_prefix_source": "query_instruction",
    "retrieval": "dense",
    "oversample": 200,
    "delete_after_question": True,
    "profile": "lme",
    "lme_max_facts": 0,
    "question_retries": 3,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-planned", type=int, required=True)
    ap.add_argument("--extract-cache", required=True)
    ap.add_argument("--lme-cache-key", required=True, choices=["auto", "content", "content_date"])
    ap.add_argument("--lme-date-mode", required=True, choices=["resolve", "verbatim"])
    ap.add_argument("--subset", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--lme-max-facts", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out_root) / f"mem0lme_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(BASE)
    meta.update(
        {
            "data": args.data,
            "n_planned": args.n_planned,
            "extract_cache": str(Path(args.extract_cache).resolve()),
            "lme_cache_key": args.lme_cache_key,
            "lme_date_mode": args.lme_date_mode,
            "subset": args.subset,
            "workers": args.workers,
            "lme_max_facts": args.lme_max_facts,
            "tag": args.tag,
        }
    )
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(run_dir)


if __name__ == "__main__":
    main()
