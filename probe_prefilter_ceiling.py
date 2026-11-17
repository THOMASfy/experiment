#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[零 LLM 成本] 测 query-aware 粗筛 (prefilter) 的召回天花板。

prefilter 的做法: 用 BGE 先按问题把 session 排序, 只对 top-K 个 session 做 LLM 提炼。
所以它的**召回上限** = "gold session 排在第几". 这个上限只需要 embedding, 不需要 LLM。

做法与 raw_bge_eval.py 完全一致 (官方 turn 粒度: 只索引 user turn, session 取最高分),
所以算出来的 top-K 召回就是 prefilter 能达到的最好情况。

用法:
    python probe_prefilter_ceiling.py --data longmemeval_s_cleaned.json --n 24
    python probe_prefilter_ceiling.py --data longmemeval_m_cleaned.json --n 12
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

KS = [8, 16, 32, 64, 128, 256]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="longmemeval_s_cleaned.json")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--random", action="store_true", help="随机抽 n 题 (默认取前 n 题)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--embedder", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--query-prefix", default="Represent this sentence for searching relevant passages: ")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.embedder, device=args.device)
    model.eval()

    data = json.load(open(args.data, encoding="utf-8"))
    data = [x for x in data if not str(x["question_id"]).endswith("_abs")]
    if args.random:
        import random

        random.Random(args.seed).shuffle(data)
    data = data[: args.n]

    print(f"{args.data}: {len(data)} questions")
    print(f"{'K':>5s} {'all-gold-in-topK':>18s} {'any-gold-in-topK':>18s}")
    hits_all = {k: 0 for k in KS}
    hits_any = {k: 0 for k in KS}
    ranks = []
    t0 = time.time()

    for qi, item in enumerate(data, 1):
        sids = item["haystack_session_ids"]
        gold = set(item.get("answer_session_ids") or [])
        docs, owner = [], []
        for i, turns in enumerate(item["haystack_sessions"]):
            user_text = " ".join(
                str(t.get("content", ""))
                for t in turns
                if isinstance(t, dict) and t.get("role") == "user" and t.get("content")
            )
            if not user_text:
                continue
            docs.append(user_text[:20000])
            owner.append(sids[i] if i < len(sids) else f"sess_{i}")
        if not docs:
            continue
        q = f"{args.query_prefix}{item['question']}"
        d_emb = model.encode(docs, batch_size=args.batch_size, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
        q_emb = model.encode([q], batch_size=1, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)[0]
        scores = d_emb @ q_emb
        best = {}
        for sid, s in zip(owner, scores):
            if sid not in best or s > best[sid]:
                best[sid] = float(s)
        ranked = [sid for sid, _ in sorted(best.items(), key=lambda kv: -kv[1])]
        pos = {sid: i + 1 for i, sid in enumerate(ranked)}
        g_ranks = [pos.get(g, 10**6) for g in gold]
        ranks.extend(g_ranks)
        for k in KS:
            if all(r <= k for r in g_ranks):
                hits_all[k] += 1
            if any(r <= k for r in g_ranks):
                hits_any[k] += 1
        if qi % 6 == 0:
            print(f"  ...{qi}/{len(data)} ({time.time()-t0:.0f}s)")

    n = len(data)
    print()
    print(f"{'K':>5s} {'all-gold-in-topK':>18s} {'any-gold-in-topK':>18s}")
    for k in KS:
        print(f"{k:>5d} {hits_all[k]/n:>17.4f} {hits_any[k]/n:>17.4f}")
    finite = [r for r in ranks if r < 10**6]
    if finite:
        print(f"\ngold session rank: median={int(np.median(finite))} "
              f"p90={int(np.percentile(finite, 90))} max={max(finite)}")
    out = {
        "data": args.data, "n": n,
        "all_gold_in_topk": {k: hits_all[k] / n for k in KS},
        "any_gold_in_topk": {k: hits_any[k] / n for k in KS},
        "gold_rank_median": int(np.median(finite)) if finite else None,
        "gold_rank_p90": int(np.percentile(finite, 90)) if finite else None,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
