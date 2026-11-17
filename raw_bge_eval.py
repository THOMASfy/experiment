#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LongMemEval 官方口径的廉价检索基线：纯 BGE，无 LLM 提炼。

用途
----
当前 mem0 V3 additive 管道每道题要对 ~47.7 个 session 各调用一次 LLM；
M 集每题 ~480 个 session，成本/耗时都会放大 10 倍。这个脚本把检索退回到
LongMemEval 官方 flat retrieval 的做法：

  1. 不调用 LLM，不生成 memory facts；
  2. 直接对 session 的 turn 文本做 BGE embedding；
  3. 按 session 聚合分数，输出 session-level recall/nDCG。

它 **不是** mem0 基线，结果必须单独标注为 `raw-bge`。它适合回答：
"在 LongMemEval 的检索指标上，花 LLM 提炼 memory 到底带来了多少增益？"

官方口径
--------
- 跳过 30 道 `_abs` 题（官方 retrieval 评估也是 skip abstention）；
- 默认只索引 user turn（和官方 `flat-* --granularity turn` 一致）；
- 对同一 session 的多个 turn 取最高分，再按 session 排序；
- 指标分母 = 非 abstention 且有 gold session 的题。

用法
----
  python raw_bge_eval.py --data longmemeval_s_cleaned.json --out-root results/raw_bge_s
  python raw_bge_eval.py --data longmemeval_s_cleaned.json --limit 59 --qids from_predictions.json
  python raw_bge_eval.py --data longmemeval_m_cleaned.json --out-root results/raw_bge_m --batch-size 128
"""

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("raw_bge_lme")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def resolve_query_prefix(bge_yaml: str, cli_prefix: str | None) -> Tuple[str, str]:
    if cli_prefix is not None:
        return cli_prefix, "cli"
    import yaml

    p = Path(bge_yaml)
    if not p.exists():
        raise FileNotFoundError(f"bge.yaml not found: {p}; pass --query-prefix explicitly")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    keys = (
        "query_instruction",
        "query_prefix",
        "instruction",
        "query_instruction_for_retrieval",
        "retrieval_instruction",
    )
    scopes = (None, "retriever", "embedder", "bge", "model")
    for scope in scopes:
        obj = cfg if scope is None else cfg.get(scope)
        if not isinstance(obj, dict):
            continue
        for k in keys:
            if k in obj and obj[k]:
                return str(obj[k]), f"{scope + '.' if scope else ''}{k}"
    raise KeyError(f"no query prefix key {keys} in {p}")


def clean_turns(turns) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for t in turns or []:
        if not isinstance(t, dict):
            continue
        content = t.get("content")
        if not content:
            continue
        out.append({"role": t.get("role") or "user", "content": str(content)})
    return out


def build_docs(
    item: Dict[str, Any],
    granularity: str,
    roles: str,
) -> List[Tuple[str, str]]:
    """返回 [(session_id, text)]；turn 粒度下同一 session 会有多条。"""
    sessions = item.get("haystack_sessions") or []
    session_ids = list(item.get("haystack_session_ids") or [])
    docs: List[Tuple[str, str]] = []
    for i, turns in enumerate(sessions):
        sid = session_ids[i] if i < len(session_ids) else f"sess_{i}"
        msgs = clean_turns(turns)
        if roles == "user":
            msgs = [m for m in msgs if m["role"] == "user"]
        if not msgs:
            continue
        if granularity == "session":
            docs.append((sid, " ".join(m["content"] for m in msgs)))
        else:
            for m in msgs:
                docs.append((sid, m["content"]))
    return docs


def rank_sessions(
    docs: List[Tuple[str, str]],
    doc_emb: np.ndarray,
    q_emb: np.ndarray,
    topk: int,
    agg: str,
) -> List[str]:
    if not docs:
        return []
    scores = doc_emb @ q_emb
    per_session: Dict[str, List[float]] = defaultdict(list)
    for (sid, _), s in zip(docs, scores):
        per_session[sid].append(float(s))
    if agg == "mean":
        sess_scores = {sid: float(np.mean(v)) for sid, v in per_session.items()}
    elif agg == "top2":
        sess_scores = {
            sid: float(np.mean(sorted(v, reverse=True)[:2])) for sid, v in per_session.items()
        }
    else:
        sess_scores = {sid: float(max(v)) for sid, v in per_session.items()}
    ranked = sorted(sess_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [sid for sid, _ in ranked[:topk]]


def evaluate(preds: List[Dict[str, Any]], topk: int, out_dir: Path, logger: logging.Logger):
    ks = sorted({k for k in (1, 3, 5, 10, 20) if k <= topk} | {topk})

    def calc(items: List[Dict[str, Any]], only_ok: bool = False):
        pool = [p for p in items if p.get("answer_session_ids")]
        if only_ok:
            pool = [p for p in pool if p.get("status") == "ok"]
        n = len(pool)
        res: Dict[str, float] = {}
        for k in ks:
            r_sum = 0.0
            ndcg_sum = 0.0
            for p in pool:
                gold = set(p.get("answer_session_ids") or [])
                ret = (p.get("retrieved_sessions") or [])[:k]
                hits = sum(1 for s in ret if s in gold)
                r_sum += hits / len(gold)
                dcg = sum(1.0 / math.log2(i + 2) for i, s in enumerate(ret) if s in gold)
                idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
                ndcg_sum += (dcg / idcg) if idcg > 0 else 0.0
            res[f"recall@{k}"] = round(r_sum / n, 6) if n else 0.0
            res[f"ndcg@{k}"] = round(ndcg_sum / n, 6) if n else 0.0
        res["n"] = n
        return res

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in preds:
        by_type[p.get("question_type", "unknown") or "unknown"].append(p)

    agg = {
        "n_total": len(preds),
        "n_ok": sum(1 for p in preds if p.get("status") == "ok"),
        "n_failed": sum(1 for p in preds if p.get("status") != "ok"),
        "n_with_gold": sum(1 for p in preds if p.get("answer_session_ids")),
        "topk": topk,
        "retrieval": "raw-bge",
        "ks": ks,
        "overall": calc(preds),
        "overall_excluding_failed": calc(preds, only_ok=True),
        "by_question_type": {qt: calc(items) for qt, items in sorted(by_type.items())},
    }
    (out_dir / "aggregate.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("aggregate: %s", json.dumps(agg["overall"], ensure_ascii=False))
    logger.info("saved %s", out_dir / "aggregate.json")
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="longmemeval_s_cleaned.json")
    ap.add_argument("--out-root", default="results/raw_bge_s")
    ap.add_argument("--resume-dir", default=None, help="已有 run 目录；会跳过已完成的 qid")
    ap.add_argument("--subset", default=None, help="JSON list of question_id")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--skip-abs", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--granularity", choices=["turn", "session"], default="turn")
    ap.add_argument("--roles", choices=["user", "all"], default="user")
    ap.add_argument("--agg", choices=["max", "mean", "top2"], default="max")
    ap.add_argument("--embedder", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--query-prefix", default=None)
    ap.add_argument("--bge-yaml", default="bge.yaml")
    args = ap.parse_args()

    if args.resume_dir:
        run_dir = Path(args.resume_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(args.out_root) / f"raw_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir / "run.log")

    query_prefix, prefix_source = resolve_query_prefix(args.bge_yaml, args.query_prefix)
    logger.info("BGE query prefix: %r (source=%s)", query_prefix, prefix_source)

    t0 = time.time()
    from sentence_transformers import SentenceTransformer

    logger.info("loading embedder %s on %s", args.embedder, args.device)
    model = SentenceTransformer(args.embedder, device=args.device)
    model.eval()

    logger.info("loading dataset %s", args.data)
    dataset = json.load(open(args.data, encoding="utf-8"))
    if args.subset:
        sub_ids = set(json.load(open(args.subset, encoding="utf-8")))
        dataset = [x for x in dataset if str(x["question_id"]) in sub_ids]
    if args.skip_abs:
        dataset = [x for x in dataset if not str(x["question_id"]).endswith("_abs")]
    if args.limit > 0:
        dataset = dataset[: args.limit]
    n_planned = len(dataset)

    pred_path = run_dir / "predictions.jsonl"
    completed: Dict[str, Dict[str, Any]] = {}
    if pred_path.exists():
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            completed[str(rec["question_id"])] = rec
    todo = [x for x in dataset if str(x["question_id"]) not in completed]

    meta = {
        "method": "raw-bge",
        "data": args.data,
        "subset": args.subset,
        "topk": args.topk,
        "skip_abs": bool(args.skip_abs),
        "n_planned": n_planned,
        "granularity": args.granularity,
        "roles": args.roles,
        "agg": args.agg,
        "embedder": args.embedder,
        "embedding_dims": int(model.get_sentence_embedding_dimension()),
        "query_prefix": query_prefix,
        "query_prefix_source": prefix_source,
        "batch_size": args.batch_size,
        "no_llm": True,
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "run_dir=%s planned=%d already_done=%d todo=%d granularity=%s roles=%s agg=%s",
        run_dir,
        n_planned,
        len(completed),
        len(todo),
        args.granularity,
        args.roles,
        args.agg,
    )

    preds: List[Dict[str, Any]] = list(completed.values())
    with open(pred_path, "a", encoding="utf-8") as out_f:
        for i, item in enumerate(todo, 1):
            qid = str(item["question_id"])
            question = item["question"]
            docs = build_docs(item, args.granularity, args.roles)
            if not docs:
                rec = {
                    "question_id": qid,
                    "question_type": item.get("question_type", ""),
                    "answer_session_ids": item.get("answer_session_ids", []),
                    "retrieved_sessions": [],
                    "ranked_session_ids": [],
                    "n_sessions": len(item.get("haystack_sessions") or []),
                    "n_docs": 0,
                    "status": "ok",
                }
            else:
                texts = [t for _, t in docs]
                doc_emb = model.encode(
                    texts,
                    batch_size=args.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                q_emb = model.encode(
                    [f"{query_prefix}{question}"],
                    batch_size=1,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )[0]
                ranked = rank_sessions(docs, doc_emb, q_emb, args.topk, args.agg)
                rec = {
                    "question_id": qid,
                    "question_type": item.get("question_type", ""),
                    "answer_session_ids": item.get("answer_session_ids", []),
                    "retrieved_sessions": ranked,
                    "ranked_session_ids": ranked,
                    "n_sessions": len(item.get("haystack_sessions") or []),
                    "n_docs": len(docs),
                    "status": "ok",
                }
            preds.append(rec)
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            if i == 1 or i % 25 == 0 or i == len(todo):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed else 0.0
                logger.info(
                    "[%d/%d] qid=%s n_docs=%d top3=%s | %.1f q/min",
                    i,
                    len(todo),
                    qid,
                    rec["n_docs"],
                    rec["retrieved_sessions"][:3],
                    rate * 60,
                )

    # 按数据集顺序重写，resume 不重复
    order = {str(x["question_id"]): i for i, x in enumerate(dataset)}
    preds.sort(key=lambda p: order.get(str(p["question_id"]), 10**9))
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    evaluate(preds, args.topk, run_dir, logger)
    logger.info("done in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
