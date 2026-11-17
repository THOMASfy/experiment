#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查一次 run 的落盘状态 (只读, 不修改任何数据)。

用法:
    python inspect_run.py results/mem0_smoke/mem0_20261112-142442
"""

import collections
import json
import sqlite3
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main(run_dir):
    run = Path(run_dir)
    print("=" * 74)
    print(f"RUN: {run.resolve()}")
    print("=" * 74)

    # ---------------- 1. 文件清单 ----------------
    print("\n[1] 产物文件")
    for p in sorted(run.rglob("*")):
        if p.is_file():
            print(f"   {p.relative_to(run)}  ({p.stat().st_size:,} bytes)")

    # ---------------- 2. Qdrant ----------------
    print("\n[2] Qdrant 向量库")
    from qdrant_client import QdrantClient

    qpath = run / "qdrant_storage"
    client = QdrantClient(path=str(qpath))
    per_user_points = {}
    per_user_sessions = {}
    for col in client.get_collections().collections:
        recs, _ = client.scroll(col.name, limit=100000, with_payload=True, with_vectors=False)
        info = client.get_collection(col.name)
        size = info.config.params.vectors.size
        dist = info.config.params.vectors.distance
        print(f"   collection '{col.name}': {len(recs)} points | size={size} distance={dist}")
        if not recs:
            print("      (空)")
            continue
        keys = sorted(recs[0].payload.keys())
        print(f"      payload keys: {keys}")
        by_user = collections.Counter(r.payload.get("user_id") for r in recs)
        for u, n in sorted(by_user.items(), key=lambda x: str(x[0])):
            sess = {r.payload.get("session_id") for r in recs if r.payload.get("user_id") == u}
            per_user_points[u] = n
            per_user_sessions[u] = sess
            print(f"      user_id={u}: points={n}, distinct session_id={len(sess)}")

    # ---------------- 3. SQLite history ----------------
    print("\n[3] SQLite history.db")
    db = run / "history.db"
    if db.exists():
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        cur = con.cursor()
        tables = [
            r[0]
            for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        print(f"   tables: {tables}")
        for t in tables:
            n = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"      {t}: {n:,} rows")
        if "history" in tables:
            rows = cur.execute(
                "SELECT event, COUNT(*) FROM history GROUP BY event ORDER BY 2 DESC"
            ).fetchall()
            print(f"      history by event: {rows}")
        if "messages" in tables:
            rows = cur.execute(
                "SELECT session_scope, COUNT(*) FROM messages GROUP BY session_scope ORDER BY 2 DESC LIMIT 6"
            ).fetchall()
            print(f"      messages by session_scope (top6): {rows}")
        con.close()
    else:
        print("   (不存在)")

    # ---------------- 4. predictions vs 库内数据 ----------------
    print("\n[4] predictions.jsonl vs 库内数据")
    pred_path = run / "predictions.jsonl"
    if not pred_path.exists():
        print("   (不存在)")
        return
    preds = [json.loads(l) for l in open(pred_path, encoding="utf-8") if l.strip()]
    print(f"   predictions: {len(preds)} 行")
    dataset_path = Path("longmemeval_s_cleaned.json")
    ds = {}
    if dataset_path.exists():
        for x in json.load(open(dataset_path, encoding="utf-8")):
            ds[str(x["question_id"])] = x

    for p in preds:
        qid = str(p["question_id"])
        uid = f"lme_{qid}"
        stored = per_user_points.get(uid, 0)
        stored_sess = per_user_sessions.get(uid, set())
        ret = p.get("retrieved_sessions") or []
        gold = p.get("answer_session_ids") or []
        item = ds.get(qid, {})
        hay = {str(s) for s in (item.get("haystack_session_ids") or [])}
        print(f"   qid={qid}  type={p.get('question_type')}")
        print(f"      status={p.get('status')}  n_memories_added={p.get('n_memories_added')}")
        print(f"      库内 points={stored}  distinct sessions={len(stored_sess)}")
        print(f"      数据集 haystack sessions={len(hay)}  gold={gold}")
        missing = [s for s in ret if s not in stored_sess]
        print(f"      检索返回 {len(ret)} 个 session, 其中不在库内: {missing or '无 (全部命中)'}")
        print(f"      gold 是否在库内: {[g in stored_sess for g in gold]}")
        print(f"      gold 是否在 haystack_sessions: {[g in hay for g in gold]}")
        if ret:
            print(f"      gold 在检索排名中的位置: "
                  f"{[ret.index(g) + 1 if g in ret else None for g in gold]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/mem0_smoke/mem0_20261112-142442")
