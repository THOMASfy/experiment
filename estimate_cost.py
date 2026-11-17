#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""估算剩余所有 run 的 LLM 成本 (off-peak 价, DeepSeek)。"""

import json

CHARS_PER_TOK = 4


def stats(path):
    d = json.load(open(path, encoding="utf-8"))
    na = [x for x in d if not str(x["question_id"]).endswith("_abs")]
    ab = [x for x in d if str(x["question_id"]).endswith("_abs")]

    def sess(rows):
        n = sum(len(x["haystack_sessions"]) for x in rows)
        ch = sum(
            sum(len(t.get("content", "")) for s in x["haystack_sessions"] for t in s)
            for x in rows
        )
        return n, ch

    n1, c1 = sess(na)
    n2, c2 = sess(ab)
    return len(na), n1, c1, len(ab), n2, c2


def cost(calls, sys_cached, session_tok, other_tok, out_tok):
    """off-peak: cache_hit $0.003/M, cache_miss $0.15/M, output $0.60/M"""
    hit = sys_cached * calls
    miss = (session_tok + other_tok) * calls
    out = out_tok * calls
    return hit / 1e6 * 0.003 + miss / 1e6 * 0.15 + out / 1e6 * 0.60


for path, label in [("longmemeval_s_cleaned.json", "S"), ("longmemeval_m_cleaned.json", "M")]:
    nq, n, ch, nq_ab, n_ab, ch_ab = stats(path)
    print(
        f"{label}: 非拒答 {nq} 题 / {n} sessions / 平均 {ch // n} chars "
        f"(~{ch // n // CHARS_PER_TOK} tok) | 拒答 {nq_ab} 题 / {n_ab} sessions "
        f"({ch_ab // n_ab} chars)"
    )

print()
LME_SYS, LME_OTHER, LME_OUT = 600, 150, 400
STK_SYS, STK_OTHER, STK_OUT = 8155, 1600, 450

rows = [
    ("S stock  剩余 411 题", 19600, STK_SYS, 2730, STK_OTHER, STK_OUT),
    ("S lme-auto 全量 470 题", 19470, LME_SYS, 2730, LME_OTHER, LME_OUT),
    ("M lme-auto 全量 470 题", 101437, LME_SYS, 2707, LME_OTHER, LME_OUT),
    ("M 若用 stock mem0 (对照)", 223364, STK_SYS, 2707, STK_OTHER, STK_OUT),
    ("S 的 30 道拒答题 (stock)", 1431, STK_SYS, 2730, STK_OTHER, STK_OUT),
    ("M 的 30 道拒答题 (lme)", 3200, LME_SYS, 2707, LME_OTHER, LME_OUT),
]
print(f"{'item':30s} {'LLM calls':>10s} {'cost off-peak':>14s} {'active h (16w)':>16s}")
total = 0.0
for name, calls, sc, sess, oth, out in rows:
    c = cost(calls, sc, sess, oth, out)
    h = calls / 16 * 17.6 / 3600
    print(f"{name:30s} {calls:10,d} {'$' + format(c, ',.2f'):>13s} {h:15.1f} h")
    if "对照" not in name:
        total += c
print(f"\n实际要花的部分合计 (off-peak) = ${total:,.2f}   (peak 价则约 x2 = ${total*2:,.2f})")
