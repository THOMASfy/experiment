#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_mem0_lme.py 的离线自检 (不需要 mem0 / torch / GPU)。

覆盖的是这次修正里最容易悄悄坏掉的几块纯逻辑:
  - bge.yaml query 前缀解析 (含"必须硬失败"的语义)
  - has_answer 剔除
  - dense 检索结果去重 + 截断
  - Recall@K / NDCG@K 的分母口径
  - OpenAI patch 的 max_tokens 硬下限 + JSON 校验重试

运行:  python test_run_mem0_lme.py
"""

import json
import logging
import sys
import tempfile
import types
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import openai

import run_mem0_lme as R

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail else ""))


# ---------------------------------------------------------------------------
print("\n[1] bge.yaml query prefix 解析")
tmp = Path(tempfile.mkdtemp())

def _raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


check("缺文件 -> FileNotFoundError", _raises(lambda: R.resolve_bge_query_prefix(str(tmp / "nope.yaml"), None), FileNotFoundError))

p = tmp / "a.yaml"
p.write_text("query_instruction: 'AAA: '\n", encoding="utf-8")
check("顶层 query_instruction", R.resolve_bge_query_prefix(str(p), None) == ("AAA: ", "query_instruction"))

p = tmp / "b.yaml"
p.write_text("retriever:\n  query_prefix: 'BBB: '\n", encoding="utf-8")
check("嵌套 retriever.query_prefix", R.resolve_bge_query_prefix(str(p), None) == ("BBB: ", "retriever.query_prefix"))

p = tmp / "c.yaml"
p.write_text("model:\n  name: bge\n  instruction: 'CCC: '\n", encoding="utf-8")
check("model.instruction", R.resolve_bge_query_prefix(str(p), None) == ("CCC: ", "model.instruction"))

p = tmp / "d.yaml"
p.write_text("something_else: 1\n", encoding="utf-8")
check("找不到 key -> KeyError (不再静默回退)", _raises(lambda: R.resolve_bge_query_prefix(str(p), None), KeyError))

check("cli override 优先", R.resolve_bge_query_prefix(str(tmp / "nope.yaml"), "ZZZ: ") == ("ZZZ: ", "cli"))


# ---------------------------------------------------------------------------
print("\n[2] turn 清洗 (剔除 has_answer)")
turns = [
    {"role": "user", "content": "hello", "has_answer": True},
    {"role": "assistant", "content": "hi", "has_answer": False},
    {"role": "user", "content": ""},
    {"role": "user"},
    "not-a-dict",
]
clean = R.clean_turns(turns)
check("只保留 role/content", all(set(c) == {"role", "content"} for c in clean))
check("丢掉空 content", len(clean) == 2)
check("has_answer 被剔除", "has_answer" not in clean[0])


# ---------------------------------------------------------------------------
print("\n[3] dense 检索结果去重 + 截断")


def h(sid):
    return types.SimpleNamespace(payload={"session_id": sid}, score=0.9)


hits = [h("s3"), h("s1"), h("s3"), h("s2"), h(None), h("s4")]
ranked = R.hits_to_sessions(hits, topk=3)
check("去重且保序", ranked == ["s3", "s1", "s2"], str(ranked))
check("空 payload 不炸", R.hits_to_sessions([types.SimpleNamespace(payload=None)], 5) == [])


# ---------------------------------------------------------------------------
print("\n[4] 指标口径")


def mk(qid, gold, ret, status="ok", qt="multi-session"):
    return {
        "question_id": qid,
        "question_type": qt,
        "answer_session_ids": gold,
        "retrieved_sessions": ret,
        "status": status,
    }


preds = [
    mk("q1", ["a"], ["a"]),            # recall@1 = 1
    mk("q2", ["b"], ["x", "b"]),        # recall@1 = 0, recall@2 = 1
    mk("q3", ["c"], ["y"]),             # 全 0
    mk("q4", [], [], status="ok"),      # 无 gold -> 不进分母
    mk("q5", ["d"], [], status="failed"),  # 失败 -> 不进分母
]
out = R.evaluate_and_save(preds, topk=2, retrieval="dense", out_dir=tmp, logger=logging.getLogger("t"))
ov = out["overall"]
ovx = out["overall_excluding_failed"]
check("分母 = 有 gold 的题 (4, 失败的题按 0 分计入)", ov["n"] == 4, str(ov["n"]))
check("recall@1 = 1/4 (保守口径, 分母与其它基线一致)", abs(ov["recall@1"] - 0.25) < 1e-6, str(ov["recall@1"]))
check("recall@2 = 2/4", abs(ov["recall@2"] - 0.5) < 1e-6, str(ov["recall@2"]))
check("excluding_failed 分母 = 3", ovx["n"] == 3, str(ovx["n"]))
check("excluding_failed recall@1 = 1/3", abs(ovx["recall@1"] - 1 / 3) < 1e-6, str(ovx["recall@1"]))
check("excluding_failed recall@2 = 2/3", abs(ovx["recall@2"] - 2 / 3) < 1e-6, str(ovx["recall@2"]))
check("n_total=5", out["n_total"] == 5)
check("n_failed=1", out["n_failed"] == 1)
check("n_with_gold=4", out["n_with_gold"] == 4)
check("aggregate.json 已落盘", (tmp / "aggregate.json").exists())

print("     (原方案口径下 recall@1 会被算成 0.2 —— 分母 5 但空 gold 被跳过)")

print("\n[4b] NDCG 手算校验")
import math

one = [mk("q1", ["a", "b"], ["a", "x"])]
out2 = R.evaluate_and_save(one, topk=2, retrieval="dense", out_dir=tmp, logger=logging.getLogger("t"))
# dcg = 1/log2(2) = 1 ; idcg = 1/log2(2) + 1/log2(3) = 1 + 0.63093
expected = 1.0 / (1.0 + 1.0 / math.log2(3))
check("ndcg@2 手算一致", abs(out2["overall"]["ndcg@2"] - expected) < 1e-6,
      f"{out2['overall']['ndcg@2']} vs {expected:.6f}")


# ---------------------------------------------------------------------------
print("\n[5] OpenAI patch: max_tokens 硬下限 + JSON 校验")

from openai.resources.chat.completions import Completions

captured = []
calls = {"n": 0}
fake_content = ["not json at all", '{"memory": []}']


def fake_create(self, *args, **kwargs):
    captured.append(dict(kwargs))
    calls["n"] += 1
    content = fake_content[min(calls["n"] - 1, len(fake_content) - 1)]
    msg = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=msg, finish_reason="stop")
    return types.SimpleNamespace(choices=[choice])


orig_create = Completions.create
orig_init = openai.OpenAI.__init__
Completions.create = fake_create
logger = logging.getLogger("patch-test")
logger.addHandler(logging.NullHandler())

R.patch_openai_for_mem0(logger, max_retries=3, max_tokens=8192)
try:
    Completions.create(
        None,
        model="deepseek-flash",
        messages=[{"role": "user", "content": "x"}],
        response_format={"type": "json_object"},
        max_tokens=2000,          # <- mem0 一定会带上这个值
        temperature=0.0,
    )
finally:
    Completions.create = orig_create
    openai.OpenAI.__init__ = orig_init

check("mem0 传的 max_tokens=2000 被抬到 8192", captured[0]["max_tokens"] == 8192, str(captured[0]["max_tokens"]))
check("thinking 被强制关闭", captured[0].get("extra_body", {}).get("thinking") == {"type": "disabled"})
check("不发 reasoning_effort (DeepSeek OpenAI 格式不接受 none)",
      "reasoning_effort" not in captured[0], str(captured[0].get("reasoning_effort")))
check("坏 JSON 触发重试 (调用 2 次)", calls["n"] == 2, f"calls={calls['n']}")

# thinking=omit 时不应注入 extra_body
captured.clear()
calls["n"] = 0
Completions.create = fake_create
R.patch_openai_for_mem0(logger, max_retries=1, max_tokens=4096, thinking="omit")
try:
    Completions.create(None, model="m", messages=[{"role": "user", "content": "x"}])
finally:
    Completions.create = orig_create
    openai.OpenAI.__init__ = orig_init
check("thinking=omit 时不注入 extra_body", "extra_body" not in captured[0], str(captured[0].get("extra_body")))

# 显式指定 reasoning_effort 时才发送
captured.clear()
Completions.create = fake_create
R.patch_openai_for_mem0(logger, max_retries=1, max_tokens=4096, reasoning_effort="high")
try:
    Completions.create(None, model="m", messages=[{"role": "user", "content": "x"}])
finally:
    Completions.create = orig_create
    openai.OpenAI.__init__ = orig_init
check("显式指定时发送 reasoning_effort=high", captured[0].get("reasoning_effort") == "high")


# ---------------------------------------------------------------------------
print("\n[6] native 检索用的新 API (mem0 main 已不接受顶层 user_id/limit)")
import inspect

src = (
    "def search(self, query, *, top_k=20, filters=None, threshold=0.1, **kwargs): ..."
)
check("dense 路径不调用 Memory.search(user_id=...)", "memory.search(" not in Path(R.__file__).read_text(encoding="utf-8").split("def native_retrieve")[0].split("def dense_retrieve")[1])

# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"PASS={len(PASS)}  FAIL={len(FAIL)}")
if FAIL:
    print("FAILED:")
    for f in FAIL:
        print("  -", f)
    raise SystemExit(1)
print("ALL TESTS PASSED")
