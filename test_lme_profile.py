#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lme_profile.py 的离线自检 (不需要 GPU / 不调 LLM)。

覆盖:
  - 缓存键三种模式 (content / content_date / auto)
  - fact 解析 (去重 + 截断 + 容错)
  - 缓存命中路径: 第二次同样内容不再调 LLM, 也不再 embed
  - 落库 payload 与 stock mem0 一致 (user_id / session_id / timestamp)
  - observation date 真的进了 prompt (stock mem0 这里有 bug)

运行: python test_lme_profile.py
"""

import json
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import lme_profile as L

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail else ""))


# ---------------------------------------------------------------------------
class FakeEmbed:
    def __init__(self):
        self.calls = 0
        self.texts = []

    def embed_batch(self, texts, action=None):
        self.calls += 1
        self.texts.extend(texts)
        return [[0.1] * 8 for _ in texts]

    def embed(self, text, memory_action=None):
        self.calls += 1
        self.texts.append(text)
        return [0.1] * 8


class FakeStore:
    def __init__(self):
        self.payloads = []

    def insert(self, vectors, ids=None, payloads=None):
        for p in payloads or []:
            self.payloads.append(p)


class FakeDB:
    def add_history(self, *a, **kw):
        pass

    def batch_add_history(self, rows):
        self.rows = rows


class FakeLLM:
    def __init__(self, owner):
        self.owner = owner

    def generate_response(self, messages=None, response_format=None, **kw):
        self.owner.llm_calls += 1
        self.owner.last_messages = messages
        return self.owner.llm_reply


class FakeMemory:
    """只实现 lme 摄入路径真正用到的那几个方法。"""

    def __init__(self, llm_reply=None):
        self.embedding_model = FakeEmbed()
        self.vector_store = FakeStore()
        self.db = FakeDB()
        self.llm_calls = 0
        self.llm = FakeLLM(self)
        self.llm_reply = llm_reply or json.dumps(
            {
                "memory": [
                    {"id": "0", "text": "User owns a dog named Max", "attributed_to": "user"},
                    {"id": "1", "text": "User was advised to install CBC", "attributed_to": "user"},
                ]
            }
        )

    def _create_memory(self, data, existing_embeddings, metadata=None):
        self.vector_store.insert(
            vectors=[existing_embeddings[data]], payloads=[metadata]
        )
        return f"mem_{len(self.vector_store.payloads)}"


# ---------------------------------------------------------------------------
print("\n[1] 缓存键模式")
turns = [{"role": "user", "content": "I went to Paris yesterday with Anna"}]
turns_plain = [{"role": "user", "content": "I own a dog named Max"}]
k_c = L.session_cache_key(turns_plain, "2023/05/20", "content")
k_cd = L.session_cache_key(turns_plain, "2023/05/20", "content_date")
k_auto_plain = L.session_cache_key(turns_plain, "2023/05/20", "auto")
k_auto_rel = L.session_cache_key(turns, "2023/05/20", "auto")
k_auto_rel2 = L.session_cache_key(turns, "2023/06/01", "auto")
check("content 模式忽略日期", k_c == L.session_cache_key(turns_plain, "2024/01/01", "content"))
check("content_date 模式区分日期", k_cd != L.session_cache_key(turns_plain, "2024/01/01", "content_date"))
check("auto: 无相对时间 -> 只按内容", k_auto_plain == k_c)
check("auto: 有相对时间 -> 带日期", k_auto_rel != k_auto_plain)
check("auto: 相对时间 session 换日期 -> key 变了", k_auto_rel != k_auto_rel2)

print("\n[2] fact 解析")
facts = L.parse_facts(
    json.dumps(
        {
            "memory": [
                {"id": "0", "text": "A"},
                {"id": "1", "text": "a"},
                {"id": "2", "text": "B"},
                {"id": "3", "text": ""},
            ]
        }
    ),
    max_facts=0,
)
check("去重 + 丢弃空串", facts == ["A", "B"], str(facts))
check("max_facts 截断", L.parse_facts(json.dumps({"memory": [{"text": str(i)} for i in range(10)]}), 3) == ["0", "1", "2"])
check("坏 JSON 不炸", L.parse_facts("not json") == [])
check("markdown 围栏能剥", L.parse_facts('```json\n{"memory":[{"text":"X"}]}\n```') == ["X"])

print("\n[3] observation date 进 prompt (stock mem0 这里丢掉了)")
msgs = [{"role": "user", "content": "I went to Paris yesterday"}]
m = L.build_messages(msgs, "2023/05/20 (Sat) 02:21", 6, "lme", "resolve")
user_text = m[1]["content"]
check("prompt 里出现真实 observation date", "2023/05/20" in user_text)
check("prompt 里没有 today's date", "2026-11-12" not in user_text)
check("prompt 不含 Existing Memories 内容", "## Existing Memories\n[]" in user_text or "## Existing Memories" in user_text)
check("system prompt 是 lme 精简版 (远短于 stock 8.2k)", len(m[0]["content"]) < 6000, f"{len(m[0]['content'])} chars")
m_stock = L.build_messages(msgs, "2023/05/20", 6, "stock", "resolve")
check("stock prompt 可选且更长", len(m_stock[0]["content"]) > 20000, f"{len(m_stock[0]['content'])} chars")

print("\n[4] 写入路径 + 缓存")
tmpdir = Path(tempfile.mkdtemp())
cache = L.ExtractionCache(str(tmpdir / "cache.sqlite"))
mem = FakeMemory()
item = {
    "question_id": "q1",
    "haystack_session_ids": ["s1", "s2"],
    "haystack_dates": ["2023/05/20", "2023/05/21"],
    "haystack_sessions": [
        [{"role": "user", "content": "I own a dog named Max"}],
        [{"role": "assistant", "content": "Nice, Max is a good name"}],
    ],
}
n = L.write_sessions_lme(mem, item, "lme_q1", __import__("logging").getLogger("t"), cache)
check("写入了 fact 数", n == 4, f"n={n}")  # 2 session x 2 facts
check("每个 session 调了一次 LLM", mem.llm_calls == 2, f"calls={mem.llm_calls}")
check("一次 embed_batch (整题一次)", mem.embedding_model.calls == 1, f"calls={mem.embedding_model.calls}")
pay = mem.vector_store.payloads
check("payload 带 user_id", all(p.get("user_id") == "lme_q1" for p in pay))
check("payload 带 session_id", {p["session_id"] for p in pay} == {"s1", "s2"})
check("payload 带 timestamp", {p["timestamp"] for p in pay} == {"2023/05/20", "2023/05/21"})

# 第二个 question: session 内容相同 -> 应该全部命中缓存
mem2 = FakeMemory()
n2 = L.write_sessions_lme(mem2, item, "lme_q1b", __import__("logging").getLogger("t"), cache)
check("缓存命中: 0 次 LLM 调用", mem2.llm_calls == 0, f"calls={mem2.llm_calls}")
check("缓存命中: 0 次 embed", mem2.embedding_model.calls == 0, f"calls={mem2.embedding_model.calls}")
check("缓存命中: fact 数一致", n2 == 4, f"n={n2}")
check("缓存命中后 payload 仍带各自的 session_id", {p["session_id"] for p in mem2.vector_store.payloads} == {"s1", "s2"})

print("\n[5] max_facts 截断真的生效")
mem3 = FakeMemory()
mem3.llm_reply = json.dumps({"memory": [{"text": f"fact {i}"} for i in range(12)]})
n3 = L.write_sessions_lme(
    mem3,
    {
        "question_id": "q3",
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2023/05/20"],
        "haystack_sessions": [[{"role": "user", "content": "hello"}]],
    },
    "lme_q3",
    __import__("logging").getLogger("t"),
    None,
    max_facts=5,
)
check("每 session 最多 5 条", n3 == 5, f"n={n3}")

print("\n[6] cache key=content 时, 换日期也命中")
cache2 = L.ExtractionCache(str(tmpdir / "cache2.sqlite"))
item_a = dict(item, haystack_dates=["2023/05/20", "2023/05/21"])
item_b = dict(item, haystack_dates=["2024/09/09", "2024/09/10"])
m4 = FakeMemory()
L.write_sessions_lme(m4, item_a, "u1", __import__("logging").getLogger("t"), cache2, key_mode="content")
m5 = FakeMemory()
L.write_sessions_lme(m5, item_b, "u2", __import__("logging").getLogger("t"), cache2, key_mode="content")
check("content 模式: 换日期仍然 0 次 LLM", m5.llm_calls == 0, f"calls={m5.llm_calls}")
m6 = FakeMemory()
L.write_sessions_lme(m6, item_b, "u3", __import__("logging").getLogger("t"), cache2, key_mode="content_date")
check("content_date 模式: 换日期要重新调 LLM", m6.llm_calls == 2, f"calls={m6.llm_calls}")

print("\n" + "=" * 60)
print(f"PASS {len(PASS)} / FAIL {len(FAIL)}")
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
