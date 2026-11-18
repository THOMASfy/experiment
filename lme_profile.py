#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`mem0-lme` profile: LongMemEval 适配版的 mem0 V3 additive 摄入端。

背景
----
stock mem0 V3 (`Memory.add` -> `_add_to_vector_store`) 每个 session 发一次 LLM 提炼,
请求里固定带三块东西:

  * system prompt `ADDITIVE_EXTRACTION_PROMPT`   ~8.2k token
  * `## Existing Memories` (Phase 1 检索 top-10 已有记忆) —— 只用于去重/关联
  * `## Last k Messages`  (Phase 0 从 history db 取最近 10 条消息)

在 LongMemEval 上这三块都有问题:

  1. `Existing Memories` 让模型把"和已有记忆语义等价"的新事实直接丢掉。
     V3 是 ADD-only, 本来就不会 UPDATE, 所以这条 skip 规则只会丢事实 ——
     对 `knowledge-update` 这类题是硬伤。
  2. `Last k Messages` 把上一个 session 的尾巴混进来, 而 LongMemEval 每个
     session 都是独立对话, 这个上下文没有价值, 只会污染 + 烧 token。
  3. **Observation Date 根本没传进去**: `_add_to_vector_store` 调
     `generate_additive_extraction_prompt()` 时既没传 `timestamp` 也没传
     `current_date`, 于是 prompt 里的 Observation Date = 运行当天(2026-11-12),
     而 session 实际发生在 2022/2023 年。相对时间("yesterday/last week")被
     解析到完全错误的日期, 直接写进 memory 正文 -> 检索时语义也歪。

本模块做四件事, **LLM 提炼本身保留**:

  A. 上下文无关提炼: 不带 Existing Memories / Last k Messages。
  B. 传真实的 observation date(可选 resolve / verbatim 两种时间策略)。
  C. LongMemEval 专用 system prompt + custom instructions(收紧约束):
     只抽"可能被提问的个人事实", 保留精确名字/日期/数字, 丢弃通用知识和
     助手泛泛建议, 每条 session 最多 N 条记忆。
  D. **按 session 内容寻址的提炼缓存** (最大的省钱点):
     LongMemEval_M 里 223,364 个 session 只有 49,522 份不同内容(78% 跨题复用),
     缓存命中即完全跳过 LLM 调用。

落库仍然走 mem0 自己的 `_create_memory` (同样的 payload / hash / history),
检索口径仍然是 pure dense BGE, 所以和 stock mem0 的结果可以直接对比。

缓存键的三种模式 (`--lme-cache-key`):
  content       : 只按 session 内容 —— 最省, 但相对时间日期可能沿用首次出现的那天
  content_date  : 内容 + observation date —— 最保守, 复用率最低
  auto (默认)   : 文本里含相对时间词就带 date, 否则只按内容 (兼顾省钱与正确性)
"""

import hashlib
import json
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 1. 收紧约束后的 system prompt
# ---------------------------------------------------------------------------
LME_SYSTEM_PROMPT = """\
# ROLE

You are a Memory Extractor for a long-term personal memory system.
Your only operation is ADD: turn one conversation session into a small set of
self-contained, retrieval-friendly memory statements.

The downstream consumer is a **retrieval** system: a later question about the
user will be embedded and matched against these memories. A memory is useful
only if it contains the concrete entity, number, date or preference that the
question would be answered from.

# WHAT TO EXTRACT

Extract only information that is about THE USER and that a future question
could plausibly ask about:

- personal details (names, relationships, places, pets, possessions, numbers)
- preferences, habits, opinions, constraints, plans, goals
- events the user did, with when/where/who
- commitments, deadlines, recurring routines
- explicit changes to any of the above (old value -> new value, both kept)

Keep facts stated by the assistant ONLY when they are about the user
(recommendations, plans, schedules, agreements created for the user).
Drop generic assistant advice, generic world knowledge, code, and questions.

# HOW TO WRITE EACH MEMORY

- One statement per memory, 1-2 sentences, 10-60 words.
- Self-contained: replace pronouns with names / "the user".
- Keep exact entities, quantities and dates verbatim ("416 pages", "March 3",
  "the 2023 budget"). Never round, never paraphrase a number.
- If the session changes a previously known value, keep BOTH the old and the
  new value in one memory.
- Do NOT invent dates, names or quantities that are not in the session.
- Do NOT duplicate the same fact twice.
- Order the memories by how likely they are to be asked about later.

# GROUNDING RULES

- The session's Observation Date is the only anchor for relative time.
  Resolve "yesterday" / "last week" / "in three days" against it.
- Never use the Current Date for that purpose.
- If the session states no time information, do not attach a date to the fact.
- Everything you write must be supported by the session text.

# OUTPUT

A single JSON object, nothing else:

{"memory": [{"id": "0", "text": "<memory statement>", "attributed_to": "user"},
            {"id": "1", "text": "<memory statement>", "attributed_to": "user"}]}

Use "user" unless the fact is explicitly about the assistant's recommendation.
Return {"memory": []} if the session contains nothing worth remembering.
"""

LME_CUSTOM_INSTRUCTIONS = """\
- Extract at most {max_facts} memories for this session; if more candidates
  exist, keep the {max_facts} most likely to be queried later.
- Keep every proper noun, quantity, date and negation exactly as written.
- Sessions in this benchmark are independent conversations; do not assume any
  earlier context.
"""

# ---------------------------------------------------------------------------
# 1b. 短事实版 prompt: 同样的信息, 但每条 fact 更短 -> output token 少 -> 更快更便宜
# ---------------------------------------------------------------------------
LME_SYSTEM_PROMPT_SHORT = """\
# ROLE

You are a Memory Extractor for a long-term personal memory system.
Your only operation is ADD: turn one conversation session into a small set of
short, self-contained, retrieval-friendly memory statements.

The downstream consumer is a retrieval system: a later question about the user
will be embedded and matched against these memories. A memory is useful only if
it contains the concrete entity, number, date or preference the question would
be answered from.

# WHAT TO EXTRACT

Only information about THE USER that a future question could ask about:
personal details, preferences, habits, plans, goals, events with when/where/who,
commitments, deadlines, routines, and explicit changes to any of these.

Keep assistant messages only when they are about the user (recommendations,
plans, agreements made for the user). Drop generic advice, world knowledge,
code, and questions.

# HOW TO WRITE EACH MEMORY (SHORT FORM)

- ONE clause per memory, ideally 5-25 words. No long narrative.
- Self-contained: replace pronouns with names / "the user".
- Keep exact entities, quantities and dates verbatim ("416 pages", "March 3").
- If the session changes a previously known value, keep old AND new in one memory.
- Do NOT invent dates, names or quantities.
- Do NOT duplicate the same fact.
- Order by how likely each is to be queried later.

# GROUNDING RULES

- The session's Observation Date is the only anchor for relative time.
  Resolve "yesterday" / "last week" against it, never against the Current Date.
- If the session states no time information, do not attach a date.

# OUTPUT

A single JSON object, nothing else:

{"memory": [{"id": "0", "text": "<short memory statement>", "attributed_to": "user"}]}

Return {"memory": []} if the session contains nothing worth remembering.
"""

# ---------------------------------------------------------------------------
# 1c. 批量版 prompt: 一次调用提炼 K 个 session, 每个 session 单独归组
# ---------------------------------------------------------------------------
LME_SYSTEM_PROMPT_BATCH = """\
# ROLE

You are a Memory Extractor for a long-term personal memory system.
Your only operation is ADD. You will be given SEVERAL independent conversation
sessions, each with its own session id, and you must extract memories for each
of them SEPARATELY.

The downstream consumer is a retrieval system: a later question about the user
will be embedded and matched against these memories. A memory is useful only if
it contains the concrete entity, number, date or preference the question would
be answered from.

# WHAT TO EXTRACT (per session, independently)

Only information about THE USER that a future question could ask about:
personal details, preferences, habits, plans, goals, events with when/where/who,
commitments, deadlines, routines, and explicit changes to any of these.

Keep assistant messages only when they are about the user. Drop generic advice,
world knowledge, code, and questions.

# HOW TO WRITE EACH MEMORY

- One clause per memory, ideally 5-25 words.
- Self-contained: replace pronouns with names / "the user".
- Keep exact entities, quantities and dates verbatim.
- If a session changes a previously known value, keep old AND new in one memory.
- Do NOT invent dates, names or quantities. Do NOT duplicate the same fact.
- NEVER mix facts between sessions: every fact must come from the one session
  it is listed under, using that session's observation date.

# GROUNDING RULES

- Resolve relative time against THAT session's Observation Date, never against
  the Current Date.
- If a session states no time information, do not attach a date.

# OUTPUT

A single JSON object with one entry per session, keyed by session id:

{"sessions": [{"session_id": "<the id given in the input>",
                "facts": ["<short memory statement>", "..."]},
               {"session_id": "<another id>", "facts": []}]}

Include EVERY session id you were given, even if its facts list is empty.
"""

LME_CUSTOM_INSTRUCTIONS_BATCH = """\
- At most {max_facts} memories per session; keep the {max_facts} most likely to
  be queried later.
- Keep every proper noun, quantity, date and negation exactly as written.
- The sessions are independent conversations; do not carry context between them.
"""

# 相对时间词 (用于 cache key 的 auto 模式)
_RELATIVE_RE = re.compile(
    r"(?i)\b(yesterday|today|tonight|tomorrow|last\s+(night|week|month|year|weekend)|"
    r"next\s+(week|month|year)|this\s+(week|month|year|morning|afternoon|evening)|"
    r"\d+\s+(day|week|month|year)s?\s+ago|a\s+(day|week|month|year)\s+ago|"
    r"recently|earlier|later|since\s+last)\b"
)


# ---------------------------------------------------------------------------
# 2. 提炼缓存 (sqlite, 线程安全)
# ---------------------------------------------------------------------------
class ExtractionCache:
    """按 session 内容寻址的 LLM 提炼结果缓存。

    facts 表:  key -> JSON list[str]      (提炼出的记忆文本)
    vectors 表: key -> float32 blob       (文本的 BGE 向量, 避免重复 embed)
    """

    def __init__(self, path: str, store_vectors: bool = True):
        self.path = path
        self.store_vectors = store_vectors
        self._lock = threading.Lock()
        # isolation_level=None (autocommit): 之前用默认的隐式事务, 多线程共享一个
        # 连接时, A 的 COMMIT 会把 B 未提交的 INSERT 一起提交掉, B 再 COMMIT 就报
        # "cannot commit - no transaction is active" -> 题目被判失败。
        # autocommit 下每条语句自己提交, 彻底消除事务状态竞争。
        self._conn = sqlite3.connect(
            path, check_same_thread=False, timeout=60.0, isolation_level=None
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS facts (key TEXT PRIMARY KEY, facts TEXT NOT NULL)"
        )
        if store_vectors:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB NOT NULL)"
            )
        self.hits = 0
        self.misses = 0

    # -- facts -------------------------------------------------------------
    def get_facts(self, key: str) -> Optional[List[str]]:
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT facts FROM facts WHERE key=?", (key,)
                ).fetchone()
        except Exception:
            return None
        if row is None:
            with self._lock:
                self.misses += 1
            return None
        with self._lock:
            self.hits += 1
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None

    def put_facts(self, key: str, facts: Sequence[str]) -> None:
        # 缓存写失败绝不能影响题目本身
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO facts (key, facts) VALUES (?, ?)",
                    (key, json.dumps(list(facts), ensure_ascii=False)),
                )
        except Exception:
            pass

    # -- vectors -----------------------------------------------------------
    @staticmethod
    def vec_key(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get_vec(self, text: str) -> Optional[np.ndarray]:
        if not self.store_vectors:
            return None
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT vec FROM vectors WHERE key=?", (self.vec_key(text),)
                ).fetchone()
        except Exception:
            return None
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32)

    def put_vec(self, text: str, vec: np.ndarray) -> None:
        if not self.store_vectors:
            return
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO vectors (key, vec) VALUES (?, ?)",
                    (self.vec_key(text), np.asarray(vec, dtype=np.float32).tobytes()),
                )
        except Exception:
            pass

    def commit(self) -> None:
        # autocommit 模式下不需要显式 commit; 保留方法只为兼容调用方
        try:
            with self._lock:
                self._conn.commit()
        except Exception:
            pass

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"hits": self.hits, "misses": self.misses}


# ---------------------------------------------------------------------------
# 3. key / prompt / 解析
# ---------------------------------------------------------------------------
def session_cache_key(turns: List[Dict[str, Any]], date: str, key_mode: str) -> str:
    blob = json.dumps(turns, ensure_ascii=False, sort_keys=True).encode("utf-8")
    h = hashlib.md5(blob).hexdigest()
    if key_mode == "content":
        return h
    if key_mode == "content_date":
        return f"{h}|{date}"
    if key_mode == "auto":
        text = " ".join(str(t.get("content", "")) for t in turns).lower()
        return f"{h}|{date}" if _RELATIVE_RE.search(text) else h
    raise ValueError(f"unknown lme cache key mode: {key_mode}")


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    return text


def parse_facts(raw: str, max_facts: int = 0) -> List[str]:
    """从 LLM 返回里解析出记忆文本列表 (去重 + 截断)。"""
    raw = _strip_code_fences(raw)
    if not raw:
        return []
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    items = obj.get("memory") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return []
    out: List[str] = []
    seen = set()
    for it in items:
        if isinstance(it, str):
            text = it
        elif isinstance(it, dict):
            text = it.get("text") or ""
        else:
            continue
        text = (text or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    if max_facts and len(out) > max_facts:
        out = out[:max_facts]
    return out


def parse_facts_batch(raw: str, max_facts: int = 0) -> Dict[str, List[str]]:
    """解析批量提炼的返回值: {"sessions": [{"session_id": ..., "facts": [...]}]}"""
    raw = _strip_code_fences(raw)
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return {}
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}
    items = obj.get("sessions") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return {}
    out: Dict[str, List[str]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        sid = str(it.get("session_id") or "").strip()
        if not sid:
            continue
        facts = it.get("facts")
        if isinstance(facts, str):
            facts = [facts]
        if not isinstance(facts, list):
            facts = []
        clean: List[str] = []
        seen = set()
        for f in facts:
            text = (f.get("text") if isinstance(f, dict) else f) or ""
            text = str(text).strip()
            if not text or text.lower() in seen:
                continue
            seen.add(text.lower())
            clean.append(text)
        if max_facts and len(clean) > max_facts:
            clean = clean[:max_facts]
        out[sid] = clean
    return out


def format_session_turns(msgs: List[Dict[str, str]]) -> str:
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs)


def build_batch_messages(
    sessions: List[Tuple[str, str, List[Dict[str, str]]]],
    max_facts: int,
    fact_style: str,
) -> List[Dict[str, str]]:
    """把 K 个 session 拼成一次调用。sessions = [(session_id, date, msgs), ...]"""
    blocks = []
    for sid, dt, msgs in sessions:
        blocks.append(
            f"### SESSION id={sid}\n"
            f"Observation Date: {dt}\n"
            f"Conversation:\n{format_session_turns(msgs)}"
        )
    user_prompt = (
        "Extract memories for each session below.\n\n"
        + "\n\n".join(blocks)
        + "\n\n# Output:"
    )
    if fact_style == "short":
        system_prompt = LME_SYSTEM_PROMPT_SHORT.replace(
            "# OUTPUT\n\nA single JSON object, nothing else:\n\n"
            '{"memory": [{"id": "0", "text": "<short memory statement>", "attributed_to": "user"}]}\n\n'
            'Return {"memory": []} if the session contains nothing worth remembering.',
            "# OUTPUT\n\nA single JSON object, nothing else:\n\n"
            '{"sessions": [{"session_id": "<id>", "facts": ["<short memory statement>", "..."]}]}\n\n'
            "Include every session id you were given, even with an empty facts list.",
        )
    else:
        system_prompt = LME_SYSTEM_PROMPT_BATCH
    custom = LME_CUSTOM_INSTRUCTIONS_BATCH.format(max_facts=max_facts or 99)
    return [
        {"role": "system", "content": system_prompt + "\n\n" + custom},
        {"role": "user", "content": user_prompt},
    ]


def prefilter_session_ids(
    memory,
    item: Dict[str, Any],
    topk: int,
    query_prefix: str,
    logger,
) -> Optional[set]:
    """[PRE-FILTER] 用 BGE 先按问题粗筛 session, 只对命中的 session 做 LLM 提炼。

    ⚠️ 这是 **query-aware** 的: 摄入阶段就用到了问题本身。
       它把 LLM 调用从 |sessions| 降到 topk, 但严格说已经不是"先建记忆再检索"
       的记忆系统, 而是 "retrieve-then-extract" 两阶段流水线。
       论文里必须单独标注, 不能算作 mem0 的原版结果。
    """
    sessions = item.get("haystack_sessions") or []
    session_ids = list(item.get("haystack_session_ids") or [])
    if topk <= 0 or len(sessions) <= topk:
        return None

    docs: List[str] = []
    keep_ids: List[str] = []
    for i, turns in enumerate(sessions):
        msgs = clean_turns_local(turns)
        if not msgs:
            continue
        user_text = " ".join(m["content"] for m in msgs if m["role"] == "user")
        if not user_text:
            user_text = " ".join(m["content"] for m in msgs)
        docs.append(user_text[:20000])
        keep_ids.append(session_ids[i] if i < len(session_ids) else f"sess_{i}")
    if len(docs) <= topk:
        return None

    q = f"{query_prefix}{item.get('question', '')}" if query_prefix else item.get("question", "")
    q_emb = np.asarray(memory.embedding_model.embed(q, "search"), dtype=np.float32)
    d_emb = np.asarray(memory.embedding_model.embed_batch(docs, "add"), dtype=np.float32)
    qn = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    dn = d_emb / (np.linalg.norm(d_emb, axis=1, keepdims=True) + 1e-9)
    scores = dn @ qn
    order = np.argsort(-scores)[:topk]
    chosen = {keep_ids[int(i)] for i in order}
    logger.debug(
        "prefilter: %d sessions -> top%d kept (best score %.3f)",
        len(docs),
        topk,
        float(scores[order[0]]) if len(order) else 0.0,
    )
    return chosen


def clean_turns_local(turns) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for t in turns or []:
        if not isinstance(t, dict):
            continue
        content = t.get("content")
        if not content:
            continue
        out.append({"role": t.get("role") or "user", "content": str(content)})
    return out


def build_messages(
    msgs: List[Dict[str, str]],
    date: str,
    max_facts: int,
    prompt_mode: str,
    date_mode: str,
    fact_style: str = "rich",
) -> List[Dict[str, str]]:
    """构造 LLM 请求 (system + user)。"""
    from mem0.configs.prompts import (
        ADDITIVE_EXTRACTION_PROMPT,
        generate_additive_extraction_prompt,
    )

    if prompt_mode == "stock":
        system_prompt = ADDITIVE_EXTRACTION_PROMPT
    elif fact_style == "short":
        system_prompt = LME_SYSTEM_PROMPT_SHORT
    else:
        system_prompt = LME_SYSTEM_PROMPT

    # context-free: 不给 Existing Memories, 不给 Last k Messages
    # Observation Date / Current Date 都设成 session 的真实日期:
    #   - resolve 模式: 相对时间按 session 日期解析成绝对日期
    #   - verbatim 模式: 保留原始相对表述
    # 这样 prompt 里不会出现"运行当天", 避免把 2022/2023 的对话解析到 2026。
    kwargs: Dict[str, Any] = {"current_date": date}
    if date_mode == "resolve":
        kwargs["timestamp"] = date          # Observation Date = session 真实日期

    user_prompt = generate_additive_extraction_prompt(
        summary=None,
        recently_extracted_memories=None,
        existing_memories=[],
        new_messages=msgs,
        last_k_messages=[],
        custom_instructions=LME_CUSTOM_INSTRUCTIONS.format(max_facts=max_facts or 99),
        **kwargs,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------------------------------------------------------------
# 4. 写入 (LLM 提炼 + mem0 落库)
# ---------------------------------------------------------------------------
def write_sessions_lme(
    memory,
    item: Dict[str, Any],
    uid: str,
    logger,
    cache: Optional[ExtractionCache],
    max_facts: int = 0,
    key_mode: str = "auto",
    date_mode: str = "resolve",
    prompt_mode: str = "lme",
    fact_style: str = "rich",
    batch_sessions: int = 1,
    prefilter_topk: int = 0,
    query_prefix: str = "",
    write_history: bool = False,
    session_level: bool = False,
    empty_fallback: bool = True,
) -> int:
    """lme profile 的写入路径。

    与 stock mem0 的区别只在"怎么得到 fact 文本", 落库和检索完全一致:
      * fact 文本 -> payload/hash/text_lemmatized/history 全部按 mem0 原生格式
      * 检索 -> dense_retrieve() pure BGE cosine

    可选的三档"轻量化"(都只影响摄入端):
      * fact_style=short + max_facts : output token 少 -> 更快更便宜
      * batch_sessions=K             : K 个 session 一次调用 (两级缓存, 命中仍按 session)
      * prefilter_topk=K            : query-aware 粗筛, ⚠️ 会改变方法性质
    """
    sessions = item.get("haystack_sessions") or []
    session_ids = list(item.get("haystack_session_ids") or [])
    dates = list(item.get("haystack_dates") or [])

    # --- 0. 可选: query-aware 粗筛 ---
    allowed = None
    if prefilter_topk and prefilter_topk > 0:
        allowed = prefilter_session_ids(memory, item, prefilter_topk, query_prefix, logger)

    # --- 1. 逐 session 取 fact 文本 (缓存命中就完全不调 LLM) ---
    pending: List[Tuple[str, str, List[Dict[str, str]], str]] = []
    per_session: List[Tuple[str, str, List[str]]] = []
    raw_texts: Dict[str, str] = {}
    n_llm_calls = 0

    for i, turns in enumerate(sessions):
        sid = session_ids[i] if i < len(session_ids) else f"sess_{i}"
        dt = dates[i] if i < len(dates) else ""
        msgs = clean_turns_local(turns)
        if not msgs:
            continue
        raw_texts[sid] = " ".join(str(m.get("content", "")) for m in msgs)
        if allowed is not None and sid not in allowed:
            continue
        key = session_cache_key(msgs, str(dt), key_mode)
        facts = cache.get_facts(key) if cache else None
        if facts is None:
            pending.append((sid, str(dt), msgs, key))
        else:
            per_session.append((sid, str(dt), facts))

    # --- 2. 对未命中的 session 调 LLM (可批量) ---
    if batch_sessions and batch_sessions > 1 and pending:
        for start in range(0, len(pending), batch_sessions):
            chunk = pending[start : start + batch_sessions]
            messages = build_batch_messages(
                [(sid, dt, msgs) for sid, dt, msgs, _ in chunk], max_facts, fact_style
            )
            raw = memory.llm.generate_response(
                messages=messages,
                response_format={"type": "json_object"},
            )
            n_llm_calls += 1
            parsed = parse_facts_batch(raw, max_facts)
            for sid, dt, msgs, key in chunk:
                facts = parsed.get(sid, [])
                if cache is not None:
                    cache.put_facts(key, facts)
                per_session.append((sid, dt, facts))
    else:
        for sid, dt, msgs, key in pending:
            messages = build_messages(
                msgs, dt, max_facts, prompt_mode, date_mode, fact_style=fact_style
            )
            raw = memory.llm.generate_response(
                messages=messages,
                response_format={"type": "json_object"},
            )
            facts = parse_facts(raw, max_facts)
            n_llm_calls += 1
            if cache is not None:
                cache.put_facts(key, facts)
            per_session.append((sid, dt, facts))

    # --- 2b. session 级收口: 每个 session 合成一条记忆向量 ---
    # empty_fallback=True  : facts 为空时回退原始对话文本 (老行为, 会把 26% 的
    #                        空提炼 session 变成原文检索 -> 与 fact 级不可比)
    # empty_fallback=False : facts 为空则该 session 不入库 (strict, 与 fact 级
    #                        共享同一信息源, 差额只反映收口方式)
    if session_level:
        collapsed: List[Tuple[str, str, List[str]]] = []
        for sid, dt, facts in per_session:
            if facts:
                text = "\n".join(facts)
            else:
                text = raw_texts.get(sid, "") if empty_fallback else ""
            text = (text or "").strip()
            if not text:
                continue
            collapsed.append((sid, str(dt), [text]))
        per_session = collapsed

    # --- 3. 一题一次性 embed (比 stock 每 session 一次 embed_batch 快很多) ---
    texts: List[str] = []
    for _, _, facts in per_session:
        texts.extend(facts)
    uniq_texts = list(dict.fromkeys(texts))
    vectors: Dict[str, np.ndarray] = {}
    missing = []
    for t in uniq_texts:
        v = cache.get_vec(t) if cache else None
        if v is not None:
            vectors[t] = v
        else:
            missing.append(t)
    if missing:
        embs = memory.embedding_model.embed_batch(missing, "add")
        for t, e in zip(missing, embs):
            vec = np.asarray(e, dtype=np.float32)
            vectors[t] = vec
            if cache is not None:
                cache.put_vec(t, vec)
    if cache is not None:
        cache.commit()

    # --- 4. 落库 (payload / hash / history 与 stock mem0 一致, 但整题一次批量写) ---
    from mem0.utils.lemmatization import lemmatize_for_bm25

    now_iso = datetime.now(timezone.utc).isoformat()
    all_vectors: List[List[float]] = []
    all_ids: List[str] = []
    all_payloads: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    n_memories = 0

    for sid, dt, facts in per_session:
        for text in facts:
            vec = vectors.get(text)
            if vec is None:
                continue
            payload = {
                "user_id": uid,
                "session_id": sid,
                "timestamp": dt,
                "data": text,
                "hash": hashlib.md5(text.encode("utf-8")).hexdigest(),
                "created_at": now_iso,
                "updated_at": now_iso,
                "text_lemmatized": lemmatize_for_bm25(text),
            }
            memory_id = str(uuid.uuid4())
            all_vectors.append(vec.tolist())
            all_ids.append(memory_id)
            all_payloads.append(payload)
            history_rows.append(
                {
                    "memory_id": memory_id,
                    "old_memory": None,
                    "new_memory": text,
                    "event": "ADD",
                    "created_at": now_iso,
                    "is_deleted": 0,
                }
            )
            n_memories += 1

    if all_ids:
        # 一次 upsert 整题的记忆 (stock mem0 是每 session 一次, 这里更快)
        memory.vector_store.insert(
            vectors=all_vectors,
            ids=all_ids,
            payloads=all_payloads,
        )
        # [LME] 不写 mem0 的 history.db:
        #   1) 我们的检索口径是 pure dense, 不用 history 表;
        #   2) mem0 的 SQLiteManager.batch_add_history (storage.py:193) 用
        #      BEGIN + 隐式事务 + COMMIT, 多线程下会抛
        #      "cannot start a transaction within a transaction" /
        #      "cannot commit - no transaction is active",
        #      异常还会被它 except 分支里的 ROLLBACK 盖成
        #      "error return without exception set" -> 整题被判失败。
        #      (M 集每题几千条, 写 history 也是纯开销)
        if write_history:
            try:
                memory.db.batch_add_history(history_rows)
            except Exception as e:
                logger.warning("history write failed (ignored): %r", e)

    logger.debug(
        "uid=%s lme: %d sessions -> %d facts (%d llm calls, batch=%d, prefilter=%s, cache %s)",
        uid,
        len(per_session),
        n_memories,
        n_llm_calls,
        batch_sessions,
        prefilter_topk or "off",
        cache.stats() if cache else "off",
    )
    return n_memories


def fast_delete_user(memory, uid: str) -> int:
    """按 user_id 过滤直接删 Qdrant 点, 代替 mem0 的逐条 delete_all。

    mem0 的 delete_all 会 list + 逐条 delete + 逐条写 history,
    M 集每题几千条记忆、几百道题, 逐条删要花掉几小时。
    lme profile 不做实体链接, 所以只清主 collection 就够了。
    """
    try:
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            FilterSelector,
            MatchValue,
        )

        store = memory.vector_store
        client = store.client
        client.delete(
            collection_name=store.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=uid))]
                )
            ),
        )
        return 0
    except Exception:
        # 兜底: 回到 mem0 原生删除
        try:
            memory.delete_all(user_id=uid)
        except Exception:
            pass
        return -1

