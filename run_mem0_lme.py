#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mem0 x LongMemEval_S 检索评测 —— 修正版 runner (BGE 对齐 / pure-dense)

这是 `mem0_longmemeval_s.md` 内嵌 `run_mem0_lme.py` 的可直接替换版本。
每一处改动都用 [FIX-xx] 标注，完整对照表见同目录 `PLAN_FIXES.md`。

目标版本: mem0ai/mem0 @ main (V3 additive extraction pipeline)
检索口径: pure dense BGE (与 SeCom / A-Mem / ThinkQE 同表口径一致)

用法:
    python3 run_mem0_lme.py --data longmemeval_s_cleaned.json --topk 10 --workers 16
    python3 run_mem0_lme.py --resume results/mem0_bge_s/mem0_20260101-120000 ...
"""

import argparse
import functools
import inspect
import json
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openai
from dotenv import load_dotenv
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# 0. 环境变量 (必须在 import mem0 之前设置)
# ---------------------------------------------------------------------------
for env_candidate in [
    Path.home() / "dot_env" / "openai.env",
    Path("/workspace/dot_env/openai.env"),
    Path(".env"),
]:
    if env_candidate.exists():
        load_dotenv(env_candidate, override=False)

if os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.environ["OPENAI_API_BASE"]

# [FIX-C5a] mem0 默认开启 PostHog telemetry: 每次 add/search 都会外发请求,
# 并在 ~/.mem0 下额外创建 migrations_qdrant 集合。评测机上必须关掉。
# (可用 --telemetry on 覆盖; 必须在 import mem0 之前生效)
os.environ["MEM0_TELEMETRY"] = "False"

DEFAULT_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# DeepSeek 官方定价 (https://api-docs.deepseek.com/quick_start/pricing), per 1M tokens
DEEPSEEK_PRICING = {
    "off_peak": {"cache_hit": 0.003, "cache_miss": 0.15, "output": 0.60},
    "peak": {"cache_hit": 0.006, "cache_miss": 0.30, "output": 1.20},
}

# 累计 LLM 用量与上下文缓存命中情况。
# mem0 的提炼 system prompt 是固定的 ~8.2k token, 占每次请求 input 的 90%,
# 而 DeepSeek 的 Context Caching 默认开启且前缀一致 -> 应该几乎全部命中,
# 这是本次评测成本的主要变量, 必须可观测。
_LLM_USAGE = {
    "calls": 0,
    "retried_calls": 0,
    "prompt_tokens": 0,
    "cache_hit_tokens": 0,
    "cache_miss_tokens": 0,
    "completion_tokens": 0,
}
_USAGE_LOCK = threading.Lock()


def _record_usage(resp, retried: bool) -> None:
    u = getattr(resp, "usage", None)
    if u is None:
        return
    with _USAGE_LOCK:
        _LLM_USAGE["calls"] += 1
        if retried:
            _LLM_USAGE["retried_calls"] += 1
        _LLM_USAGE["prompt_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
        _LLM_USAGE["cache_hit_tokens"] += int(getattr(u, "prompt_cache_hit_tokens", 0) or 0)
        _LLM_USAGE["cache_miss_tokens"] += int(getattr(u, "prompt_cache_miss_tokens", 0) or 0)
        _LLM_USAGE["completion_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)


def llm_usage_summary() -> Dict[str, Any]:
    with _USAGE_LOCK:
        s = dict(_LLM_USAGE)
    total_in = s["cache_hit_tokens"] + s["cache_miss_tokens"]
    s["cache_hit_ratio"] = round(s["cache_hit_tokens"] / total_in, 4) if total_in else 0.0
    s["est_cost_usd"] = {
        tier: round(
            (s["cache_hit_tokens"] / 1e6) * p["cache_hit"]
            + (s["cache_miss_tokens"] / 1e6) * p["cache_miss"]
            + (s["completion_tokens"] / 1e6) * p["output"],
            2,
        )
        for tier, p in DEEPSEEK_PRICING.items()
    }
    return s

# Windows 控制台默认 cp950, 打印中文 help / 日志会 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# bge.yaml 里可能出现的 query 前缀 key (含 SeCom 仓库历史命名)
PREFIX_KEYS = (
    "query_instruction",
    "query_prefix",
    "instruction",
    "query_instruction_for_retrieval",
    "retrieval_instruction",
)
PREFIX_SCOPES = (None, "retriever", "embedder", "bge", "model")


# ---------------------------------------------------------------------------
# 1. 日志
# ---------------------------------------------------------------------------
def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("mem0_lme")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# 2. Monkey-Patch OpenAI SDK
# ---------------------------------------------------------------------------
def _extract_json_ok(text: str) -> bool:
    if not text or not text.strip():
        return False
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    try:
        json.loads(cleaned)
        return True
    except Exception:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if m:
            try:
                json.loads(m.group(0))
                return True
            except Exception:
                return False
        return False


def patch_openai_for_mem0(
    logger: logging.Logger,
    max_retries: int = 3,
    max_tokens: int = 8192,
    timeout: float = 180.0,
    thinking: str = "disabled",
    reasoning_effort: Optional[str] = None,
    connect_timeout_s: float = 20.0,
):
    """强制关闭 thinking + 兜底 max_tokens + 暴露 bad response。

    参数
    ----
    thinking:
        "disabled" / "enabled" —— 通过 extra_body 传 {"thinking": {"type": ...}}。
        "omit" —— 完全不传该参数(交给服务端默认)。
        DeepSeek 官方文档: thinking 模式 *默认开启*, 默认 effort=high。
    reasoning_effort:
        官方 DeepSeek API 在 OpenAI 格式下只接受 low / high / max;
        "none" 只是 Anthropic 格式的取值, 传给 OpenAI 格式的端点会被 400。
        因此默认 None = 不发这个参数 (thinking 已关掉时 effort 本来就无意义)。
    """
    from openai.resources.chat.completions import Completions

    orig_create = Completions.create
    orig_openai_init = openai.OpenAI.__init__

    # [FIX-C9] mem0 构造 OpenAI client 时不带 timeout, 代理挂起会让 worker
    # 永久阻塞。给所有 client 一个默认超时。
    # [ROBUST] 断网/休眠唤醒时, connect 会一直挂到 read timeout 才报错 ——
    # 一次 180s, 3 次重试 x 3 次题级重试 = 一题能白等 27 分钟。
    # 这里给 connect 单独一个短超时(默认 20s), 让"网络不通"快速失败,
    # 剩下的时间交给题级重试去等网络恢复。
    import httpx

    connect_timeout = min(float(connect_timeout_s), float(timeout))

    def patched_openai_init(self, *args, **kwargs):
        kwargs.setdefault(
            "timeout",
            httpx.Timeout(
                connect=connect_timeout,
                read=float(timeout),
                write=float(timeout),
                pool=connect_timeout,
            ),
        )
        return orig_openai_init(self, *args, **kwargs)

    openai.OpenAI.__init__ = patched_openai_init

    @functools.wraps(orig_create)
    def wrapped_create(self, *args, **kwargs):
        # 强制关闭 deepseek thinking (官方 DeepSeek 默认开启 thinking=high)
        if thinking and thinking != "omit":
            extra_body = dict(kwargs.get("extra_body") or {})
            extra_body["thinking"] = {"type": thinking}
            kwargs["extra_body"] = extra_body

        # [FIX-D1] 不再无条件塞 reasoning_effort="none":
        #   DeepSeek 官方 API 的 OpenAI 格式只接受 low/high/max, "none" 会被 400。
        #   需要时通过 --llm-reasoning-effort 显式传。
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        else:
            kwargs.pop("reasoning_effort", None)

        # [FIX-B2] 关键修复:
        #   mem0 的 LLMBase._get_common_params() 总是把 max_tokens 放进请求体
        #   (params["max_tokens"] = self.config.max_tokens, 默认 2000),
        #   所以原方案的 kwargs.setdefault("max_tokens", 4096) 永远不生效,
        #   提炼用的 JSON 仍可能被截断。这里改成硬下限。
        try:
            current = int(kwargs.get("max_tokens") or 0)
        except (TypeError, ValueError):
            current = 0
        kwargs["max_tokens"] = max(current, int(max_tokens))

        expect_json = (
            isinstance(kwargs.get("response_format"), dict)
            and kwargs["response_format"].get("type") == "json_object"
        )

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = orig_create(self, *args, **kwargs)
                choice = resp.choices[0]
                content = choice.message.content or ""
                # 用量统计必须放在 JSON 校验 *之前* —— 校验失败被重试的调用
                # 同样计费, 不能漏。
                _record_usage(resp, retried=attempt > 1)
                # Mem0 要求 JSON 时直接校验, 防止内部静默吞掉解析失败。
                # [FIX-D2] DeepSeek JSON 模式在无法识别为 JSON 时会把内容替换成
                # 空格(U+0020), 看起来是空字符串; 只有 finish_reason 能区分
                # "被截断" 和 "模型没按 JSON 输出"。
                if expect_json and not _extract_json_ok(content):
                    logger.error(
                        "bad response (invalid JSON, attempt %d/%d, finish_reason=%s, "
                        "len=%d, stripped_len=%d): %r",
                        attempt,
                        max_retries,
                        getattr(choice, "finish_reason", None),
                        len(content),
                        len(content.strip()),
                        content[:300],
                    )
                    raise ValueError("LLM returned invalid JSON for Mem0")
                return resp
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    time.sleep(1.5 * attempt)
                else:
                    logger.error("bad response (exhausted %d retries): %s", max_retries, e)
        raise last_err

    Completions.create = wrapped_create


# ---------------------------------------------------------------------------
# 3. 线程安全包装
# ---------------------------------------------------------------------------
def make_mem0_local_components_threadsafe(memory) -> threading.RLock:
    """串行化所有落盘 / 显存的本地组件。

    [FIX-C2] 原方案只包了 vector_store.{insert,search,...} 和
    embedding_model.embed, 漏掉了三个真正会并发炸掉的热点:
      * embedding_model.embed_batch —— mem0 V3 Phase 3 的主路径, 会让 16 个
        线程在单卡上并发跑 SentenceTransformer.encode。
      * memory.entity_store —— 懒加载出来的 *第二个* collection, 和主
        collection 共用同一个 embedded Qdrant client。mem0 自己的注释写明
        "local mode is not thread-safe", 且它内置的 _write_lock 只覆盖
        delete/update, 不覆盖 insert。Phase 7 实体链接每次 add 都会打它。
      * vector_store.keyword_search —— fastembed BM25 encoder。

    注意必须用 RLock 而不是 Lock:
      Qdrant.search_batch 失败时会 fallback 到 self.search(...), 于是出现
      search_batch -> search 的 *同线程嵌套* 加锁; 普通 Lock 会直接死锁。
    """
    lock = threading.RLock()

    def wrap(obj, method_names):
        if obj is None:
            return
        for name in method_names:
            fn = getattr(obj, name, None)
            if not callable(fn):
                continue
            if getattr(fn, "_mem0_lme_locked", False):
                continue

            @functools.wraps(fn)
            def _locked(*args, _fn=fn, **kwargs):
                with lock:
                    return _fn(*args, **kwargs)

            _locked._mem0_lme_locked = True
            setattr(obj, name, _locked)

    wrap(
        getattr(memory, "vector_store", None),
        [
            "insert",
            "search",
            "search_batch",
            "keyword_search",
            "update",
            "delete",
            "get",
            "list",
            "delete_col",
            "reset",
        ],
    )
    wrap(getattr(memory, "embedding_model", None), ["embed", "embed_batch"])

    # mem0 的 SQLiteManager 自带 threading.Lock, 包一层只是让临界区显式。
    wrap(
        getattr(memory, "db", None),
        [
            "add_history",
            "batch_add_history",
            "get_history",
            "save_messages",
            "get_last_messages",
            "reset",
        ],
    )

    # 必须在 worker 启动 *之前* 实体化懒加载的 entity_store,
    # 否则两个线程会同时创建它(两个 collection / 两个 client)。
    try:
        entity_store = memory.entity_store
    except Exception:
        entity_store = None
    wrap(
        entity_store,
        ["insert", "search", "search_batch", "update", "delete", "get", "list"],
    )

    return lock


# ---------------------------------------------------------------------------
# 4. bge.yaml query prefix 解析
# ---------------------------------------------------------------------------
def _deep_find_prefix(cfg) -> Optional[str]:
    try:
        items = list(cfg.items())
    except Exception:
        return None
    for k, v in items:
        if k in PREFIX_KEYS and isinstance(v, str) and v.strip():
            return v
        if OmegaConf.is_config(v):
            got = _deep_find_prefix(v)
            if got:
                return got
    return None


def resolve_bge_query_prefix(bge_yaml_path: str, cli_override: Optional[str]) -> Tuple[str, str]:
    """返回 (prefix, source)。

    [FIX-C4] 原方案在 key 找不到时 *静默* 回退到硬编码前缀, 这会让 query 编码
    和 SeCom / A-Mem / ThinkQE 静默错位。现在: 文件缺失或 key 缺失 = 硬错误,
    除非显式传 --query-prefix。
    """
    if cli_override is not None:
        return cli_override, "cli"

    p = Path(bge_yaml_path)
    if not p.exists():
        raise FileNotFoundError(
            f"bge.yaml 不存在: {p.resolve()}\n"
            "请从 SeCom 仓库拷贝共享检索配置 "
            "(experiments/baselines/external/_remote_secom_dropin/bge.yaml) "
            "到本脚本同级目录, 或用 --query-prefix 显式指定。\n"
            "注意: 静默使用默认前缀会导致与其它基线的 Recall 不可比。"
        )

    cfg = OmegaConf.load(p)
    for scope in PREFIX_SCOPES:
        for key in PREFIX_KEYS:
            try:
                val = (
                    OmegaConf.select(cfg, key)
                    if scope is None
                    else OmegaConf.select(cfg, f"{scope}.{key}")
                )
            except Exception:
                val = None
            if isinstance(val, str) and val.strip():
                src = key if scope is None else f"{scope}.{key}"
                return val, src

    val = _deep_find_prefix(cfg)
    if isinstance(val, str) and val.strip():
        return val, "deep-scan"

    raise KeyError(
        f"在 {p.resolve()} 中找不到 query 前缀配置。\n"
        f"已尝试 key={list(PREFIX_KEYS)} 以及 scope={list(PREFIX_SCOPES)}。\n"
        "请检查 bge.yaml 的实际字段名, 或用 --query-prefix 显式指定。"
    )


def get_git_commit(repo_dir: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def get_pkg_version(pkg: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    for name in (pkg, "mem0ai", "mem0"):
        try:
            return version(name)
        except PackageNotFoundError:
            continue
        except Exception:
            break
    return "unknown"


# ---------------------------------------------------------------------------
# 5. API 面自检 (避免再来一轮 API 漂移)
# ---------------------------------------------------------------------------
def api_check(memory, logger) -> Dict[str, Any]:
    """打印关键 API 的真实签名, 便于在正式跑之前确认版本。"""
    info: Dict[str, Any] = {}
    try:
        info["Memory.search"] = str(inspect.signature(type(memory).search))
    except Exception as e:
        info["Memory.search"] = f"<error {e}>"
    try:
        info["Memory.add"] = str(inspect.signature(type(memory).add))
    except Exception as e:
        info["Memory.add"] = f"<error {e}>"
    try:
        info["Memory.delete_all"] = str(inspect.signature(type(memory).delete_all))
    except Exception as e:
        info["Memory.delete_all"] = f"<error {e}>"
    try:
        info["vector_store.search"] = str(
            inspect.signature(type(memory.vector_store).search)
        )
    except Exception as e:
        info["vector_store.search"] = f"<error {e}>"
    try:
        info["embedder.embed"] = str(
            inspect.signature(type(memory.embedding_model).embed)
        )
    except Exception as e:
        info["embedder.embed"] = f"<error {e}>"
    info["embedding_dims"] = getattr(
        getattr(memory.embedding_model, "config", None), "embedding_dims", None
    )
    # [FIX-C7] 之前读的是 vector_store.config.embedding_model_dims (pydantic 配置对象,
    # 这里恒为 None), 会误导人以为维度没锁死。实际生效的是 Qdrant 实例上的
    # embedding_model_dims, 它决定了集合的向量维度。
    info["vector_store_dims"] = getattr(memory.vector_store, "embedding_model_dims", None)
    try:
        info["collection_vector_size"] = {
            c.name: memory.vector_store.client.get_collection(c.name)
            .config.params.vectors.size
            for c in memory.vector_store.client.get_collections().collections
        }
    except Exception as e:
        info["collection_vector_size"] = f"<error {e}>"
    for k, v in info.items():
        logger.info("api-check | %-22s = %s", k, v)
    return info


# ---------------------------------------------------------------------------
# 6. 单题处理
# ---------------------------------------------------------------------------
def clean_turns(turns) -> List[Dict[str, str]]:
    """重建 turn 为 {role, content} —— 顺带剔除 has_answer 等辅助字段。"""
    out: List[Dict[str, str]] = []
    for t in turns or []:
        if not isinstance(t, dict):
            continue
        content = t.get("content")
        if not content:
            continue
        out.append({"role": t.get("role") or "user", "content": content})
    return out


def write_sessions(memory, item, uid, logger) -> int:
    """按时间顺序串行写入每个 Session (infer=True, mem0 V3 additive 提炼)。"""
    sessions = item.get("haystack_sessions") or []
    session_ids = list(item.get("haystack_session_ids") or [])
    dates = list(item.get("haystack_dates") or [])

    n_memories = 0
    n_empty = 0
    for i, turns in enumerate(sessions):
        sid = session_ids[i] if i < len(session_ids) else f"sess_{i}"
        dt = dates[i] if i < len(dates) else ""
        msgs = clean_turns(turns)
        if not msgs:
            n_empty += 1
            continue
        res = memory.add(
            msgs,
            user_id=uid,
            metadata={"session_id": sid, "timestamp": dt},
            infer=True,
        )
        results = (res or {}).get("results") or []
        n_memories += len(results)
        if not results:
            logger.debug("session %s produced 0 memories", sid)
    logger.debug(
        "uid=%s wrote %d sessions (%d empty) -> %d memories",
        uid,
        len(sessions) - n_empty,
        n_empty,
        n_memories,
    )
    return n_memories


def dense_retrieve(memory, question, uid, query_prefix, top_k):
    """[FIX-C1][FIX-B1] pure dense BGE 检索。

    不用 mem0 的 Memory.search():
      1. 现在的签名是 search(query, *, top_k, filters, ...), 顶层 user_id 会被
         _reject_top_level_entity_params() 直接 raise ValueError;
      2. 更重要的是它的 _search_vector_store 是 *混合* 检索
         (dense + fastembed BM25 稀疏向量 + entity boost), 还带一个
         threshold=0.1 的硬过滤 —— 和 SeCom / A-Mem / ThinkQE 那一行的口径
         完全不同, 数字没法同表比较。

    这里完全对齐其它基线: 自己加 BGE query 前缀 -> embed -> 向量库取原始
    cosine 邻居 -> 不做阈值过滤。
    """
    q = f"{query_prefix}{question}" if query_prefix else question
    vec = memory.embedding_model.embed(q, "search")
    hits = memory.vector_store.search(
        query=q, vectors=vec, top_k=top_k, filters={"user_id": uid}
    )
    return hits


def native_retrieve(memory, question, uid, query_prefix, top_k):
    """mem0 原生混合检索 (仅作对照用)。

    [FIX-B1] 作用域必须走 filters=, 条数必须走 top_k=。
    """
    q = f"{query_prefix}{question}" if query_prefix else question
    res = memory.search(q, filters={"user_id": uid}, top_k=top_k)
    items = res.get("results", res) if isinstance(res, dict) else res
    ranked: List[str] = []
    seen = set()
    for it in items or []:
        meta = it.get("metadata") or {}
        sid = meta.get("session_id")
        if sid and sid not in seen:
            seen.add(sid)
            ranked.append(sid)
    return ranked


def hits_to_sessions(hits, topk) -> List[str]:
    """按得分顺序去重 session_id 并截断到 topk。"""
    ranked: List[str] = []
    seen = set()
    for h in hits or []:
        payload = getattr(h, "payload", None)
        if not isinstance(payload, dict):
            payload = {}
        sid = payload.get("session_id")
        if not sid or sid in seen:
            continue
        seen.add(sid)
        ranked.append(sid)
        if len(ranked) >= topk:
            break
    return ranked


def process_one_question(
    memory,
    item: Dict[str, Any],
    topk: int,
    query_prefix: str,
    oversample: int,
    retrieval: str,
    logger: logging.Logger,
    cleanup_after: bool = False,
    ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    qid = str(item["question_id"])
    uid = f"lme_{qid}"

    use_lme = bool(ctx and ctx.get("profile") == "lme")

    if use_lme:
        # [LME] 适配版摄入端: LLM 提炼保留, 但上下文无关 + 传真实 observation
        # date + LongMemEval 专用约束 + 按 session 内容寻址的提炼缓存。
        import lme_profile

        # 幂等: 清掉上次(可能被中断的)残留记忆
        lme_profile.fast_delete_user(memory, uid)
        n_memories = lme_profile.write_sessions_lme(
            memory,
            item,
            uid,
            logger,
            ctx.get("cache"),
            max_facts=int(ctx.get("max_facts") or 0),
            key_mode=str(ctx.get("cache_key") or "auto"),
            date_mode=str(ctx.get("date_mode") or "resolve"),
            prompt_mode=str(ctx.get("prompt_mode") or "lme"),
            fact_style=str(ctx.get("fact_style") or "rich"),
            batch_sessions=int(ctx.get("batch_sessions") or 1),
            prefilter_topk=int(ctx.get("prefilter_topk") or 0),
            query_prefix=str(ctx.get("query_prefix") or ""),
            session_level=bool(ctx.get("session_level", False)),
        )
    else:
        # 幂等: 清掉上次(可能被中断的)残留记忆
        try:
            memory.delete_all(user_id=uid)
        except Exception:
            pass
        n_memories = write_sessions(memory, item, uid, logger)

    if retrieval == "dense":
        hits = dense_retrieve(memory, item["question"], uid, query_prefix, oversample)
        ranked = hits_to_sessions(hits, topk)
    else:
        ranked = native_retrieve(memory, item["question"], uid, query_prefix, oversample)

    result = {
        "question_id": qid,
        "question_type": item.get("question_type", ""),
        "answer_session_ids": item.get("answer_session_ids", []),
        "retrieved_sessions": ranked,
        "ranked_session_ids": ranked,
        "n_sessions": len(item.get("haystack_sessions") or []),
        "n_memories_added": n_memories,
        "status": "ok",
    }

    # [M-run] M 每道题约 500 个 session，若不在题后清理，Qdrant 会累积到
    # 数百万点 (local mode 会越来越慢)。预测已落盘，删掉该 user 的记忆不影响
    # resume；下次重跑同一题时本来也会先 delete_all。
    if cleanup_after:
        if use_lme:
            import lme_profile

            lme_profile.fast_delete_user(memory, uid)
        else:
            try:
                memory.delete_all(user_id=uid)
            except Exception as e:
                logger.warning("cleanup uid=%s failed: %s", uid, e)

    return result


def run_question_with_retry(
    memory,
    item,
    topk,
    query_prefix,
    oversample,
    retrieval,
    logger,
    max_attempts: int,
    cleanup_after: bool = False,
    ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """[FIX-C3] 题级重试 + 失败也落一条记录。

    原方案里 memory.add 抛 LLMError 时, 该题会被整个丢掉, predictions.jsonl
    永远凑不满 470 行, 而且没有任何重试。现在: 题级重试; 重试耗尽后写一条
    status="failed" 的记录, 保证文件行数恒等于题数, 且失败可审计。
    """
    qid = str(item["question_id"])
    last_err: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return process_one_question(
                memory,
                item,
                topk,
                query_prefix,
                oversample,
                retrieval,
                logger,
                cleanup_after=cleanup_after,
                ctx=ctx,
            )
        except Exception as e:
            last_err = e
            logger.warning(
                "question %s failed (attempt %d/%d): %s",
                qid,
                attempt,
                max_attempts,
                e,
            )
            if attempt < max_attempts:
                # [ROBUST] 网络类错误要等久一点再重试: 断网时 2s/4s 的退避
                # 会把 3 次机会在 10 秒内烧光, 然后整题被判 failed。
                msg = str(e).lower()
                is_conn = any(
                    k in msg
                    for k in ("connection", "timeout", "temporarily unavailable", "network")
                )
                time.sleep((30.0 if is_conn else 2.0) * attempt)

    logger.error("question %s permanently failed: %r", qid, last_err)
    return {
        "question_id": qid,
        "question_type": item.get("question_type", ""),
        "answer_session_ids": item.get("answer_session_ids", []),
        "retrieved_sessions": [],
        "ranked_session_ids": [],
        "n_sessions": len(item.get("haystack_sessions") or []),
        "n_memories_added": 0,
        "status": "failed",
        "error": repr(last_err),
    }


# ---------------------------------------------------------------------------
# 7. 多档 K 评估
# ---------------------------------------------------------------------------
def evaluate_and_save(
    preds: List[Dict[str, Any]],
    topk: int,
    retrieval: str,
    out_dir: Path,
    logger: logging.Logger,
):
    ks = sorted({k for k in (1, 3, 5, 10, 20) if k <= topk} | {topk})

    def _calc_group(items: List[Dict[str, Any]], only_ok: bool = False) -> Dict[str, float]:
        # [FIX-C5b] 分母口径修正:
        #   原方案 n = len(items), 但空 gold 的题在分子里被 continue 掉了 ->
        #   系统性低估 recall。
        #   现在分母 = "有 gold session 的题"。失败的题(retrieved 为空)按 0 分计入,
        #   这样分母恒等于 470, 与其它基线那一行可比; 同时额外输出一个
        #   *_excluding_failed 变体, 方便看"跑通率"对分数的影响。
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
        "retrieval": retrieval,
        "ks": ks,
        "overall": _calc_group(preds),
        "overall_excluding_failed": _calc_group(preds, only_ok=True),
        "by_question_type": {
            qt: _calc_group(items) for qt, items in sorted(by_type.items())
        },
    }
    (out_dir / "aggregate.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Saved multi-K metrics (%s) to %s", ks, out_dir / "aggregate.json")
    return agg


# ---------------------------------------------------------------------------
# 8. main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="longmemeval_s_cleaned.json")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--skip-abs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--model", type=str, default=os.getenv("MEM0_LLM_MODEL", "deepseek-flash"))
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument(
        "--llm-connect-timeout",
        type=float,
        default=20.0,
        help="connect 超时秒数; 断网时快速失败, 交给题级重试去等网络",
    )
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument(
        "--llm-thinking",
        choices=["disabled", "enabled", "omit"],
        default="disabled",
        help="DeepSeek thinking 模式; 官方默认 enabled, 评测一律关掉 (disabled)",
    )
    parser.add_argument(
        "--llm-reasoning-effort",
        type=str,
        default=None,
        help="DeepSeek OpenAI 格式只接受 low/high/max; 默认不发该参数",
    )
    parser.add_argument("--question-retries", type=int, default=3)
    parser.add_argument(
        "--repair-passes",
        type=int,
        default=3,
        help="收尾前补跑 failed 题的轮数 (0 = 不补跑; 注意 aggregate.json 是调度器的完成标志)",
    )
    parser.add_argument(
        "--repair-wait",
        type=float,
        default=60.0,
        help="每轮补跑前的等待秒数 (给断网/限流留恢复时间)",
    )
    parser.add_argument("--bge-yaml", type=str, default="bge.yaml")
    parser.add_argument("--query-prefix", type=str, default=None)
    parser.add_argument("--embedder", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--embedding-dims", type=int, default=768)
    parser.add_argument("--collection", type=str, default="mem0_lme_bge_base")
    parser.add_argument(
        "--retrieval",
        choices=["dense", "native"],
        default="dense",
        help="dense = pure BGE cosine (同 SeCom/A-Mem/ThinkQE 口径); native = mem0 混合检索",
    )
    parser.add_argument(
        "--oversample",
        type=int,
        default=0,
        help="检索过采样条数; 0 = auto (max(topk*10, 200))",
    )
    parser.add_argument(
        "--delete-after-question",
        action="store_true",
        help="每题检索完成后删掉该 user 的记忆; 适合 M 这种 500 sessions/题的大 run",
    )
    # ---------------- [LME] 适配版摄入端 ----------------
    parser.add_argument(
        "--profile",
        choices=["stock", "lme"],
        default="stock",
        help="stock = 原生 mem0 V3 add(); lme = LongMemEval 适配版摄入端(LLM 提炼保留)",
    )
    parser.add_argument(
        "--extract-cache",
        type=str,
        default=None,
        help="lme profile 的提炼缓存 (sqlite); 同一 session 内容只调一次 LLM",
    )
    parser.add_argument(
        "--lme-max-facts",
        type=int,
        default=0,
        help="lme profile: 每个 session 最多保留多少条记忆 (0 = 不限制)",
    )
    parser.add_argument(
        "--lme-cache-key",
        choices=["auto", "content", "content_date"],
        default="auto",
        help="缓存键: auto = 含相对时间的 session 带日期, 否则只按内容",
    )
    parser.add_argument(
        "--lme-date-mode",
        choices=["resolve", "verbatim"],
        default="resolve",
        help="resolve = 相对时间按 observation date 解析成绝对日期; verbatim = 保留原始相对表述",
    )
    parser.add_argument(
        "--lme-prompt",
        choices=["lme", "stock"],
        default="lme",
        help="lme profile 的 system prompt: lme = 收紧约束版; stock = 复用 mem0 原 prompt",
    )
    parser.add_argument(
        "--lme-fact-style",
        choices=["rich", "short"],
        default="rich",
        help="short = 每条 fact 更短 (5-25 词), output token 少 -> 更快更便宜",
    )
    parser.add_argument(
        "--lme-batch-sessions",
        type=int,
        default=1,
        help="一次 LLM 调用提炼几个 session (两级缓存: 命中仍按 session 计)",
    )
    parser.add_argument(
        "--lme-prefilter-topk",
        type=int,
        default=0,
        help="⚠️ query-aware 粗筛: 只对 BGE 命中的 top-k 个 session 做 LLM 提炼。"
        "0 = 关闭。开启会改变方法性质 (retrieve-then-extract), 不能算原版 mem0。",
    )
    parser.add_argument(
        "--session-level",
        action="store_true",
        help="session 级收口: LLM 提炼不变, 但每个 session 合成一条记忆、只落一个向量,"
        " 检索直接返回 session (对齐 BGE/ThinkQE/SeCom 等 session 级检索口径)。",
    )
    parser.add_argument("--mem0-repo", type=str, default="mem0")
    parser.add_argument(
        "--telemetry",
        choices=["off", "on"],
        default="off",
        help="mem0 PostHog telemetry; 评测一律关掉 (off)",
    )
    parser.add_argument("--out-root", type=str, default="results/mem0_bge_s")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    os.environ["MEM0_TELEMETRY"] = "True" if args.telemetry == "on" else "False"

    if args.resume:
        run_dir = Path(args.resume)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(args.out_root) / f"mem0_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir / "run.log")
    patch_openai_for_mem0(
        logger,
        max_retries=args.llm_retries,
        max_tokens=args.max_tokens,
        timeout=args.llm_timeout,
        thinking=args.llm_thinking,
        reasoning_effort=args.llm_reasoning_effort,
        connect_timeout_s=args.llm_connect_timeout,
    )

    from mem0 import Memory

    query_prefix, prefix_source = resolve_bge_query_prefix(args.bge_yaml, args.query_prefix)
    logger.info("BGE query prefix: %r (source=%s)", query_prefix, prefix_source)

    # 双处锁死 768 维 (BAAI/bge-base-en-v1.5)
    mem0_config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": args.model,
                "temperature": 0.0,
                # [FIX-B2] mem0 总是把 config.max_tokens 发出去, 默认 2000 不够。
                "max_tokens": args.max_tokens,
                "top_p": 0.1,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": args.embedder,
                "embedding_dims": args.embedding_dims,
                "model_kwargs": {"device": "cuda"},
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": args.collection,
                "embedding_model_dims": args.embedding_dims,
                "path": str(run_dir / "qdrant_storage"),
                "on_disk": False,
            },
        },
        "history_db_path": str(run_dir / "history.db"),
    }

    memory = Memory.from_config(mem0_config)
    make_mem0_local_components_threadsafe(memory)
    api_info = api_check(memory, logger)

    # [LME] 适配版摄入端的上下文 (提炼缓存等)
    ctx: Optional[Dict[str, Any]] = None
    if args.profile == "lme":
        import lme_profile

        cache = lme_profile.ExtractionCache(args.extract_cache) if args.extract_cache else None
        ctx = {
            "profile": "lme",
            "cache": cache,
            "max_facts": args.lme_max_facts,
            "cache_key": args.lme_cache_key,
            "date_mode": args.lme_date_mode,
            "prompt_mode": args.lme_prompt,
            "fact_style": args.lme_fact_style,
            "batch_sessions": args.lme_batch_sessions,
            "prefilter_topk": args.lme_prefilter_topk,
            "query_prefix": query_prefix,
            "session_level": bool(args.session_level),
        }
        logger.info(
            "[LME] profile=lme cache=%s max_facts=%d cache_key=%s date_mode=%s prompt=%s",
            args.extract_cache or "<none>",
            args.lme_max_facts,
            args.lme_cache_key,
            args.lme_date_mode,
            args.lme_prompt,
        )
        # 检索是 pure dense BGE, BM25 稀疏向量用不上 —— 关掉能省掉每次 insert
        # 的 fastembed 编码 (M 集每题几千条, 逐条 insert 时这部分很贵)。
        try:
            memory.vector_store._has_bm25_slot = False
        except Exception:
            pass

    # 加载与过滤数据
    dataset = json.load(open(args.data, encoding="utf-8"))
    if args.subset:
        sub_ids = set(json.load(open(args.subset, encoding="utf-8")))
        dataset = [x for x in dataset if str(x["question_id"]) in sub_ids]
    if args.skip_abs:
        dataset = [x for x in dataset if not str(x["question_id"]).endswith("_abs")]
    if args.limit > 0:
        dataset = dataset[: args.limit]

    n_planned = len(dataset)
    oversample = args.oversample or max(args.topk * 10, 200)
    meta = {
        "method": "mem0" if args.profile == "stock" else "mem0-lme",
        "mem0_commit": get_git_commit(args.mem0_repo),
        "mem0_version": get_pkg_version("mem0"),
        "data": args.data,
        "subset": args.subset,
        "topk": args.topk,
        "skip_abs": bool(args.skip_abs),
        "n_planned": n_planned,
        "workers": args.workers,
        "llm_model": args.model,
        "llm_max_tokens": args.max_tokens,
        "llm_thinking": args.llm_thinking,
        "llm_reasoning_effort": args.llm_reasoning_effort,
        "llm_base_url": os.getenv("OPENAI_BASE_URL"),
        "embedder": args.embedder,
        "embedding_dims": args.embedding_dims,
        "query_prefix": query_prefix,
        "query_prefix_source": prefix_source,
        "retrieval": args.retrieval,
        "oversample": oversample,
        "delete_after_question": bool(args.delete_after_question),
        "profile": args.profile,
        "extract_cache": args.extract_cache,
        "lme_max_facts": args.lme_max_facts,
        "lme_cache_key": args.lme_cache_key,
        "lme_date_mode": args.lme_date_mode,
        "lme_prompt": args.lme_prompt,
        "lme_fact_style": args.lme_fact_style,
        "lme_batch_sessions": args.lme_batch_sessions,
        "lme_prefilter_topk": args.lme_prefilter_topk,
        "session_level": bool(args.session_level),
        "question_retries": args.question_retries,
        "api_check": api_info,
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 读取已完成预测 (支持 --resume)
    pred_path = run_dir / "predictions.jsonl"
    completed: Dict[str, Dict[str, Any]] = {}
    if pred_path.exists():
        with open(pred_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 崩溃时可能留下半行
                completed[str(rec["question_id"])] = rec

    # [FIX-C3] 之前失败的题在 resume 时重跑
    todo = [
        x
        for x in dataset
        if str(x["question_id"]) not in completed
        or completed[str(x["question_id"])].get("status") != "ok"
    ]
    already_done = n_planned - len(todo)
    logger.info(
        "Run dir: %s | planned=%d | already_done=%d | todo=%d | workers=%d | "
        "retrieval=%s | oversample=%d",
        run_dir,
        n_planned,
        already_done,
        len(todo),
        args.workers,
        args.retrieval,
        oversample,
    )

    write_lock = threading.Lock()
    # 崩溃可能留下没有换行符的半行, 追加前先补一个换行, 避免和旧内容粘成一行
    if pred_path.exists() and pred_path.stat().st_size > 0:
        with open(pred_path, "rb") as rf:
            rf.seek(-1, 2)
            if rf.read(1) != b"\n":
                with open(pred_path, "a", encoding="utf-8") as nf:
                    nf.write("\n")
    with open(pred_path, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            fut_to_qid = {
                pool.submit(
                    run_question_with_retry,
                    memory,
                    item,
                    args.topk,
                    query_prefix,
                    oversample,
                    args.retrieval,
                    logger,
                    args.question_retries,
                    args.delete_after_question,
                    ctx,
                ): str(item["question_id"])
                for item in todo
            }
            for fut in as_completed(fut_to_qid):
                qid = fut_to_qid[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    logger.exception("Failed qid=%s: %s", qid, e)
                    continue
                with write_lock:
                    completed[qid] = rec
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                logger.info(
                    "[%d/%d] Done qid=%s status=%s top3=%s",
                    len(completed),
                    n_planned,
                    qid,
                    rec.get("status"),
                    rec.get("retrieved_sessions", [])[:3],
                )

    # [ROBUST] 修复轮: 上面这轮里如果因为断网/休眠唤醒等原因有题被判 failed,
    # 在写 aggregate.json 之前再补跑几轮。
    # 关键点: 调度器把 aggregate.json 当作"跑完"的标志, 如果带着 failed 题就写
    # aggregate.json, 调度器会直接退出, 这个 run 就带着一堆 failed 结束了。
    # 所以这里必须把 failed 题补回来(或者明确放弃)才落 aggregate.json。
    repair_passes = max(0, int(args.repair_passes))
    for rpass in range(1, repair_passes + 1):
        broken = [
            x
            for x in dataset
            if str(x["question_id"]) in completed
            and completed[str(x["question_id"])].get("status") != "ok"
        ]
        if not broken:
            break
        logger.warning(
            "[repair %d/%d] %d questions failed, retrying: %s",
            rpass,
            repair_passes,
            len(broken),
            [str(x["question_id"]) for x in broken][:10],
        )
        time.sleep(args.repair_wait)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            fut_to_qid = {
                pool.submit(
                    run_question_with_retry,
                    memory,
                    item,
                    args.topk,
                    query_prefix,
                    oversample,
                    args.retrieval,
                    logger,
                    args.question_retries,
                    args.delete_after_question,
                    ctx,
                ): str(item["question_id"])
                for item in broken
            }
            for fut in as_completed(fut_to_qid):
                qid = fut_to_qid[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    logger.exception("[repair] Failed qid=%s: %s", qid, e)
                    continue
                with write_lock:
                    completed[qid] = rec
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                logger.info(
                    "[repair %d/%d] qid=%s status=%s",
                    rpass,
                    repair_passes,
                    qid,
                    rec.get("status"),
                )

    # [FIX-C3] 收尾重写一遍 predictions.jsonl:
    # 保证文件行数 == 题数 (resume 重跑不会产生重复行), 且顺序与数据集一致。
    ordered_preds = [
        completed[str(x["question_id"])]
        for x in dataset
        if str(x["question_id"]) in completed
    ]
    with open(pred_path, "w", encoding="utf-8") as f:
        for rec in ordered_preds:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    agg = evaluate_and_save(ordered_preds, args.topk, args.retrieval, run_dir, logger)
    usage = llm_usage_summary()
    if ctx and ctx.get("cache") is not None:
        cache_stats = ctx["cache"].stats()
        usage["extract_cache"] = cache_stats
        logger.info(
            "extract cache: hits=%d misses=%d (hit rate %.1f%%)",
            cache_stats["hits"],
            cache_stats["misses"],
            100.0
            * cache_stats["hits"]
            / max(cache_stats["hits"] + cache_stats["misses"], 1),
        )
    (run_dir / "llm_usage.json").write_text(
        json.dumps(usage, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    agg["llm_usage"] = usage
    (run_dir / "aggregate.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "LLM usage: calls=%d retried=%d prompt_tok=%d cache_hit=%d (%.1f%%) "
        "completion_tok=%d | est_cost off_peak=$%.2f peak=$%.2f",
        usage["calls"],
        usage["retried_calls"],
        usage["prompt_tokens"],
        usage["cache_hit_tokens"],
        usage["cache_hit_ratio"] * 100,
        usage["completion_tokens"],
        usage["est_cost_usd"]["off_peak"],
        usage["est_cost_usd"]["peak"],
    )
    logger.info(
        "Finished. n_total=%d n_ok=%d n_failed=%d n_with_gold=%d",
        agg["n_total"],
        agg["n_ok"],
        agg["n_failed"],
        agg["n_with_gold"],
    )
    logger.info("overall: %s", json.dumps(agg["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
