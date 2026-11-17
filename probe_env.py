#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跑正式评测前的环境预检 (Windows / Linux 通用)。

检查项:
  1. 依赖包是否装齐 (mem0 / torch / sentence-transformers / qdrant-client ...)
  2. CUDA 是否可用, GPU 型号与显存
  3. BGE 模型能否加载 (首次会下载 ~440MB)
  4. 代理是否可达, mem0 的请求形态 (temperature/top_p/max_tokens/json_object/
     thinking-disabled) 是否被 deepseek-v4-free 接受, 能否返回合法 JSON
  5. 数据集文件是否就位, 题量是否符合预期 (500 / 470)
  6. bge.yaml query 前缀能否解析 (拿不到就告诉你用 --query-prefix)

用法:  python probe_env.py
"""

import io
import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OK, BAD, WARN = [], [], []


def _cuda_ok() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def ok(m):
    OK.append(m)
    print(f"  [ OK ] {m}")


def bad(m):
    BAD.append(m)
    print(f"  [FAIL] {m}")


def warn(m):
    WARN.append(m)
    print(f"  [WARN] {m}")


print("=" * 72)
print("1) 依赖包")
print("=" * 72)
for mod, pkg in [
    ("mem0", "mem0ai"),
    ("torch", "torch"),
    ("sentence_transformers", "sentence-transformers"),
    ("qdrant_client", "qdrant-client"),
    ("openai", "openai"),
    ("dotenv", "python-dotenv"),
    ("omegaconf", "omegaconf"),
    ("huggingface_hub", "huggingface_hub"),
]:
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", "?")
        ok(f"{mod:<22} {v}")
    except Exception as e:
        bad(f"{mod:<22} 未安装 ({e.__class__.__name__}) -> pip install {pkg}")

if BAD:
    print("\n依赖不全, 先装齐再继续。")

print()
print("=" * 72)
print("2) GPU / CUDA")
print("=" * 72)
try:
    import torch

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        ok(
            f"CUDA 可用 | {props.name} | {props.total_memory / 1024**3:.1f} GiB | "
            f"torch {torch.__version__} | cuda {torch.version.cuda}"
        )
    else:
        bad(f"CUDA 不可用 -> BGE 会跑在 CPU 上 (慢很多)。torch {torch.__version__}")
except Exception as e:
    bad(f"torch 不可用: {e}")

print()
print("=" * 72)
print("3) BGE 模型加载")
print("=" * 72)
try:
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda" if _cuda_ok() else "cpu")
    dim = m.get_sentence_embedding_dimension()
    v = m.encode("hello", convert_to_numpy=True)
    if dim == 768 and len(v) == 768:
        ok(f"BAAI/bge-base-en-v1.5 加载成功, dim={dim} (与 768 对齐)")
    else:
        bad(f"BGE 维度异常: model={dim} vector={len(v)} (必须是 768)")
except Exception as e:
    bad(f"BGE 加载失败: {e}")

print()
print("=" * 72)
print("4) LLM 代理 (mem0 的请求形态)")
print("=" * 72)
# 加载 env 文件 (与 runner 同一套优先级)
try:
    from dotenv import load_dotenv

    for cand in [
        Path.home() / "dot_env" / "openai.env",
        Path("/workspace/dot_env/openai.env"),
        Path(".env"),
    ]:
        if cand.exists():
            load_dotenv(cand, override=False)
            ok(f"env 文件: {cand}")
except Exception as e:
    warn(f"dotenv: {e}")

if os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.environ["OPENAI_API_BASE"]

base = os.getenv("OPENAI_BASE_URL")
key = os.getenv("OPENAI_API_KEY")
model = os.getenv("MEM0_LLM_MODEL", "deepseek-flash")
thinking = os.getenv("MEM0_LLM_THINKING", "disabled")

if not base or not key:
    bad(f"环境变量缺失 (OPENAI_BASE_URL={base!r} OPENAI_API_KEY={'set' if key else 'MISSING'})")
else:
    ok(f"base_url={base}  model={model}  thinking={thinking}")

    # 陷阱: mem0 的 OpenAILLM 只要看到 OPENROUTER_API_KEY 就会改走 OpenRouter,
    # 不管 OPENAI_BASE_URL 是什么。这会让所有请求打到错误的端点。
    if os.getenv("OPENROUTER_API_KEY"):
        bad(
            "环境里存在 OPENROUTER_API_KEY -> mem0 会强制走 OpenRouter 而不是 DeepSeek! "
            "请先 `set OPENROUTER_API_KEY=` 清除它。"
        )
    else:
        ok("无 OPENROUTER_API_KEY (mem0 不会误走 OpenRouter)")

    try:
        import openai

        client = openai.OpenAI(api_key=key, base_url=base, timeout=60.0)

        # --- 探针 A: mem0 的真实请求形态 (temperature/top_p/max_tokens/json_object/thinking) ---
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": 'Return a JSON object with key "memory".'},
                {"role": "user", "content": 'Extract the fact and reply with {"memory": ["x"]}'},
            ],
            temperature=0.0,
            top_p=0.1,
            max_tokens=8192,
            response_format={"type": "json_object"},
        )
        if thinking != "omit":
            kwargs["extra_body"] = {"thinking": {"type": thinking}}
        # 注意: 不发 reasoning_effort —— DeepSeek 官方 OpenAI 格式只接受
        # low/high/max, 发 "none" 会被 400。

        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        content = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning_content", None)
        fr = choice.finish_reason

        if reasoning:
            warn(f"返回里带 reasoning_content -> thinking 可能没关掉 (len={len(reasoning)})")
        else:
            ok("无 reasoning_content -> thinking 已关闭")

        try:
            json.loads(content)
            ok(f"JSON 模式正常, finish_reason={fr}: {content[:120]}")
        except Exception:
            if content.strip() == "":
                bad(
                    "content 空白但 finish_reason=stop —— 这是 DeepSeek JSON 模式的已知行为"
                    "(无法识别为 JSON 时内容被替换成空格)。runner 的重试逻辑能兜住, "
                    "但说明提示词/模型对该任务不稳定, 建议先看冒烟日志。"
                )
            else:
                warn(f"返回不是合法 JSON, finish_reason={fr}: {content[:200]!r}")

        # --- 探针 B: 验证 400 边界 —— 故意传非法的 reasoning_effort, 确认端点行为 ---
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                reasoning_effort="none",
                max_tokens=16,
            )
            warn("端点接受了 reasoning_effort='none' (与文档不符, 但可用)")
        except Exception as e:
            ok(
                f"端点拒绝 reasoning_effort='none' -> {e.__class__.__name__}: "
                f"{str(e)[:160]}  (已确认 runner 不再发该参数)"
            )

        # --- 探针 C: 上下文缓存是否生效 (成本的关键变量) ---
        # mem0 的提炼 system prompt 固定 ~8.2k token, 占每次 input 的 90%。
        # DeepSeek 的 Context Caching 默认开启, 这里用真实的 system prompt
        # 连发两次, 看第二次的 prompt_cache_hit_tokens。
        try:
            from mem0.configs.prompts import ADDITIVE_EXTRACTION_PROMPT

            sysmsg = {"role": "system", "content": ADDITIVE_EXTRACTION_PROMPT}
            usrmsg = {"role": "user", "content": 'Reply with {"memory": []}'}
            hits = []
            for i in range(2):
                r = client.chat.completions.create(
                    model=model,
                    messages=[sysmsg, usrmsg],
                    max_tokens=64,
                    **({"extra_body": {"thinking": {"type": thinking}}} if thinking != "omit" else {}),
                )
                u = getattr(r, "usage", None)
                hits.append(
                    (
                        int(getattr(u, "prompt_tokens", 0) or 0),
                        int(getattr(u, "prompt_cache_hit_tokens", 0) or 0),
                    )
                )
            (p1, h1), (p2, h2) = hits
            ok(f"缓存探针: 第1次 prompt={p1} hit={h1} | 第2次 prompt={p2} hit={h2}")
            if h2 > 0:
                ok(f"上下文缓存生效 (第2次命中 {h2} tok) -> 成本按 cache-hit 价计")
            else:
                warn("第2次没命中缓存 -> 成本会按全量 cache-miss 价计, 约为命中时的 50 倍")
        except Exception as e:
            warn(f"缓存探针失败 (不影响主流程): {e.__class__.__name__}: {str(e)[:200]}")
    except Exception as e:
        bad(f"代理调用失败: {e.__class__.__name__}: {e}")

print()
print("=" * 72)
print("5) 数据集")
print("=" * 72)
ds = Path("longmemeval_s_cleaned.json")
if ds.exists():
    try:
        data = json.load(open(ds, encoding="utf-8"))
        non_abs = [x for x in data if not str(x["question_id"]).endswith("_abs")]
        ok(f"{ds.name}: total={len(data)}  non_abs={len(non_abs)}")
        if len(data) == 500 and len(non_abs) == 470:
            ok("题量符合预期 (500 / 470)")
        else:
            warn(f"题量与预期(500/470)不符, 请核对数据集版本")
        s0 = non_abs[0]
        missing = [
            k
            for k in (
                "question_id",
                "question",
                "question_type",
                "answer_session_ids",
                "haystack_sessions",
                "haystack_session_ids",
            )
            if k not in s0
        ]
        if missing:
            bad(f"样本缺少字段: {missing}")
        else:
            ok(
                f"字段齐全 | sample qid={s0['question_id']} "
                f"type={s0.get('question_type')} "
                f"sessions={len(s0['haystack_sessions'])} "
                f"gold={s0.get('answer_session_ids')}"
            )
    except Exception as e:
        bad(f"数据集解析失败: {e}")
else:
    bad(f"找不到 {ds.name} (当前目录: {Path.cwd()})")

print()
print("=" * 72)
print("6) bge.yaml query 前缀")
print("=" * 72)
try:
    import run_mem0_lme as R

    prefix, source = R.resolve_bge_query_prefix("bge.yaml", None)
    ok(f"bge.yaml 解析成功: {prefix!r} (source={source})")
except FileNotFoundError as e:
    warn(f"{e}")
    print("         -> 没有 bge.yaml 也能跑, 用 --query-prefix 指定:")
    print('            --query-prefix "Represent this sentence for searching relevant passages: "')
except KeyError as e:
    bad(f"bge.yaml 存在但读不到 query 前缀:\n{e}")
except Exception as e:
    warn(f"bge.yaml 检查异常: {e}")

print()
print("=" * 72)
print(f"汇总: OK={len(OK)}  FAIL={len(BAD)}  WARN={len(WARN)}")
print("=" * 72)
if BAD:
    print("有 FAIL 项, 先修掉再跑正式评测。")
    raise SystemExit(1)
if WARN:
    print("有 WARN 项, 确认无误后再跑。")
print("可以开跑了。")
