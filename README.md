# LongMemEval 记忆实验（正版跑实验逻辑）

本仓库只放 **LongMemEval S/M 检索基线的一线运行代码 + 极小逐题结果**，不含数据、向量库和提炼缓存。
重型产物（>100MB）留在跑实验的设备本地，GitHub 不放。

## 两个检索口径（务必分清）

| 口径 | 标志 | 说明 | 可否与主表同列 |
|---|---|---|---|
| fact 级（原版） | 默认 | 每个 session 提炼多条 fact，每条 fact 一个向量，检索后映射回 session（≈fact 级 max-pool） | 否，分数偏高，与 BGE/ThinkQE/SeCom 不可比 |
| session 级 | `--session-level` | LLM 提炼不变，但每个 session 合成一条记忆、只落一个向量，检索直接返回 session | **是，这是正版收口口径** |

两者都用 `--retrieval dense`（pure BGE cosine，不用 mem0 原生混合检索）。

## 目录

| 路径 | 内容 |
|---|---|
| `run_mem0_lme.py` | mem0 主 runner（`--profile lme`/`stock`，可 `--resume`） |
| `lme_profile.py` | lme profile 的 prompt / fact 提炼 / session 级收口 |
| `longmemeval/eval_retrieval_official.py` | 官方口径复算（recall_all / ndcg，跳过 abstention） |
| `raw_bge_eval.py` | 纯 BGE 检索基线 |
| `Secom/区/secom_lme/` | SeCom + LLMLingua-2 基线 |
| `schedule_runner.py` | DeepSeek 低峰调度（有 LLM 的 full run 才需要） |
| `mem0_results/` | 已发布的 mem0 逐题结果（只有 session id） |
| `subsets/` | smoke/val 小规模子集 |
| `make_subset.py` | 按题型比例抽 question-id 子集 |

## 本地必须有、但 GitHub 不传的东西

| 本地文件 | 作用 | GitHub 是否有 |
|---|---|---|
| `longmemeval_s_cleaned.json` / `longmemeval_m_cleaned.json` | 数据 | 否（太大，不传） |
| `mem0/` | mem0ai 2.2.1 源码 checkout | 否（上游仓库，不 vendored） |
| `results/mem0_*/`（Qdrant store、history.db） | 运行目录，>100MB | 否 |
| `results/extract_cache/cache_resolve_auto.sqlite` | mem0 提炼缓存 | **可恢复**：facts-only 5MB 在仓库，全量带向量版在 release `mem0-s-cache-v1` |

跑 S 前先恢复提炼缓存（跳过 2067 次 LLM）：

```bash
python mem0_results/restore_extract_cache.py --out results/extract_cache/cache_resolve_auto.sqlite
```

`.gitignore` 已挡掉数据/向量库/原始缓存；**不要 `git add` 大文件。**

## 环境

- Python 3.12 venv（原设备 `.venv_mem0`）：`mem0ai==2.2.1`（commit `f1791da`）、`qdrant-client`、`sentence-transformers` + `BAAI/bge-base-en-v1.5`。
- 官方 DeepSeek OpenAI 接口，模型 `deepseek-flash`，关闭 thinking。

## 运行命令（在含数据的设备上执行）

```bat
set "OPENAI_API_KEY=sk-<你的key>"
set "OPENAI_API_BASE=https://api.deepseek.com"
set "OPENAI_BASE_URL=https://api.deepseek.com"
```

### 1) S-150 session 级（先小规模验证，0 LLM）

先准备 150 子集（用你锁定的配对 150，`subsets/s_150_locked.json`，已入库）：

- `subsets/s_150_locked.json`：论文 150 附表锁定的 question-id 列表（150 题）。
- `subsets/s_150.json`：`make_subset.py --n 150 --seed 42` 按题型配额抽出的 **151 题**，只用于探路，不是配对的 150 附表。

```bat
python run_mem0_lme.py ^
  --data longmemeval_s_cleaned.json ^
  --subset subsets\s_150_locked.json ^
  --topk 10 --workers 24 ^
  --retrieval dense --model deepseek-flash --max-tokens 8192 ^
  --llm-thinking disabled --question-retries 3 ^
  --embedder BAAI/bge-base-en-v1.5 --embedding-dims 768 ^
  --query-prefix "Represent this sentence for searching relevant passages: " ^
  --oversample 200 --delete-after-question --skip-abs ^
  --profile lme ^
  --extract-cache results\extract_cache\cache_resolve_auto.sqlite ^
  --lme-cache-key auto --lme-date-mode resolve --lme-prompt lme --lme-fact-style rich --lme-batch-sessions 1 ^
  --session-level ^
  --out-root results\mem0_bge_session
```

### 2) S 全量 470 session 级

同上，去掉 `--subset`，加 `--limit 470`。

### 3) S 全量 470 fact 级（对照，原版）

去掉 `--session-level`，`--out-root results\mem0_bge_s`。

### 4) M session 级（需先充值，成本远高于 S）

`--data longmemeval_m_cleaned.json`，其余同 S；M 每题约 500 个 session。

## 为什么复用提炼缓存 = 0 LLM

- 提炼缓存按 **session 内容** 寻址，不按题目。
- S 已跑完 470/470，缓存里已有全部唯一 session 的 fact。
- 用同一个 `--extract-cache` 重跑 session 级，`cache.get_facts()` 全部命中，**不再调用 LLM**，只做本地 BGE embed + Qdrant 检索。
- 确认：看 run.log 的 `extract cache` 命中率接近 100%，且 `llm_usage.json` 的 `calls=0`。
- 不带缓存或缓存缺失 = 完整 LLM 重跑（S 约 2067 次调用，off-peak 约 $0.81）。

## 官方口径打分

```bash
python longmemeval/eval_retrieval_official.py \
  results/mem0_bge_session/mem0_<timestamp>/predictions.jsonl \
  --gold-key answer_session_ids --retrieved-key retrieved_sessions \
  --ks 5 10
```

口径锁定：

- `recall_all@5/10`（主指标）、`ndcg@5/10`（排名指标），另看 `recall_any@5/10`。
- 跳过 `question_id` 以 `_abs` 结尾的 abstention 题。
- 全量分母 = **470**（500 题含 30 道 abstention）。
- **不要引** runner 自带 `aggregate.json` 的 `recall`，那是 fractional recall（`hits/len(gold)`），不是官方口径。

## 已发布结果

| what | where | headline |
|---|---|---|
| mem0 S fact 级 470/470 | `mem0_results/` | `recall_all@5 = 0.9234`, `ndcg@5 = 0.9279` |
| SeCom + BGE M 470/470 | `longmemeval/RETRIEVAL_RESULTS.md` | `recall_all@5 = 0.6489`, `ndcg@5 = 0.6931` |

> 注意：mem0 S `0.9234` 是 fact 级（不可比）。论文主表应使用 session 级（`--session-level`）重跑结果。mem0 M 只完成 98/470（余额不足），不可引用。
