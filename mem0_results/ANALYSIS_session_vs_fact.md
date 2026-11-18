# S 上 session 级 vs fact 级：0.9404 是原文兜底造的虚高，strict 口径下 fact 级更好

日期：2026-11-18　数据：LongMemEval-S（470 非 abstention）　LLM 调用：**0**（全部缓存命中）

## TL;DR

`--session-level` 在 S-470 上拿到 `recall_all@5 = 0.9404`，比 fact 级的 `0.9234` **高**。
原因是它比 fact 级多做了一件不该混进同一列的事：**空提炼的 session 回退到原始对话文本**
（缓存里 26.3% 的 session 提炼结果为空），而 fact 级下这些 session **一个向量都没有**，
任何 k 都取不到。

这一项足以解释全部差额：470 里 `single-session-assistant` 的 19 道失分题，
**19/19 的 gold session 提炼结果都是 `[]`**；命中的 37 道则全部非空。
37/56 = **0.6607**，正好等于 fact 级该题型在 @5 与 @10 上的同一数值（曲线完全平），
即这些 session 从未进入索引。

第二个嫌疑（`oversample=200` 截断）**已用实验排除**：151 题 fact 级把候选池从 200 提到
2000，`retrieved_sessions` 逐题完全一致（0/151 差异），官方指标也一模一样。
截断在 S 上不是实际约束，详见下文「差异 #2」。

**结论：0.9404 是「74% fact 级 + 26% 原文 BGE 级」的混合结果，不能作为
"mem0 提炼后 session 级"的主表数字，也不能和 BGE / ThinkQE / SeCom 的纯 session 检索并表。**

去掉原文兜底后（strict，470/470，`calls=0`）得到干净的收口对照：
`recall_all@5 = 0.9064`、`nDCG@5 = 0.9086`，**低于 fact 级的 0.9234 / 0.9279**，
assistant 精确回落到 0.6607。也就是说：口径对齐后 **fact 级才是更好的粒度**，
`0.9404` 的高出部分全部来自那 26% 的原文兜底。详见「strict session 级」一节。

## 结果一览（官方口径，跳过 `_abs`，分母 = 题数）

| run | 口径 | n | recall_all@5 | nDCG@5 | recall_all@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| `mem0_s_151_fact` | fact 级 | 151 | 0.9007 | 0.9158 | 0.9338 | 0.9197 |
| `mem0_s_151_fact_os2000` | fact 级，`oversample=2000` | 151 | 0.9007 | 0.9158 | 0.9338 | 0.9197 |
| `mem0_s_151_session` | session 级 | 151 | 0.9536 | 0.9503 | 0.9934 | 0.9562 |
| `mem0_s_470_fact`（即 `mem0_lme_s`） | fact 级 | 470 | 0.9234 | 0.9279 | 0.9489 | 0.9315 |
| `mem0_s_session_470` | session 级（含原文兜底） | 470 | **0.9404** | **0.9453** | **0.9809** | **0.9530** |
| `mem0_s_session_470_strict` | session 级 strict | 470 | 0.9064 | 0.9086 | 0.9426 | 0.9151 |

辅助指标（470）：

| 口径 | recall_any@5 | fractional@5 | recall_any@10 | fractional@10 |
|---|---:|---:|---:|---:|
| fact 级 | 0.9574 | 0.9465 | 0.9574 | 0.9541 |
| session 级 | 0.9872 | 0.9694 | 0.9936 | 0.9893 |

> 151 子集是 `make_subset.py --n 150 --seed 42` 按题型配额取整后得到的 **151 题**
> （见 `subsets/s_150.json`），不是某个外部锁定的 150 id 列表；151 与 470 的差异
> 属于抽样噪声，不要拿两者互相比较。

## 470 分题型

| 题型 | n | fact R@5 | session R@5 | Δ | fact nDCG@5 | session nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| knowledge-update | 72 | 1.0000 | 0.9861 | −0.0139 | 0.9935 | 0.9807 |
| multi-session | 121 | 0.9504 | 0.9504 | 0.0000 | 0.9755 | 0.9647 |
| **single-session-assistant** | 56 | **0.6607** | **1.0000** | **+0.3393** | 0.6607 | 0.9832 |
| single-session-preference | 30 | 1.0000 | 0.9333 | −0.0667 | 0.8672 | 0.8385 |
| single-session-user | 64 | 1.0000 | 0.9844 | −0.0156 | 0.9885 | 0.9514 |
| temporal-reasoning | 127 | 0.9134 | 0.8583 | −0.0551 | 0.9471 | 0.9121 |

两个方向都真实存在：

- **+** assistant 类是"空提炼 → 索引里没有"，session 级的原文回退把它救回来；
- **−** temporal / preference / user 类是"收口损失"，把 4.2 条 fact 平均成一条向量后，
  同题多候选 session 的区分度变差（nDCG 掉得比 recall_all 更明显）。

## 差异 #1：空提炼的 session 在 fact 级不存在

`lme_profile.write_sessions_lme()` 的收口逻辑：

```python
if session_level:
    for sid, dt, facts in per_session:
        text = "\n".join(facts) if facts else raw_texts.get(sid, "")   # ← 原文兜底
```

缓存现状：**20,227 条 session 提炼中 5,327 条为空列表（26.3%）**。
fact 级对空列表不产生任何记忆点 → 该 session 在 Qdrant 里不可见。

逐题核对（470 fact 级，`single-session-assistant`）：

| | 题数 | gold session 的 facts |
|---|---:|---|
| 命中 | 37 | 全部非空 |
| 未命中 | 19 | **全部为空 `[]`** |

同一核对可复现：`.cache/diag_assistant.py`（本次分析用的临时脚本，未入库）。

## 差异 #2（已排除）：fact 级候选池截断

`run_mem0_lme.py` 的检索路径是**先截点、再去重**：

```python
hits = memory.vector_store.search(..., top_k=oversample)   # oversample = 200
ranked = hits_to_sessions(hits, topk)                      # 先截 200 点，再去重成 session
```

数字上确实存在风险窗口：

- fact 级每题记忆点：**均值 203.6 / 中位数 203 / 最大 307**，**258/470 题 > 200**
- session 级每题只有 ~48 个点（每题 session 均值 47.6），**从不触发截断**

**实验结论：截断在 S 上不是实际约束。**
151 题 fact 级、`--oversample 2000`（候选池放大 10 倍，远超最大 307 点）：

| | recall_all@5 | nDCG@5 | recall_all@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| `oversample=200` | 0.9007 | 0.9158 | 0.9338 | 0.9197 |
| `oversample=2000` | 0.9007 | 0.9158 | 0.9338 | 0.9197 |

逐题对比：**0/151 题的 `retrieved_sessions` 有任何差异**，`n_memories_added` 也完全相同，
分题型表逐项一致（assistant 仍是 0.5000，temporal 仍是 0.9024）。
即 gold session 的最佳 fact 本来就落在前 200 名内，放大候选池只是把长尾点取回来。

同时这也解释了 `recall_any@10 == recall_any@5`（470 fact 级 0.9574）：
那是"多 gold 题只缺一个 session"的现象，不是排名深度不够。

所以差额**只剩差异 #1**。这一点很关键：修 fact 级不需要动 `oversample`，
只需要把 session 级的原文兜底去掉，两边就共享同一信息源。

## strict session 级（干净口径，470）

`--session-level-strict`：`facts == []` 的 session 不入库，其余完全不变。
470/470 ok，`calls=0`，缓存 `hits=24,435 misses=0`。

| 口径 | recall_all@5 | nDCG@5 | recall_all@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| **fact 级** | **0.9234** | **0.9279** | **0.9489** | **0.9315** |
| session 级（含原文兜底） | 0.9404 | 0.9453 | 0.9809 | 0.9530 |
| session 级 strict | 0.9064 | 0.9086 | 0.9426 | 0.9151 |

去掉兜底后 session 级**低于** fact 级：R@5 −0.0170、nDCG@5 −0.0193、
R@10 −0.0063、nDCG@10 −0.0164。

分题型（strict vs fact，470）：

| 题型 | n | fact R@5 | strict R@5 | Δ | fact nDCG@5 | strict nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| knowledge-update | 72 | 1.0000 | 0.9861 | −0.0139 | 0.9935 | 0.9807 |
| multi-session | 121 | 0.9504 | 0.9587 | +0.0083 | 0.9755 | 0.9693 |
| single-session-assistant | 56 | 0.6607 | 0.6607 | 0.0000 | 0.6607 | 0.6518 |
| single-session-preference | 30 | 1.0000 | 0.9333 | −0.0667 | 0.8672 | 0.8385 |
| single-session-user | 64 | 1.0000 | 0.9844 | −0.0156 | 0.9885 | 0.9535 |
| temporal-reasoning | 127 | 0.9134 | 0.8740 | −0.0394 | 0.9471 | 0.9171 |

三点值得记下来：

1. **assistant 精确回落到 0.6607**，与 fact 级逐位相同。这是差异 #1 的闭环证明：
   只要把"空提炼 session 用原文顶上"这条路切断，session 级在该题型上就只会
   拿到 fact 级本来就拿到的 37/56。
2. strict 比含兜底低 0.0340（R@5）——这 0.034 就是兜底制造的虚高。
3. 其余题型 strict 仍略低于 fact（multi-session 的 recall 小涨 0.0083，但 nDCG 跌 0.0062），
   说明"把平均 4.2 条 fact 压成一条向量"本身是净损失。**在 mem0 这条链路上，
   session 收口不是更好的粒度。**

## 锁定的 150 附录（`subsets/s_150_locked.json`）

150 题全部非 abstention（multi 40 / temporal 40 / KU 23 / user 21 / assistant 17 /
preference 9），与 seed42 的 151 只重叠 56 题。两套都 150/150 ok，`calls=0`，缓存 0 缺失。

| 口径 | recall_all@5 | nDCG@5 | recall_all@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| **fact 级** | 0.9067 | **0.9175** | 0.9333 | **0.9217** |
| session 级 strict | **0.9133** | 0.8922 | **0.9400** | 0.8970 |

分题型（strict vs fact）：

| 题型 | n | fact R@5 | strict R@5 | Δ | fact nDCG@5 | strict nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| single-session-assistant | 17 | 0.5294 | **0.5294** | 0.0000 | 0.5294 | 0.5000 |
| multi-session | 40 | 0.9250 | 0.9750 | +0.0500 | 0.9607 | 0.9520 |
| knowledge-update | 23 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.9832 |
| single-session-user | 21 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.9473 |
| temporal-reasoning | 40 | 0.9250 | 0.9250 | 0.0000 | 0.9575 | 0.9276 |
| single-session-preference | 9 | 1.0000 | 0.8889 | −0.1111 | 0.8770 | 0.8479 |

和 470 一致、也最关键的一条：**assistant 在两个口径下逐位相同（0.5294 = 9/17）**。
这是"空提炼 session 在 strict 下根本不存在"的直接结果，和 470 的 0.6607 是同一现象，
说明差异 #1 的归因不依赖具体抽样。

在这 150 题上 recall 反而 strict 略高（+0.0066 @5、+0.0067 @10），但 nDCG 明显更低
（−0.0253 @5、−0.0247 @10）。解释：收口后候选变少，命中"任一" gold session 更容易，
但一条向量要同时代表一个 session 里的多个主题，排序质量下降。
注意 n=150 时**单题权重就是 1/150 = 0.0067**，所以 R@5 的差值恰好等于**一道题**，
不能据此声称"session 级 recall 更好"；能站住的是"nDCG 稳定更差 + assistant 完全一致"。

## 结论与建议

- **主表口径：fact 级。** 470 题 `recall_all@5 = 0.9234`、`nDCG@5 = 0.9279`
  （`mem0_lme_s/`）；锁定 150 附表 `0.9067 / 0.9175`（`mem0_s_locked150_fact/`）。
- **session 级只作消融，且必须用 strict。** 470：`0.9064 / 0.9086`
  （`mem0_s_session_470_strict/`）；150：`0.9133 / 0.8922`
  （`mem0_s_locked150_session_strict/`）。含原文兜底的 `0.9404` 是虚高，
  任何表格都不应引用。
- 两套数据上 **fact 级 nDCG 都更高**（470 +0.0193、150 +0.0253），recall 在 470 上
  fact 更高、在 150 上只差一道题。所以稳妥的结论是"**nDCG 稳定占优，recall 打平**"，
  而不是"session 级 recall 更好"。
- 若日后在更大数据集上复用，仍建议把 `hits_to_sessions` 改成"先按 session 去重、
  再截 top-k"，作为与数据规模无关的稳健写法（在 S 上已证明不是必要项，见差异 #2）。
- M 数据集仍未跑完，需要 API 余额，与本次分析无关。

实现细节（`lme_profile.py`）：

```python
if facts:
    text = "\n".join(facts)
else:
    text = raw_texts.get(sid, "") if empty_fallback else ""   # strict: 空
```

3 题冒烟验证（`89527b6b` / `6ae235be` / `ceb54acb`，都是 assistant 空 facts 题）：
入库向量数 = 该题**非空** session 数（33 / 32 / 39，与离线统计逐题相等），
gold session 全部从检索结果中消失，`calls=0`。

## 复现与证据

```bash
# session 级 470
python run_mem0_lme.py --data longmemeval_s_cleaned.json --limit 470 \
  --topk 10 --workers 24 --retrieval dense --model deepseek-flash \
  --embedder BAAI/bge-base-en-v1.5 --embedding-dims 768 \
  --query-prefix "Represent this sentence for searching relevant passages: " \
  --oversample 200 --delete-after-question --skip-abs --profile lme \
  --extract-cache results/extract_cache/cache_resolve_auto.sqlite \
  --lme-cache-key auto --lme-date-mode resolve --lme-prompt lme \
  --lme-fact-style rich --lme-batch-sessions 1 \
  --session-level --out-root results/mem0_bge_session

# strict session 级 470（去掉空 facts 的原文兜底）
#   同上，但把 --session-level 换成 --session-level-strict
python run_mem0_lme.py --data longmemeval_s_cleaned.json --limit 470 \
  --topk 10 --workers 16 --retrieval dense --model deepseek-flash \
  --max-tokens 8192 --llm-thinking disabled --question-retries 3 \
  --embedder BAAI/bge-base-en-v1.5 --embedding-dims 768 \
  --query-prefix "Represent this sentence for searching relevant passages: " \
  --oversample 200 --delete-after-question --skip-abs --profile lme \
  --extract-cache results/extract_cache/cache_resolve_auto.sqlite \
  --lme-cache-key auto --lme-date-mode resolve --lme-prompt lme \
  --lme-fact-style rich --lme-batch-sessions 1 \
  --session-level-strict --out-root results/mem0_bge_session_strict

# 官方复算
python longmemeval/eval_retrieval_official.py \
  results/mem0_bge_session/mem0_20261118-150914/predictions.jsonl \
  --gold-key answer_session_ids --retrieved-key retrieved_sessions --ks 5 10
```

每个结果目录都带 `evidence.log`，内含 `extract cache: hits=… misses=0`、
`LLM usage: calls=0 … est_cost $0.00`、`Finished. n_total=… n_ok=…`。

## 已知瑕疵

- `mem0_s_session_470/meta.json` 与 `llm_usage.json` 是 **resume 覆盖后**的状态
  （主跑 24 workers，收尾崩溃后用 8 workers 补跑 1 题；`hits=43` 只统计补跑那一轮）。
  主跑证据见 `evidence.log`。
- 主跑在 repair 阶段崩过一次：`OperationalError('cannot rollback - no transaction is active')`
  （sqlite 多线程），补跑后 470/470 全部 ok。
- 进程退出码 1 是 `qdrant_client.__del__` 在解释器关闭时的
  `ImportError: sys.meta_path is None`，不影响产物。
