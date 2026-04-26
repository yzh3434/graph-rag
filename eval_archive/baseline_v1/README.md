# Baseline v1 — 归档说明

## 这次 run 是什么

第一次端到端 baseline 评测，跑完 100 条 testset 拿到 7 项指标。

**重要前提**：v1 使用的 testset `expected_strategy` 标签是 `testset_generator.py` 里写死的 `TYPE_TO_STRATEGY` 映射，**该映射存在教条问题**——把 `entity_relation` 和 `causal` 类查询无脑标记为需图 RAG，实际它们用 hybrid 检索完全够用。

> 后续修订到 v2 时，已将 entity_relation 与 causal 改为 `hybrid_traditional`，仅保留 multi_hop 为 `graph_rag`（真·多跳遍历）、comparison 为 `combined`（保留组合策略测试覆盖）。

因此 v1 的 routing accuracy 数字偏低，**v1 数字不代表系统真实路由能力**，仅作历史对照。

---

## 配置

```json
{
  "use_bm25": false,
  "use_rrf": false,
  "use_rerank": false,
  "topic_alignment": false,
  "retrieve_top_k": 10,
  "embedding_model": "BAAI/bge-small-zh-v1.5",
  "llm_model": "deepseek-chat"
}
```

100 条样本，全部成功（n_failed = 0）。

---

## 总体指标

| 层 | 指标 | 值 | 评级 |
|---|---|---|---|
| 检索 | Hit@5 | 0.890 | ✅ |
| 检索 | Recall@5 | 0.849 | ✅ |
| 检索 | MRR@10 | 0.628 | 🟡 |
| 路由 | Routing Accuracy | 0.480 | 🔴 表面差（标签教条所致） |
| 生成 | Faithfulness | 0.671 | 🟡 被 causal 拖累 |
| 生成 | Answer Relevancy | 0.900 | ✅ |
| 系统 | Latency P50 | 9.86 s | 🔴 慢 |
| 系统 | Latency P95 | 15.97 s | 🔴 |

---

## 6 个核心洞察（这次 run 沉淀的认识）

这些洞察是基于 v1 数据得出的，**它们的发现过程比数字本身更有价值**——后续在 v2 里很多数字会变，但下面这些观察的方法学仍然成立。

### 洞察 ① 路由器对 graph_rag 类型几乎完全失效

```
expected=hybrid_traditional (n=45) → routing_correct = 0.978
expected=graph_rag         (n=45) → routing_correct = 0.044   ← 43 条被错路由
expected=combined          (n=10) → routing_correct = 0.200
```

45 条期望走 graph_rag 的查询里，43 条被路由到 hybrid_traditional——花大力气做的图 RAG 模块几乎是死代码。

### 洞察 ② 但 hybrid 居然把错路由的 query 救回来一大半（反直觉）

graph_rag 期望那 45 条的实际指标：

```
Hit@5 = 0.756    Recall@5 = 0.664    MRR@10 = 0.535
```

→ testset 标签太教条，**hybrid 检索覆盖了 75% 期望 graph 的 query**。这正是 v2 重新打标的直接动机。

### 洞察 ③ multi_hop 是唯一真正需要图的类型，且彻底崩了

| 类型 | n | Hit@5 | Recall@5 | MRR@10 |
|---|---|---|---|---|
| simple_fact | 15 | 1.000 | 1.000 | 0.617 |
| attribute_query | 15 | 1.000 | 1.000 | 0.733 |
| step_by_step | 15 | 1.000 | 1.000 | 0.600 |
| entity_relation | 15 | 1.000 | 1.000 | 0.583 |
| comparison | 10 | 1.000 | 1.000 | 0.950 |
| causal | 15 | 0.933 | 0.933 | 0.817 |
| **multi_hop** | **15** | **0.333** | **0.058** | **0.205** |

multi_hop（食材共现查询）是图 RAG 的真正考场，hybrid 完全做不了"两个食材共同出现在哪些菜里"——必须 Neo4j 多跳。修 multi_hop = 修整个项目最差的指标。

### 洞察 ④ causal 的 Faithfulness 0.235 是评测方法学问题

```
causal: Hit@5 = 0.933 (检索几乎全对)
        Faithfulness = 0.235 (跌到地板)
```

菜谱 context 只有"步骤描述"没有"为什么"的解释，causal 问题问"为什么"，LLM 必须用通用烹饪知识补——RAGAS 把这些通用知识全判定为不被支撑。**这是 RAGAS 在 causal 任务上的固有局限，不是系统 bug**。

剔除 causal 后整体 Faithfulness ≈ **0.748**，更可信。

### 洞察 ⑤ Round-robin 的 rank 2 偏置仍存在（部分类型）

各类型 MRR 反推平均命中位置：

| 类型 | MRR | 平均 rank |
|---|---|---|
| comparison | 0.95 | ~1.05 |
| causal | 0.817 | ~1.22 |
| attribute_query | 0.733 | ~1.36 |
| simple_fact | 0.617 | ~1.62 |
| step_by_step | 0.600 | ~1.67 |
| entity_relation | 0.583 | ~1.72 |
| multi_hop | 0.205 | ~4.88 |

step_by_step / entity_relation 的 rank ~1.7 暗示 round-robin 把 dual_level 第一名（不太相关）排到 vector 第一名（最相关）前面 → RRF 改进有空间，预计 MRR 0.63 → 0.72-0.75。

### 洞察 ⑥ 难度分层数据非常干净

```
easy   (n=30): Hit@5=1.000, MRR=0.675, route=0.967, faith=0.879
medium (n=30): Hit@5=1.000, MRR=0.592, route=0.500, faith=0.680
hard   (n=40): Hit@5=0.725, MRR=0.620, route=0.100, faith=0.510
```

hard 几乎被 multi_hop+causal+comparison 占满，是路由失败和检索失败的集中区。

---

## v1 → v2 该改什么

| 改动 | v2 是否做 | 备注 |
|---|---|---|
| TYPE_TO_STRATEGY 映射修订 | ✅ 已修 | entity_relation/causal → hybrid_traditional；multi_hop 保持 graph_rag；comparison 保持 combined |
| testset relabel | ✅ 已做 | 不重生成，只批量更新标签字段 |
| 路由 prompt 修改 | ❌ v2 暂不动 | 等 v2 数据出来看 multi_hop 路由情况再决定 |
| graph_rag_retrieval node_id metadata bug | ❌ v2 暂不动 | 留作独立改进点 |
| 任何 hybrid_retrieval 改进（BM25/RRF/rerank） | ❌ v2 暂不动 | v2 仍是纯 baseline |

---

## 文件清单

```
baseline_v1_summary.json       聚合 + 分组指标
baseline_v1_per_sample.jsonl   逐样本明细（含 question / answer / metrics / 检索的 node_ids）
baseline_v1_report.md          自动生成的人类可读报告（含 top 10 bad case）
baseline_smoke_*               5 条样本的 smoke 测试（先于全量 run，留作记录）
```
