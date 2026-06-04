# 收官消融实验结果（2026-06-03）

数据集：120 题 = in-domain 100（graph-grounded）+ OOD 20（库外菜，含人工 ground_truth）。
方法学：主链=累加式消融（每行叠加一个模块）；路由=受控变量消融（固定管线只换路由器）。
环境：graph-rag conda env + `KMP_DUPLICATE_LIB_OK=TRUE` + `PYTHONIOENCODING=utf-8`。

> **诚实标注**：① 主链相邻 Δ 是"在已有模块之上再加该模块"的边际收益，非模块独立贡献。② OOD 无 gold 节点，检索指标不适用故不报。③ OOD 不报 faithfulness——它衡量"答案被检索片段支撑的比例"，Tavily 短 snippet 撑不起 CRAG 的完整菜谱、而非 CRAG 的简短拒答反而易被判支撑，故对 OOD 有误导。OOD 质量以 LLM 判分正确性为准。

## 1. 主链累加（in-domain 100）

| 版本 | Hit@5 | Recall@5 | MRR@10 | Faithfulness | Answer Rel | Routing Acc | P50(ms) | P95(ms) | 答案正确性(emb) |
|---|---|---|---|---|---|---|---|---|---|
| baseline（round-robin, 纯LLM路由） | 0.890 | 0.841 | 0.625 | 0.690 | 0.889 | 0.710 | 7190 | 12172 | 0.867 |
| +retrieval（BM25/RRF） | 0.960 | 0.863 | **0.917** | 0.691 | 0.900 | 0.720 | 7281 | 10328 | 0.872 |
| +parentdoc（父文档回填） | 0.960 | 0.854 | 0.913 | **0.850** | 0.904 | 0.710 | 7628 | 10789 | 0.878 |
| +route（function-call 路由） | **0.990** | **0.915** | **0.944** | 0.851 | 0.904 | **1.000** | **4910** | 8930 | 0.893 |
| +crag（网络检索） | 0.990 | 0.915 | 0.944 | **0.884** | 0.910 | 1.000 | 4598 | 8088 | **0.909** |

**读法**：BM25/RRF→检索跃升（MRR 0.625→0.917）；父文档→忠实度跃升（Faith 0.69→0.85）；路由→检索再升 + 延迟降（RouteAcc 0.71→1.0，P50 7.6s→4.9s）；CRAG→in-domain 无回归（Faith 0.851→0.884）。

## 2. 路由专项（受控变量：固定 BM25/RRF+父文档, CRAG off）

| 路由器 | Routing Acc | P50(ms) | P95(ms) |
|---|---|---|---|
| 纯 LLM（datawhale 原始） | 0.758 | 7476 | 10962 |
| D3 规则（第一版改进，规则短路） | **1.000** | **4154** | **7984** |
| function-call（第二版改进，可扩展） | **1.000** | 5128 | 8971 |

两版自研路由都把准确率 0.758→1.0、延迟大幅下降；规则版最快（免 LLM），function-call 同样满分且更易扩展新检索源。

## 3. OOD（20 题）—— CRAG 主场

| 版本 | LLM 判分正确性 | 答案正确性(emb) | OOD 行为 |
|---|---|---|---|
| baseline | 0.11 | 0.866 | 17/20 拒答 |
| +retrieval | 0.19 | 0.876 | 16/20 拒答 |
| +parentdoc | 0.03 | 0.830 | 17/20 拒答 |
| +route | 0.10 | 0.853 | 20/20 拒答 |
| +route_rule | 0.03 | 0.856 | 18/20 拒答 |
| **+crag** | **0.83** | **0.897** | **20/20 触发网络检索作答** |

**CRAG 机制（来自运行时 flag，100% 可靠）**：OOD **20/20 触发**网络检索、**Tavily 0 失败**、in-domain **0/100 误触发**。

**核心结论**：不开 CRAG 时系统对库外问题诚实拒答（"知识库没有此菜"，不瞎编），OOD LLM 判分正确性仅 0.03–0.19；开 CRAG 后全部去网络补答，正确性升至 **0.83**（约 4–8 倍）。embedding 余弦因地板过高（拒答也 0.86）区分度弱，故以 LLM 判分为准。

## 4. 成功标准核对（plan §4.4）

- ✅ in-domain：CRAG Faithfulness 0.884 ≥ route 0.851（无回归，硬护栏过；实为 +0.03）
- ✅ OOD：CRAG 各正确性指标全场最高（LLM 判分 0.83 vs 次高 0.19）

## 5. run → 配置对照

| run_id | bm25/rrf | parent_doc | router_mode | crag |
|---|---|---|---|---|
| final_baseline | off | off | pure_llm | off |
| final_retrieval | on | off | pure_llm | off |
| final_parentdoc | on | on | pure_llm | off |
| final_route | on | on | tool_calling | off |
| final_crag | on | on | tool_calling | on |
| final_route_rule | on | on | rule | off |
