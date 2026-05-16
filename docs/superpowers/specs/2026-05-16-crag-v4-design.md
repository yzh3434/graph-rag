# CRAG v4 设计文档

- 日期：2026-05-16
- 状态：待用户正式审核
- 关联版本主线：baseline_v2 → route_v3（function calling 路由）→ retrieval_v2 → **v4（CRAG 网络检索）**

## 1. 目标与动机

现有系统的检索完全依赖本地烹饪知识图谱（Neo4j）+ 向量库。知识库未收录的菜（西餐、日餐、东南亚菜等）问出来时，检索只能召回不相关的中餐菜谱 chunk，生成层被迫基于无关上下文作答 → 幻觉或答非所问。

引入 **Corrective RAG（CRAG，纠正式 RAG，Yan et al. 2024）** 思路：生成阶段先自评检索到的上下文是否足以回答；不足则触发外部网络检索（Tavily）补充上下文后再生成。这是**领域外覆盖度/鲁棒性**改进，不改动检索与路由。

## 2. 决策记录（为什么是现在这个方案）

| 决策点 | 选项 | 结论 | 理由 |
|---|---|---|---|
| 充分性判断位置 | 路由层加第四个 tool / 生成层自评 | **生成层自评** | 充分性必须在拿到检索证据之后才能判；路由前无证据可判 |
| 触发门控 | A 廉价前置门控 / B 统一单次调用 / C 恒定两次调用 | **B** | 用户取舍：实现简单；B 唯一风险（领域内 Faithfulness 回归）会被四版全量重跑自动测到，非裸奔 |
| 「检索弱」信号 | OOD 标签 / 空结果 / RRF top-1 分 / 向量余弦 | **均不采用（走 B）** | OOD 标签作触发=数据泄漏；空结果几乎不发生；RRF 分是排名分不反映相关度。B 不需要此信号 |
| 不足信号格式 | 哨兵 token / JSON | **JSON** | JSON 可顺带回传改写后的检索词（CRAG 论文中此步对网络检索质量影响大） |
| 网络检索后端 | Tavily / DuckDuckGo | **Tavily** | 专为 LLM/RAG 设计，返回干净 snippet；用户申请 API key |
| OOD 题构造 | 纯领域外 / 含半可答 | **纯领域外** | 第一版要干净可归因；半可答留作后续（YAGNI） |
| 测试集标记是否泄漏 | — | 不泄漏 | B 的触发完全靠 LLM 自评，metadata.domain 仅用于评测报告分层，运行时不读 |

## 3. 运行时架构（B 路线，最小侵入）

- 新建 `rag_modules/crag_generation.py`：薄编排器 `CRAGGenerator`，内部持有现成 `GenerationIntegrationModule` 实例。
- **不修改** `generation_integration.py::generate_adaptive_answer` —— 领域内零回归的结构性保证。
- `config.py` 新增 `enable_crag: bool = False`。
- 接入点（两处，开关上走 CRAG，否则走原方法）：
  - `main.py` `ask_question_with_routing`（约 line 297）
  - `eval/eval_runner.py` `eval_sample`（约 line 289）

接口：

```python
class CRAGGenerator:
    def __init__(self, generation_module, config): ...
    def generate(self, question: str, documents: list[Document]) -> tuple[str, dict]:
        # 返回 (answer, crag_meta)
        # crag_meta = {
        #   "triggered": bool,        # 是否触发了网络检索
        #   "web_query": str | None,  # LLM 改写后的检索词
        #   "web_context": str,       # 网络检索拼接的上下文（评测算 Faithfulness 用）
        #   "n_web_results": int,
        #   "web_failed": bool,       # Tavily 失败/无结果时的降级标记
        # }
```

## 4. 运行时数据流（单次调用 + 一轮网络检索硬上限）

1. 用 `documents` 拼 KB 上下文（复用 `generate_adaptive_answer` 现有拼接逻辑）。
2. **LLM 调用①**：prompt = 原生成 prompt + 前置规则：
   > 若检索到的信息明显不足以回答本问题，请**只**输出 JSON `{"insufficient": true, "search_query": "<改写后的检索词>"}`；否则照常正常作答，**不要**输出任何 JSON。
3. 解析返回：
   - 是正常答案 → 返回 `(answer, {"triggered": false, ...})`。**happy path：1 次 LLM 调用，领域内行为与原系统一致。**
   - 干净识别到 insufficient 信号 → 进入第 4 步。
   - 疑似 JSON 但实为答案（宽松解析失败）→ **当作答案处理**（偏向保护领域内，不误触发）。
4. 用 `search_query` 调 Tavily（top ~5），拼 `web_context`。
5. **LLM 调用②**：KB 上下文 + web_context 一起重新生成（原 prompt 风格，注明部分信息来自网络检索）。
6. 返回 `(answer, {"triggered": true, "web_query":..., "web_context":..., ...})`。
7. **只一轮，不递归**：补完仍不足也不再搜。

## 5. 错误处理与降级（绝不硬失败）

| 情况 | 处理 |
|---|---|
| Tavily key 缺失 / API 报错 / 零结果 | 跳过网络增强，用 KB 上下文尽力答 + 标注「知识库未收录，以下为通用知识参考」；`web_failed=true` |
| LLM① 输出疑似 JSON 但实为答案 | 宽松解析；只有干净识别到 insufficient 才触发，否则当答案 |
| Tavily 超时 | 10s 超时，超时按「报错」降级 |
| LLM②（重生成）异常 | 复用现有生成模块的 try/except 降级返回错误串 |

## 6. 评测接入（关键——发现 2/3/4 的落地）

### 6.1 Faithfulness 上下文（发现 2）
`eval_runner.eval_sample` 算 Faithfulness 的 `context` 必须 = KB context **+ `crag_meta.web_context`**。否则网络证据支撑的答案会被冤判为幻觉（Faithfulness ≈ 0）。

### 6.2 领域外指标分层（发现 3）
- OOD 样本 `source_node_ids=[]` → hit@5/recall@5/mrr@10 天然为 0（预期内，KB 无法召回）。
- 不能只看混合 120 题总分。改动：
  - `eval_runner.eval_sample`：`record["domain"] = sample.metadata.get("domain", "in_domain")`。
  - OOD 样本：`metrics["routing_correct"] = None`（OOD 不计路由，`_mean` 自动跳过 None）。
  - `metrics_aggregator.aggregate_metrics`：新增 `by_domain = group_metrics(valid, "domain")`，并拆出 `aggregated_in_domain` / `aggregated_out_of_domain`；保留 overall `aggregated`。
  - `write_markdown_report`：新增「按 domain 分组」一节。

### 6.3 延迟与触发记录（发现 4）
`eval_sample` 新增计时与字段：`metrics["latency_websearch_ms"]`、`record["crag_triggered"]`、`record["web_query"]`。

## 7. 领域外测试集规格（纯 OOD，20 题，总 120）

- 先用 Cypher 枚举 Neo4j 全部 `Recipe.name` 做**避让名单**，确保 20 道菜均不在库内（库内已有「巴基斯坦牛肉咖喱」等，必须 enumerate 后避开）。
- question_type 仅用两类（其余类对 OOD 不合适）：

| question_type | 问法模板 | difficulty | 数量 |
|---|---|---|---|
| `step_by_step` | "如何制作惠灵顿牛排？" / "提拉米苏的制作步骤是什么？" | medium | 10 |
| `entity_relation` | "制作冬阴功汤需要哪些主要食材？" / "做亲子丼需要准备什么食材？" | medium | 10 |

- 每条字段：
  - `ground_truth`：助手起草通用烹饪常识 → **用户审核**
  - `expected_strategy`：`""`（合法值，OOD 不计路由）
  - `source_node_ids`：`[]`
  - `metadata`：`{"generated_by": "manual_ood", "domain": "out_of_domain"}`
- 选菜方向：西餐 / 日餐 / 东南亚 / 印度等非中餐菜系，枚举避让后最终确定。

## 8. 成功标准 / 对照（v4 故事核心，零额外成本）

四个版本（baseline_v2 / route_v3 / retrieval_v2 / v4）**都跑完整 120 题**，于是：

- **in-domain 100 题**：v4 的 Faithfulness 相比 retrieval_v2 **不下降**（B 唯一风险点，硬护栏，必须验证）；检索/路由指标持平（CRAG 不动检索路由）。
- **out-of-domain 20 题**：v1/v2/v3 因无网络检索 → OOD Faithfulness 低；v4 → 显著拉高。**这个 delta 即 CRAG 价值铁证，无需额外跑对照**（v1/v2/v3 在 OOD 切片上天然就是「CRAG off」基线）。

## 9. 不在本次范围（YAGNI）

- MCP 化（先本地 function，面试可讲「下一步迁 MCP」）
- 多轮网络检索 / 递归纠正
- 半可答（partial-OOD）题型
- 廉价前置门控 / 检索评估器（B 路线不需要）

## 10. 涉及改动文件清单

| 文件 | 改动 |
|---|---|
| `rag_modules/crag_generation.py` | 新建 `CRAGGenerator` |
| `config.py` | 加 `enable_crag: bool = False` |
| `main.py` | `ask_question_with_routing` 接入 CRAG 分支 |
| `eval/eval_runner.py` | CRAG 分支 + Faithfulness 上下文增强 + domain/延迟/触发字段 |
| `eval/metrics_aggregator.py` | `by_domain` 分组 + in/out 拆分 + 报告新增分组节 |
| `testset_output/testset.jsonl` | 追加 20 条 OOD 样本（100 → 120） |
| `requirements.txt` | 加 `tavily-python` |
| `.env` | 用户加 `TAVILY_API_KEY`（不入仓库） |

## 11. 已知风险

- B 改了生成 prompt 前缀 → 领域内 Faithfulness 可能小幅回归；靠 §8 护栏验证，回归则需收紧 prompt（让「正常作答」成为强默认）。
- Tavily 免费额度有限；评测 120 题时仅 OOD（~20）会触发，量可控。
- DeepSeek 对「要么 JSON 要么纯文本」的指令遵循偶有不稳 → 宽松解析 + 偏向当答案兜底。
