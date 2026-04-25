"""
Graph RAG 系统评测主程序（baseline & 消融通用）

MVP 7 指标：
  检索层  Hit@5、Recall@5、MRR@10
  路由层  Routing Accuracy
  生成层  Faithfulness、Answer Relevancy（手搓 RAGAS 标准算法）
  系统层  Latency P50 / P95

输出三件套：
  {run_id}_per_sample.jsonl   逐样本明细（实时落盘，中断不丢）
  {run_id}_summary.json       聚合 + 分组指标（消融对比图直接读这个）
  {run_id}_report.md          人类可读报告（含 bad case）

每次改进做消融时只需换 --run_id 与 run_config，输出 schema 完全一致。
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

import numpy as np
from tqdm import tqdm

_C9_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _C9_DIR not in sys.path:
    sys.path.insert(0, _C9_DIR)

from main import AdvancedGraphRAGSystem
from eval.utils import load_testset

logger = logging.getLogger(__name__)


# ==================== 检索层指标 ====================

def hit_at_k(retrieved: List[str], relevant: List[str], k: int) -> int:
    return int(any(rid in relevant for rid in retrieved[:k]))


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for rid in retrieved[:k] if rid in relevant)
    return hits / len(relevant)


def mrr_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    for rank, rid in enumerate(retrieved[:k], 1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


# ==================== 生成层指标（手搓 RAGAS）====================

class GenerationEvaluator:
    """
    手搓版 RAGAS 的 Faithfulness 与 Answer Relevancy。

    Faithfulness（防幻觉，越高越忠实于上下文）
      Step 1  LLM 把 answer 拆成 N 条独立陈述
      Step 2  对每条陈述，LLM 判断能否在 context 中得到支撑
      score = supported_count / total_statements

    Answer Relevancy（防答非所问，越高越切题）
      Step 1  LLM 从 answer 反推 K 个最可能的原始问题
      Step 2  embedding 计算反推问题与原 question 的余弦相似度
      score = mean(cosine_sim)
    """

    def __init__(self, llm_client, model_name: str, embed_fn):
        self.llm = llm_client
        self.model = model_name
        self.embed_fn = embed_fn

    # ---- Faithfulness ----

    def faithfulness(self, answer: str, context: str) -> Optional[float]:
        if not answer.strip() or not context.strip():
            return None
        statements = self._extract_statements(answer)
        if not statements:
            return None
        supported = sum(1 for s in statements if self._verify_statement(s, context))
        return supported / len(statements)

    def _extract_statements(self, answer: str) -> List[str]:
        prompt = f"""把下面这段答案拆解成多条独立的、可独立验证的事实陈述（每条只表达一个原子事实）。

答案：
{answer}

要求：
1. 不要遗漏答案中的关键事实，也不要凭空添加未提及的内容
2. 严格返回 JSON 格式：{{"statements": ["陈述1", "陈述2", ...]}}
3. 若答案无明确事实，返回 {{"statements": []}}
"""
        try:
            resp = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            return [s.strip() for s in data.get("statements", []) if s and s.strip()]
        except Exception as e:
            logger.warning(f"陈述抽取失败: {e}")
            return []

    def _verify_statement(self, statement: str, context: str) -> bool:
        prompt = f"""判断下面这条陈述能否从给定的上下文中得到支撑。

上下文：
{context}

陈述：
{statement}

判断标准：
- supported=true：陈述的事实在上下文中明确出现，或可由上下文直接推得
- supported=false：陈述包含上下文未提及的细节，或与上下文矛盾

要求严格返回 JSON：{{"supported": true 或 false}}
"""
        try:
            resp = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            return bool(data.get("supported", False))
        except Exception as e:
            logger.warning(f"陈述验证失败: {e}")
            return False

    # ---- Answer Relevancy ----

    def answer_relevancy(self, question: str, answer: str, n_questions: int = 3) -> Optional[float]:
        if not question.strip() or not answer.strip():
            return None
        gen_qs = self._reverse_generate_questions(answer, n_questions)
        if not gen_qs:
            return None
        try:
            q_emb = np.array(self.embed_fn(question), dtype=np.float32)
            sims = []
            for gq in gen_qs:
                gq_emb = np.array(self.embed_fn(gq), dtype=np.float32)
                sims.append(self._cosine(q_emb, gq_emb))
            return float(np.mean(sims))
        except Exception as e:
            logger.warning(f"answer_relevancy embedding 失败: {e}")
            return None

    def _reverse_generate_questions(self, answer: str, n: int) -> List[str]:
        prompt = f"""根据下面这段答案，反推出 {n} 个最可能产生这段答案的原始用户问题。

答案：
{answer}

要求：
1. 反推的问题应该是普通用户会自然问出的形式
2. 严格返回 JSON：{{"questions": ["问题1", "问题2", ...]}}
3. 共生成 {n} 个问题
"""
        try:
            resp = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            return [q.strip() for q in data.get("questions", []) if q and q.strip()]
        except Exception as e:
            logger.warning(f"反推问题失败: {e}")
            return []

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


# ==================== 评测主控 ====================

class EvalRunner:

    RETRIEVE_TOP_K = 10  # 评测时强制取 top-10，让 MRR@10 名副其实

    def __init__(
        self,
        rag_system: AdvancedGraphRAGSystem,
        testset: List[Any],
        output_dir: Path,
        run_id: str,
        run_config: Dict[str, Any],
        skip_generation_eval: bool = False,
    ):
        self.rag_system = rag_system
        self.testset = testset
        self.output_dir = output_dir
        self.run_id = run_id
        self.run_config = run_config
        self.skip_generation_eval = skip_generation_eval

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.per_sample_path = self.output_dir / f"{run_id}_per_sample.jsonl"
        self.summary_path = self.output_dir / f"{run_id}_summary.json"
        self.report_path = self.output_dir / f"{run_id}_report.md"

        if not skip_generation_eval:
            self.gen_evaluator = GenerationEvaluator(
                llm_client=rag_system.generation_module.client,
                model_name=rag_system.config.llm_model,
                embed_fn=rag_system.index_module.embeddings.embed_query,
            )
        else:
            self.gen_evaluator = None

    @staticmethod
    def _extract_node_ids(docs) -> List[str]:
        ids = []
        for doc in docs:
            md = doc.metadata or {}
            nid = md.get("node_id") or md.get("parent_id")
            if nid:
                ids.append(str(nid))
        return ids

    @staticmethod
    def _build_context(docs) -> str:
        return "\n\n".join(
            doc.page_content.strip() for doc in docs if doc.page_content and doc.page_content.strip()
        )

    def eval_sample(self, sample, idx: int) -> Dict[str, Any]:
        question = sample.question
        relevant_ids = [str(x) for x in sample.source_node_ids]
        expected_strategy = sample.expected_strategy

        record: Dict[str, Any] = {
            "sample_idx": idx,
            "question": question,
            "question_type": sample.question_type,
            "difficulty": sample.difficulty,
            "expected_strategy": expected_strategy,
            "relevant_node_ids": relevant_ids,
        }

        try:
            t0 = time.perf_counter()
            relevant_docs, analysis = self.rag_system.query_router.route_query(
                question, self.RETRIEVE_TOP_K
            )
            t_retrieve = time.perf_counter() - t0

            predicted_strategy = (
                analysis.recommended_strategy.value if analysis is not None else "unknown"
            )

            t1 = time.perf_counter()
            answer = self.rag_system.generation_module.generate_adaptive_answer(
                question, relevant_docs
            )
            t_generate = time.perf_counter() - t1
            t_total = t_retrieve + t_generate

            retrieved_ids = self._extract_node_ids(relevant_docs)
            context = self._build_context(relevant_docs)

            metrics: Dict[str, Any] = {
                "hit@5":               hit_at_k(retrieved_ids, relevant_ids, 5),
                "recall@5":            recall_at_k(retrieved_ids, relevant_ids, 5),
                "mrr@10":              mrr_at_k(retrieved_ids, relevant_ids, 10),
                "routing_correct":     int(predicted_strategy == expected_strategy),
                "latency_ms":          round(t_total * 1000, 2),
                "latency_retrieve_ms": round(t_retrieve * 1000, 2),
                "latency_generate_ms": round(t_generate * 1000, 2),
            }

            if self.gen_evaluator is not None:
                metrics["faithfulness"]     = self.gen_evaluator.faithfulness(answer, context)
                metrics["answer_relevancy"] = self.gen_evaluator.answer_relevancy(question, answer)
            else:
                metrics["faithfulness"] = None
                metrics["answer_relevancy"] = None

            record.update({
                "predicted_strategy": predicted_strategy,
                "retrieved_node_ids": retrieved_ids[:10],
                "answer": answer,
                "metrics": metrics,
                "error": None,
            })
        except Exception as e:
            logger.error(f"sample {idx} 评测失败: {e}", exc_info=True)
            record.update({
                "predicted_strategy": None,
                "retrieved_node_ids": [],
                "answer": None,
                "metrics": None,
                "error": str(e),
            })

        return record

    def run(self) -> Dict[str, Any]:
        if self.per_sample_path.exists():
            self.per_sample_path.unlink()

        per_sample_records: List[Dict[str, Any]] = []

        with open(self.per_sample_path, "a", encoding="utf-8") as fout:
            for idx, sample in enumerate(tqdm(self.testset, desc=f"评测 {self.run_id}")):
                record = self.eval_sample(sample, idx)
                per_sample_records.append(record)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

        summary = self._aggregate(per_sample_records)
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self._write_markdown_report(summary, per_sample_records)
        return summary

    def _aggregate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid = [r for r in records if r.get("metrics") is not None]
        n_total = len(records)
        n_valid = len(valid)

        def _mean(metric: str) -> Optional[float]:
            vals = [r["metrics"][metric] for r in valid
                    if r["metrics"].get(metric) is not None]
            return float(np.mean(vals)) if vals else None

        def _percentile(metric: str, p: float) -> Optional[float]:
            vals = [r["metrics"][metric] for r in valid
                    if r["metrics"].get(metric) is not None]
            return float(np.percentile(vals, p)) if vals else None

        aggregated = {
            "retrieval": {
                "hit@5":    _mean("hit@5"),
                "recall@5": _mean("recall@5"),
                "mrr@10":   _mean("mrr@10"),
            },
            "routing": {
                "accuracy": _mean("routing_correct"),
            },
            "generation": {
                "faithfulness":     _mean("faithfulness"),
                "answer_relevancy": _mean("answer_relevancy"),
            },
            "system": {
                "latency_p50_ms":          _percentile("latency_ms", 50),
                "latency_p95_ms":          _percentile("latency_ms", 95),
                "latency_retrieve_p50_ms": _percentile("latency_retrieve_ms", 50),
                "latency_generate_p50_ms": _percentile("latency_generate_ms", 50),
            },
        }

        return {
            "run_id":    self.run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config":    self.run_config,
            "n_total":   n_total,
            "n_valid":   n_valid,
            "n_failed":  n_total - n_valid,
            "aggregated":              aggregated,
            "by_question_type":        self._group_metrics(valid, "question_type"),
            "by_expected_strategy":    self._group_metrics(valid, "expected_strategy"),
            "by_difficulty":           self._group_metrics(valid, "difficulty"),
        }

    @staticmethod
    def _group_metrics(records: List[Dict[str, Any]], group_key: str) -> Dict[str, Any]:
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in records:
            buckets[r.get(group_key, "unknown")].append(r)

        out = {}
        for group, rs in buckets.items():
            def _g_mean(metric):
                vals = [r["metrics"][metric] for r in rs
                        if r["metrics"].get(metric) is not None]
                return float(np.mean(vals)) if vals else None
            out[group] = {
                "n":                len(rs),
                "hit@5":            _g_mean("hit@5"),
                "recall@5":         _g_mean("recall@5"),
                "mrr@10":           _g_mean("mrr@10"),
                "routing_correct":  _g_mean("routing_correct"),
                "faithfulness":     _g_mean("faithfulness"),
                "answer_relevancy": _g_mean("answer_relevancy"),
                "latency_ms_mean":  _g_mean("latency_ms"),
            }
        return out

    def _write_markdown_report(self, summary: Dict[str, Any], records: List[Dict[str, Any]]):
        agg = summary["aggregated"]

        def fmt(x):
            if isinstance(x, float):
                return f"{x:.4f}"
            return str(x) if x is not None else "—"

        L = []
        L.append(f"# Eval Report — `{summary['run_id']}`")
        L.append("")
        L.append(f"- **时间**：{summary['timestamp']}")
        L.append(f"- **样本数**：{summary['n_valid']} / {summary['n_total']}（失败 {summary['n_failed']}）")
        L.append(f"- **配置**：")
        L.append("  ```json")
        L.append("  " + json.dumps(summary["config"], ensure_ascii=False))
        L.append("  ```")
        L.append("")

        L.append("## 总体指标")
        L.append("")
        L.append("| 层 | 指标 | 值 |")
        L.append("|---|---|---|")
        L.append(f"| 检索 | Hit@5 | {fmt(agg['retrieval']['hit@5'])} |")
        L.append(f"| 检索 | Recall@5 | {fmt(agg['retrieval']['recall@5'])} |")
        L.append(f"| 检索 | MRR@10 | {fmt(agg['retrieval']['mrr@10'])} |")
        L.append(f"| 路由 | Routing Accuracy | {fmt(agg['routing']['accuracy'])} |")
        L.append(f"| 生成 | Faithfulness | {fmt(agg['generation']['faithfulness'])} |")
        L.append(f"| 生成 | Answer Relevancy | {fmt(agg['generation']['answer_relevancy'])} |")
        L.append(f"| 系统 | Latency P50 (ms) | {fmt(agg['system']['latency_p50_ms'])} |")
        L.append(f"| 系统 | Latency P95 (ms) | {fmt(agg['system']['latency_p95_ms'])} |")
        L.append("")

        L.append("## 按问题类型分组")
        L.append("")
        L.append("| 类型 | n | Hit@5 | Recall@5 | MRR@10 | 路由准确 | Faith | AR | 延迟均值(ms) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for k, v in summary["by_question_type"].items():
            L.append(
                f"| {k} | {v['n']} | {fmt(v['hit@5'])} | {fmt(v['recall@5'])} | "
                f"{fmt(v['mrr@10'])} | {fmt(v['routing_correct'])} | "
                f"{fmt(v['faithfulness'])} | {fmt(v['answer_relevancy'])} | "
                f"{fmt(v['latency_ms_mean'])} |"
            )
        L.append("")

        L.append("## 按预期策略分组")
        L.append("")
        L.append("| 策略 | n | Hit@5 | Recall@5 | MRR@10 | 路由准确 |")
        L.append("|---|---|---|---|---|---|")
        for k, v in summary["by_expected_strategy"].items():
            L.append(
                f"| {k} | {v['n']} | {fmt(v['hit@5'])} | {fmt(v['recall@5'])} | "
                f"{fmt(v['mrr@10'])} | {fmt(v['routing_correct'])} |"
            )
        L.append("")

        L.append("## Bad Cases — Top 10 最差 MRR")
        L.append("")
        valid = [r for r in records if r.get("metrics")]
        bad = sorted(valid, key=lambda r: r["metrics"].get("mrr@10") or 0.0)[:10]
        for r in bad:
            mrr = fmt(r["metrics"]["mrr@10"])
            L.append(f"- **[{r['question_type']} / 预期 {r['expected_strategy']} → 实际 {r['predicted_strategy']}]** MRR={mrr}")
            L.append(f"  - 问题：{r['question']}")
            L.append(f"  - 预期 node_ids：{r['relevant_node_ids']}")
            L.append(f"  - 检索 node_ids：{r['retrieved_node_ids']}")
        L.append("")

        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(L))


# ==================== main ====================

def main():
    parser = argparse.ArgumentParser(description="Graph RAG 系统评测（baseline / 消融通用）")
    parser.add_argument("--testset", type=str, default="./testset_output/testset.jsonl",
                        help="testset 文件路径（jsonl）")
    parser.add_argument("--output_dir", type=str, default="./eval_output",
                        help="评测输出目录")
    parser.add_argument("--run_id", type=str, default="baseline",
                        help="本次 run 的标识（影响输出文件名）")
    parser.add_argument("--limit", type=int, default=None,
                        help="只评测前 N 条（用于快速验证）")
    parser.add_argument("--skip_generation_eval", action="store_true",
                        help="跳过 Faithfulness / Answer Relevancy（只测检索+路由+延迟）")
    parser.add_argument("--log_level", type=str, default="WARNING")
    parser.add_argument("--config_note", type=str, default="",
                        help="本次 run 的额外说明，写入 summary.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print(f"[1/3] 加载 testset: {args.testset}")
    testset = load_testset(args.testset)
    if args.limit:
        testset = testset[: args.limit]
    print(f"      共 {len(testset)} 条样本\n")

    print("[2/3] 初始化 RAG 系统（首次启动需连接 Neo4j + Milvus + 加载 BGE 嵌入）...")
    rag_system = AdvancedGraphRAGSystem()
    rag_system.initialize_system()
    rag_system.build_knowledge_base()
    if not rag_system.system_ready:
        raise RuntimeError("RAG 系统未就绪")
    print("      系统就绪\n")

    run_config = {
        "use_bm25":         False,
        "use_rrf":          False,
        "use_rerank":       False,
        "topic_alignment":  False,
        "retrieve_top_k":   EvalRunner.RETRIEVE_TOP_K,
        "embedding_model":  rag_system.config.embedding_model,
        "llm_model":        rag_system.config.llm_model,
        "skip_generation_eval": args.skip_generation_eval,
        "note":             args.config_note,
    }

    runner = EvalRunner(
        rag_system=rag_system,
        testset=testset,
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
        run_config=run_config,
        skip_generation_eval=args.skip_generation_eval,
    )

    print(f"[3/3] 开始评测，逐样本结果实时写入 {runner.per_sample_path}")
    summary = runner.run()

    print("\n" + "=" * 60)
    print(f"评测完成: {args.run_id}")
    print("=" * 60)
    agg = summary["aggregated"]
    def _f(x): return f"{x:.4f}" if isinstance(x, float) else "—"
    print(f"  Hit@5            = {_f(agg['retrieval']['hit@5'])}")
    print(f"  Recall@5         = {_f(agg['retrieval']['recall@5'])}")
    print(f"  MRR@10           = {_f(agg['retrieval']['mrr@10'])}")
    print(f"  Routing Accuracy = {_f(agg['routing']['accuracy'])}")
    print(f"  Faithfulness     = {_f(agg['generation']['faithfulness'])}")
    print(f"  Answer Relevancy = {_f(agg['generation']['answer_relevancy'])}")
    print(f"  Latency P50 (ms) = {_f(agg['system']['latency_p50_ms'])}")
    print(f"  Latency P95 (ms) = {_f(agg['system']['latency_p95_ms'])}")
    print()
    print(f"详细报告 → {runner.report_path}")
    print(f"汇总 JSON → {runner.summary_path}")
    print(f"逐样本    → {runner.per_sample_path}")


if __name__ == "__main__":
    main()
