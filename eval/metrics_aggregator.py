"""
评测指标聚合 + markdown 报告生成

eval_runner.py 跑完一轮评测后调用这里得到 summary 与报告；
recompute_metrics.py 标签变更后基于已有 per-sample 重算时也调用同一组函数，
保证两条路径产出的数字与格式完全一致。

不依赖 RAG 系统模块，纯 numpy + 标准库。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

import numpy as np


# ==================== 内部 helper ====================

def _mean(records: List[Dict[str, Any]], metric: str) -> Optional[float]:
    vals = [r["metrics"][metric] for r in records
            if r.get("metrics") and r["metrics"].get(metric) is not None]
    return float(np.mean(vals)) if vals else None


def _percentile(records: List[Dict[str, Any]], metric: str, p: float) -> Optional[float]:
    vals = [r["metrics"][metric] for r in records
            if r.get("metrics") and r["metrics"].get(metric) is not None]
    return float(np.percentile(vals, p)) if vals else None


# ==================== 对外接口 ====================

def group_metrics(records: List[Dict[str, Any]], group_key: str) -> Dict[str, Any]:
    """按某个字段（question_type / expected_strategy / difficulty）分组聚合"""
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        buckets[r.get(group_key, "unknown")].append(r)

    out: Dict[str, Any] = {}
    for group, rs in buckets.items():
        out[group] = {
            "n":                len(rs),
            "hit@5":            _mean(rs, "hit@5"),
            "recall@5":         _mean(rs, "recall@5"),
            "mrr@10":           _mean(rs, "mrr@10"),
            "routing_correct":  _mean(rs, "routing_correct"),
            "faithfulness":     _mean(rs, "faithfulness"),
            "answer_relevancy": _mean(rs, "answer_relevancy"),
            "latency_ms_mean":  _mean(rs, "latency_ms"),
        }
    return out


def aggregate_metrics(
    records: List[Dict[str, Any]],
    run_id: str,
    run_config: Dict[str, Any],
) -> Dict[str, Any]:
    """从 per-sample 记录列表生成完整 summary（与 eval_runner 输出 schema 一致）"""
    valid = [r for r in records if r.get("metrics") is not None]
    n_total = len(records)
    n_valid = len(valid)

    aggregated = {
        "retrieval": {
            "hit@5":    _mean(valid, "hit@5"),
            "recall@5": _mean(valid, "recall@5"),
            "mrr@10":   _mean(valid, "mrr@10"),
        },
        "routing": {
            "accuracy": _mean(valid, "routing_correct"),
        },
        "generation": {
            "faithfulness":     _mean(valid, "faithfulness"),
            "answer_relevancy": _mean(valid, "answer_relevancy"),
        },
        "system": {
            "latency_p50_ms":          _percentile(valid, "latency_ms", 50),
            "latency_p95_ms":          _percentile(valid, "latency_ms", 95),
            "latency_retrieve_p50_ms": _percentile(valid, "latency_retrieve_ms", 50),
            "latency_generate_p50_ms": _percentile(valid, "latency_generate_ms", 50),
        },
    }

    return {
        "run_id":    run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config":    run_config,
        "n_total":   n_total,
        "n_valid":   n_valid,
        "n_failed":  n_total - n_valid,
        "aggregated":              aggregated,
        "by_question_type":        group_metrics(valid, "question_type"),
        "by_expected_strategy":    group_metrics(valid, "expected_strategy"),
        "by_difficulty":           group_metrics(valid, "difficulty"),
    }


def write_markdown_report(
    summary: Dict[str, Any],
    records: List[Dict[str, Any]],
    report_path: Path,
):
    """把 summary + 部分 per-sample 写成人类可读的 markdown 报告（含 bad case）"""
    agg = summary["aggregated"]

    def fmt(x):
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x) if x is not None else "—"

    L: List[str] = []
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
        L.append(
            f"- **[{r['question_type']} / 预期 {r['expected_strategy']} → 实际 {r['predicted_strategy']}]** MRR={mrr}"
        )
        L.append(f"  - 问题：{r['question']}")
        L.append(f"  - 预期 node_ids：{r['relevant_node_ids']}")
        L.append(f"  - 检索 node_ids：{r['retrieved_node_ids']}")
    L.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
