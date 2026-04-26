"""
基于已有 per-sample.jsonl + 修订后的 testset 重算评测指标

适用场景：testset 标签变更但系统输出未变（如本次 v1 strict → v2 corrected
的 expected_strategy 修订）。

不调用 RAG 系统，纯文件 IO 与 numpy 计算。直接复用 retrieved/answer/latency
等已有结果，只把每条样本的 expected_strategy 同步为新 testset 的值，进而
重算 routing_correct 与按 expected_strategy 分组的指标。
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

_C9_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _C9_DIR not in sys.path:
    sys.path.insert(0, _C9_DIR)

from eval.metrics_aggregator import aggregate_metrics, write_markdown_report


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_testset_simple(path: Path) -> List[Dict[str, Any]]:
    """轻量 testset 加载，不依赖 testset_generator/utils（避免拉入 neo4j 链）"""
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"只支持 .jsonl 或 .json，收到: {path.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="从已有 per-sample.jsonl + 修订后的 testset 重算评测指标"
    )
    parser.add_argument("--per_sample", type=str, required=True,
                        help="原 per-sample.jsonl 路径（如 baseline_v1 的）")
    parser.add_argument("--testset", type=str, default="./testset_output/testset.jsonl",
                        help="修订后的 testset 路径")
    parser.add_argument("--output_dir", type=str, default="./eval_output",
                        help="新 summary / report 写到这里")
    parser.add_argument("--run_id", type=str, default="baseline_v2",
                        help="新 run 标识")
    parser.add_argument("--config_note", type=str,
                        default="recomputed from baseline_v1 with relabeled testset (no LLM re-run)")
    args = parser.parse_args()

    per_sample_path = Path(args.per_sample)
    testset_path = Path(args.testset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] 加载原 per-sample: {per_sample_path}")
    records = load_jsonl(per_sample_path)
    print(f"      共 {len(records)} 条")

    print(f"[2/4] 加载修订后 testset: {testset_path}")
    testset = load_testset_simple(testset_path)
    print(f"      共 {len(testset)} 条")
    q2new_strategy = {s["question"]: s.get("expected_strategy") for s in testset}

    print(f"[3/4] 同步 expected_strategy + 重算 routing_correct")
    label_changed = 0
    routing_flipped = 0
    unmatched: List[str] = []

    for r in records:
        q = r.get("question")
        if q not in q2new_strategy:
            unmatched.append(q)
            continue

        new_exp = q2new_strategy[q]
        old_exp = r.get("expected_strategy")
        if old_exp != new_exp:
            label_changed += 1
        r["expected_strategy"] = new_exp

        pred = r.get("predicted_strategy")
        if r.get("metrics") is not None and pred is not None:
            old_correct = r["metrics"].get("routing_correct")
            new_correct = int(pred == new_exp)
            if old_correct != new_correct:
                routing_flipped += 1
            r["metrics"]["routing_correct"] = new_correct

    print(f"      标签变化: {label_changed} 条")
    print(f"      routing_correct 翻转: {routing_flipped} 条")
    if unmatched:
        print(f"      ⚠️ {len(unmatched)} 条 per-sample 在新 testset 中找不到匹配的 question")

    run_config = {
        "use_bm25":         False,
        "use_rrf":          False,
        "use_rerank":       False,
        "topic_alignment":  False,
        "note":             args.config_note,
        "source_per_sample": str(per_sample_path),
        "source_testset":    str(testset_path),
    }

    print(f"[4/4] 写入 {args.run_id} 三件套到 {output_dir}/")

    new_per_sample = output_dir / f"{args.run_id}_per_sample.jsonl"
    with open(new_per_sample, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = aggregate_metrics(records, args.run_id, run_config)
    summary_path = output_dir / f"{args.run_id}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    report_path = output_dir / f"{args.run_id}_report.md"
    write_markdown_report(summary, records, report_path)

    print()
    print("=" * 60)
    print(f"完成: {args.run_id}")
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
    print(f"summary    → {summary_path}")
    print(f"per-sample → {new_per_sample}")
    print(f"report     → {report_path}")


if __name__ == "__main__":
    main()
