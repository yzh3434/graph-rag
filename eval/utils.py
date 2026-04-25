"""
工具函数：保存/加载测试集、统计、验证、进度管理
"""

import os
import csv
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from .testset_generator import TestSample

logger = logging.getLogger(__name__)


def save_testset(samples: List[TestSample], output_dir: str = "./testset_output") -> Dict[str, str]:
    """保存测试集到 JSONL / JSON / CSV 三种格式，并输出统计报告"""
    os.makedirs(output_dir, exist_ok=True)
    samples_dict = [s.to_dict() for s in samples]

    jsonl_path = os.path.join(output_dir, "testset.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in samples_dict:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"已保存 JSONL: {jsonl_path} ({len(samples)} 条)")

    json_path = os.path.join(output_dir, "testset.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples_dict, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存 JSON: {json_path}")

    csv_path = os.path.join(output_dir, "testset.csv")
    fieldnames = [
        "question", "ground_truth", "question_type", "difficulty",
        "expected_strategy", "source_node_ids", "metadata",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples_dict:
            row = {
                "question": s.get("question", ""),
                "ground_truth": s.get("ground_truth", ""),
                "question_type": s.get("question_type", ""),
                "difficulty": s.get("difficulty", ""),
                "expected_strategy": s.get("expected_strategy", ""),
                "source_node_ids": ";".join(s.get("source_node_ids") or []),
                "metadata": json.dumps(s.get("metadata") or {}, ensure_ascii=False),
            }
            writer.writerow(row)
    logger.info(f"已保存 CSV: {csv_path}")

    stats_path = os.path.join(output_dir, "statistics.json")
    stats = calculate_statistics(samples)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存统计信息: {stats_path}")

    return {
        "jsonl": jsonl_path,
        "json": json_path,
        "csv": csv_path,
        "statistics": stats_path,
    }


def load_testset(file_path: str) -> List[TestSample]:
    """从 JSONL / JSON / CSV 加载测试集"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".jsonl":
        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(TestSample.from_dict(json.loads(line)))
        logger.info(f"从 JSONL 加载 {len(samples)} 条")
        return samples

    if ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = [TestSample.from_dict(d) for d in data]
        logger.info(f"从 JSON 加载 {len(samples)} 条")
        return samples

    if ext == ".csv":
        samples = []
        with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = dict(row)
                sn = d.get("source_node_ids") or ""
                d["source_node_ids"] = sn.split(";") if sn else []
                md = d.get("metadata") or "{}"
                try:
                    d["metadata"] = json.loads(md)
                except json.JSONDecodeError:
                    d["metadata"] = {}
                samples.append(TestSample.from_dict(d))
        logger.info(f"从 CSV 加载 {len(samples)} 条")
        return samples

    raise ValueError(f"不支持的文件格式: {ext}（支持 .jsonl / .json / .csv）")


def calculate_statistics(samples: List[TestSample]) -> Dict[str, Any]:
    """计算测试集统计信息"""
    if not samples:
        return {}

    type_counts: Dict[str, int] = {}
    diff_counts: Dict[str, int] = {}
    strategy_counts: Dict[str, int] = {}

    for s in samples:
        type_counts[s.question_type] = type_counts.get(s.question_type, 0) + 1
        diff_counts[s.difficulty] = diff_counts.get(s.difficulty, 0) + 1
        key = s.expected_strategy or "unknown"
        strategy_counts[key] = strategy_counts.get(key, 0) + 1

    total = len(samples)

    def _to_dist(counts: Dict[str, int]) -> Dict[str, Dict[str, float]]:
        return {
            k: {"count": v, "percentage": v / total * 100}
            for k, v in counts.items()
        }

    q_lens = [len(s.question) for s in samples]
    a_lens = [len(s.ground_truth) for s in samples]

    def _stats(arr):
        mean = sum(arr) / len(arr)
        var = sum((x - mean) ** 2 for x in arr) / len(arr)
        return {"mean": mean, "min": min(arr), "max": max(arr), "std": var ** 0.5}

    fluent_q = sum(1 for s in samples if len(s.question) >= 5 and ("？" in s.question or "?" in s.question))
    fluent_a = sum(1 for s in samples if len(s.ground_truth) >= 10)
    valid_labels = sum(1 for s in samples if s.question_type and s.difficulty)

    return {
        "total_samples": total,
        "question_type_distribution": _to_dist(type_counts),
        "difficulty_distribution": _to_dist(diff_counts),
        "expected_strategy_distribution": _to_dist(strategy_counts),
        "length_statistics": {
            "question_length": _stats(q_lens),
            "answer_length": _stats(a_lens),
        },
        "quality_metrics": {
            "fluent_questions_ratio": fluent_q / total * 100,
            "fluent_answers_ratio": fluent_a / total * 100,
            "valid_labels_ratio": valid_labels / total * 100,
        },
    }


def validate_testset(samples: List[TestSample]) -> Dict[str, Any]:
    """验证测试集基本完整性"""
    result = {"passed": True, "errors": [], "warnings": [], "statistics": {}}

    if not samples:
        result["passed"] = False
        result["errors"].append("测试集为空")
        return result

    required_fields = ["question", "ground_truth", "question_type", "difficulty"]
    valid_types = {
        "simple_fact", "attribute_query", "step_by_step",
        "entity_relation", "multi_hop", "comparison", "causal",
    }
    valid_difficulties = {"easy", "medium", "hard"}
    valid_strategies = {"hybrid_traditional", "graph_rag", "combined", None, ""}

    for i, sample in enumerate(samples):
        sd = sample.to_dict()

        for field_name in required_fields:
            if not sd.get(field_name):
                result["passed"] = False
                result["errors"].append(f"样本 {i} 缺少必需字段: {field_name}")

        if len(sample.question) < 3:
            result["warnings"].append(f"样本 {i} 问题过短: '{sample.question}'")
        if len(sample.ground_truth) < 5:
            preview = sample.ground_truth[:50]
            result["warnings"].append(f"样本 {i} 答案过短: '{preview}...'")
        if "？" not in sample.question and "?" not in sample.question:
            result["warnings"].append(f"样本 {i} 问题缺少问号: '{sample.question}'")

        if sample.question_type not in valid_types:
            result["warnings"].append(
                f"样本 {i} 问题类型无效: '{sample.question_type}'"
            )
        if sample.difficulty not in valid_difficulties:
            result["warnings"].append(
                f"样本 {i} 难度无效: '{sample.difficulty}'"
            )
        if sample.expected_strategy not in valid_strategies:
            result["warnings"].append(
                f"样本 {i} 预期策略无效: '{sample.expected_strategy}'"
            )

    result["statistics"] = calculate_statistics(samples)
    return result


def save_progress(samples: List[TestSample], output_dir: str, progress_file: str = "progress.json"):
    """保存生成进度（用于中断恢复）"""
    os.makedirs(output_dir, exist_ok=True)
    progress_path = os.path.join(output_dir, progress_file)
    progress_data = {
        "generated_count": len(samples),
        "samples": [s.to_dict() for s in samples],
        "timestamp": datetime.now().isoformat(),
    }
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress_data, f, ensure_ascii=False, indent=2)
    logger.info(f"进度已保存: {progress_path}（{len(samples)} 条）")


def load_progress(output_dir: str, progress_file: str = "progress.json") -> Optional[List[TestSample]]:
    """加载生成进度"""
    progress_path = os.path.join(output_dir, progress_file)
    if not os.path.exists(progress_path):
        return None
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [TestSample.from_dict(d) for d in data.get("samples", [])]
    except Exception as e:
        logger.error(f"加载进度失败: {e}")
        return None


def print_sample_preview(samples: List[TestSample], n: int = 5):
    """打印样本预览"""
    n = min(n, len(samples))
    print(f"\n测试集预览（共 {len(samples)} 条，显示前 {n} 条）:")
    print("=" * 80)
    for i, s in enumerate(samples[:n], 1):
        print(f"\n样本 {i}")
        print(f"  问题类型: {s.question_type}")
        print(f"  难度: {s.difficulty}")
        print(f"  预期策略: {s.expected_strategy}")
        print(f"  问题: {s.question}")
        answer_preview = s.ground_truth if len(s.ground_truth) <= 100 else s.ground_truth[:100] + "..."
        print(f"  答案: {answer_preview}")
        if i < n:
            print("-" * 40)
