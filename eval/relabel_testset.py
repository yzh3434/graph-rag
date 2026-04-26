"""
testset 标签修订工具（一次性脚本）

不重新生成测试集，只把每条样本的 expected_strategy 字段按下方
TYPE_TO_STRATEGY_VALUES 映射重新打标。

此映射必须与 eval/testset_generator.py 的 TYPE_TO_STRATEGY 保持一致；
未来调整其一时记得同步另一个。

先把 testset_output/ 下四件套备份到 testset_output/archive_v1_strict/，
然后重写 .jsonl / .json / .csv / statistics.json。

只用标准库，不依赖 testset_generator / utils（避免拉入 neo4j、openai 等）。
"""

import os
import csv
import json
import shutil
import argparse
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any


TYPE_TO_STRATEGY_VALUES: Dict[str, str] = {
    "simple_fact":     "hybrid_traditional",
    "attribute_query": "hybrid_traditional",
    "step_by_step":    "hybrid_traditional",
    "entity_relation": "hybrid_traditional",
    "multi_hop":       "graph_rag",
    "comparison":      "combined",
    "causal":          "hybrid_traditional",
}

BACKUP_FILES = ["testset.jsonl", "testset.json", "testset.csv", "statistics.json"]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def backup_existing(output_dir: Path, backup_dir: Path) -> int:
    backup_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for fname in BACKUP_FILES:
        src = output_dir / fname
        if src.exists():
            shutil.copy2(src, backup_dir / fname)
            moved += 1
            print(f"  备份 {src.name} → {backup_dir.name}/")
    return moved


def relabel(samples: List[Dict[str, Any]]):
    changed_count = 0
    breakdown: Dict[str, Dict[str, Any]] = {}
    for s in samples:
        q_type = s.get("question_type")
        if q_type not in TYPE_TO_STRATEGY_VALUES:
            print(f"  ⚠️ 跳过未知 question_type: {q_type}")
            continue
        new_strategy = TYPE_TO_STRATEGY_VALUES[q_type]
        old_strategy = s.get("expected_strategy")
        if old_strategy != new_strategy:
            entry = breakdown.setdefault(
                q_type, {"old": old_strategy, "new": new_strategy, "count": 0}
            )
            entry["count"] += 1
            s["expected_strategy"] = new_strategy
            changed_count += 1
    return changed_count, breakdown


def write_jsonl(samples: List[Dict[str, Any]], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def write_json(samples: List[Dict[str, Any]], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def write_csv(samples: List[Dict[str, Any]], path: Path):
    fieldnames = [
        "question", "ground_truth", "question_type", "difficulty",
        "expected_strategy", "source_node_ids", "metadata",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            row = {
                "question":          s.get("question", ""),
                "ground_truth":      s.get("ground_truth", ""),
                "question_type":     s.get("question_type", ""),
                "difficulty":        s.get("difficulty", ""),
                "expected_strategy": s.get("expected_strategy", ""),
                "source_node_ids":   ";".join(s.get("source_node_ids") or []),
                "metadata":          json.dumps(s.get("metadata") or {}, ensure_ascii=False),
            }
            writer.writerow(row)


def calculate_statistics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {}
    total = len(samples)
    type_counts     = Counter(s.get("question_type") for s in samples)
    diff_counts     = Counter(s.get("difficulty") for s in samples)
    strategy_counts = Counter(s.get("expected_strategy") or "unknown" for s in samples)

    def _to_dist(counts):
        return {k: {"count": v, "percentage": v / total * 100} for k, v in counts.items()}

    q_lens = [len(s.get("question", "")) for s in samples]
    a_lens = [len(s.get("ground_truth", "")) for s in samples]

    def _stats(arr):
        mean = sum(arr) / len(arr)
        var  = sum((x - mean) ** 2 for x in arr) / len(arr)
        return {"mean": mean, "min": min(arr), "max": max(arr), "std": var ** 0.5}

    fluent_q = sum(
        1 for s in samples
        if len(s.get("question", "")) >= 5
        and ("？" in s.get("question", "") or "?" in s.get("question", ""))
    )
    fluent_a = sum(1 for s in samples if len(s.get("ground_truth", "")) >= 10)
    valid_labels = sum(1 for s in samples if s.get("question_type") and s.get("difficulty"))

    return {
        "total_samples": total,
        "question_type_distribution":      _to_dist(type_counts),
        "difficulty_distribution":         _to_dist(diff_counts),
        "expected_strategy_distribution":  _to_dist(strategy_counts),
        "length_statistics": {
            "question_length": _stats(q_lens),
            "answer_length":   _stats(a_lens),
        },
        "quality_metrics": {
            "fluent_questions_ratio": fluent_q / total * 100,
            "fluent_answers_ratio":   fluent_a / total * 100,
            "valid_labels_ratio":     valid_labels / total * 100,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="按当前 TYPE_TO_STRATEGY_VALUES 重新打标 testset")
    parser.add_argument("--testset_dir", type=str, default="./testset_output")
    parser.add_argument("--backup_subdir", type=str, default="archive_v1_strict")
    parser.add_argument("--source", type=str, default="testset.jsonl",
                        help="作为标签来源的 .jsonl 文件名")
    args = parser.parse_args()

    testset_dir = Path(args.testset_dir)
    backup_dir  = testset_dir / args.backup_subdir
    source_path = testset_dir / args.source

    if not source_path.exists():
        raise FileNotFoundError(f"找不到来源文件: {source_path}")

    print("=" * 60)
    print("Step 1  备份现有 testset 到归档子目录")
    print("=" * 60)
    moved = backup_existing(testset_dir, backup_dir)
    print(f"  共备份 {moved} 个文件\n")

    print("=" * 60)
    print(f"Step 2  从 {source_path.name} 加载样本")
    print("=" * 60)
    samples = load_jsonl(source_path)
    print(f"  共 {len(samples)} 条\n")

    print("=" * 60)
    print("Step 3  按当前 TYPE_TO_STRATEGY_VALUES 重新打标")
    print("=" * 60)
    print("  当前映射：")
    for qt, st in TYPE_TO_STRATEGY_VALUES.items():
        print(f"    {qt:20s} → {st}")
    print()

    changed, breakdown = relabel(samples)
    print(f"  修改了 {changed} 条样本的 expected_strategy")
    if breakdown:
        for q_type, info in breakdown.items():
            print(f"    [{q_type}] {info['old']} → {info['new']} ({info['count']} 条)")
    print()

    print("=" * 60)
    print("Step 4  写回 testset_output/（覆盖 .jsonl / .json / .csv / statistics.json）")
    print("=" * 60)
    paths = {
        "jsonl":      testset_dir / "testset.jsonl",
        "json":       testset_dir / "testset.json",
        "csv":        testset_dir / "testset.csv",
        "statistics": testset_dir / "statistics.json",
    }
    write_jsonl(samples, paths["jsonl"])
    write_json(samples,  paths["json"])
    write_csv(samples,   paths["csv"])
    stats = calculate_statistics(samples)
    with open(paths["statistics"], "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    for fmt, p in paths.items():
        print(f"  {fmt:10s} → {p}")
    print()

    print("=" * 60)
    print("完成。新的策略分布：")
    print("=" * 60)
    dist = Counter(s.get("expected_strategy") for s in samples)
    for k, v in sorted(dist.items()):
        print(f"  {k:25s} {v:>4d}  ({v/len(samples)*100:.1f}%)")


if __name__ == "__main__":
    main()
