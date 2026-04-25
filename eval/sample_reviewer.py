"""
人工抽样审查辅助函数
随机抽取 N 条样本打印出来，检查 question、ground_truth 是否通顺、是否能从知识库支持
"""

import random
import json
from typing import List, Dict, Any, Optional
from dataclasses import asdict
import logging

from .testset_generator import TestSample

logger = logging.getLogger(__name__)

class SampleReviewer:
    """样本审查器"""
    
    def __init__(self, samples: List[TestSample]):
        """
        初始化审查器
        
        Args:
            samples: 测试样本列表
        """
        self.samples = samples
    
    def random_sample(self, n: int = 10, seed: Optional[int] = None) -> List[TestSample]:
        """
        随机抽取样本
        
        Args:
            n: 抽取数量
            seed: 随机种子
            
        Returns:
            随机抽取的样本列表
        """
        if seed is not None:
            random.seed(seed)
        
        if n >= len(self.samples):
            return self.samples.copy()
        
        return random.sample(self.samples, n)
    
    def print_samples(self, samples: Optional[List[TestSample]] = None, n: int = 10):
        """
        打印样本以供审查
        
        Args:
            samples: 要打印的样本列表（None则随机抽取）
            n: 随机抽取数量（当samples为None时有效）
        """
        if samples is None:
            samples = self.random_sample(n)
        
        print("\n" + "="*80)
        print("测试样本审查报告")
        print("="*80)
        
        for i, sample in enumerate(samples, 1):
            print(f"\n样本 {i}/{len(samples)}:")
            print(f"  问题类型: {sample.question_type}")
            print(f"  难度等级: {sample.difficulty}")
            print(f"  预期策略: {sample.expected_strategy}")
            print(f"  问题: {sample.question}")
            print(f"  答案: {sample.ground_truth}")
            
            if sample.source_node_ids:
                print(f"  源节点ID: {sample.source_node_ids}")
            
            # 提供审查标记
            print(f"\n  审查意见:")
            print(f"    1. 问题是否通顺: [ ]是 [ ]否")
            print(f"    2. 答案是否准确: [ ]是 [ ]否")
            print(f"    3. 是否可从知识库支持: [ ]是 [ ]否")
            print(f"    4. 标注是否合理: [ ]是 [ ]否")
            print(f"    备注: ________________________________")
            
            if i < len(samples):
                print("-" * 80)
        
        print("\n" + "="*80)
        print(f"共审查 {len(samples)} 个样本")
        print("="*80)
    
    def check_quality(self, sample: TestSample) -> Dict[str, Any]:
        """
        检查单个样本的质量
        
        Args:
            sample: 测试样本
            
        Returns:
            质量检查结果字典
        """
        checks = {}
        
        # 1. 检查问题是否通顺
        checks["question_fluent"] = self._check_fluency(sample.question)
        
        # 2. 检查答案是否通顺
        checks["answer_fluent"] = self._check_fluency(sample.ground_truth)
        
        # 3. 检查问题长度
        checks["question_length"] = len(sample.question)
        checks["question_length_ok"] = 5 <= checks["question_length"] <= 100
        
        # 4. 检查答案长度
        checks["answer_length"] = len(sample.ground_truth)
        checks["answer_length_ok"] = 10 <= checks["answer_length"] <= 1000
        
        # 5. 检查标注一致性
        checks["label_consistency"] = self._check_label_consistency(sample)
        
        # 6. 检查是否有明显的生成问题
        checks["no_generation_artifacts"] = self._check_generation_artifacts(sample)
        
        # 计算总体质量分数
        checks["quality_score"] = self._calculate_quality_score(checks)
        
        return checks
    
    def _check_fluency(self, text: str) -> bool:
        """检查文本是否通顺"""
        # 简单规则：检查是否有明显的不通顺模式
        problematic_patterns = [
            "问题：问题：",  # 重复前缀
            "答案：答案：",
            "。。",  # 重复句号
            "？？",
            "！！",
            "  ",  # 双空格
            "\n\n\n",  # 多换行
        ]
        
        for pattern in problematic_patterns:
            if pattern in text:
                return False
        
        # 检查文本是否过短
        if len(text.strip()) < 3:
            return False
        
        # 检查是否包含常见中文标点
        if not any(char in text for char in ["，", "。", "？", "！", "、"]):
            # 如果没有标点，至少要有一定长度
            if len(text) > 20:
                return False
        
        return True
    
    def _check_label_consistency(self, sample: TestSample) -> bool:
        """检查标注一致性"""
        # 检查问题类型和难度的逻辑一致性
        question_type = sample.question_type.lower()
        difficulty = sample.difficulty.lower()
        
        # 简单事实型问题通常是简单的
        if "simple" in question_type and difficulty == "hard":
            return False
        
        # 多跳推理问题通常是中等或困难的
        if "multi_hop" in question_type and difficulty == "easy":
            return False
        
        # 检查预期策略是否合理
        expected_strategy = sample.expected_strategy.lower() if sample.expected_strategy else ""
        if "simple" in question_type and expected_strategy == "graph_rag":
            return False
        
        if "multi_hop" in question_type and expected_strategy == "hybrid_traditional":
            return False
        
        return True
    
    def _check_generation_artifacts(self, sample: TestSample) -> bool:
        """检查生成痕迹"""
        artifacts = [
            "问题：",  # 可能残留的提示词
            "答案：",
            "问：",
            "答：",
            "Q:",
            "A:",
            "###",
            "```",
            "作为一位",
            "请根据",
            "输出格式",
        ]
        
        for artifact in artifacts:
            if artifact in sample.question or artifact in sample.ground_truth:
                # 如果出现在开头，可能是残留
                if sample.question.startswith(artifact) or sample.ground_truth.startswith(artifact):
                    return False
        
        return True
    
    def _calculate_quality_score(self, checks: Dict[str, Any]) -> float:
        """计算质量分数"""
        total_checks = 0
        passed_checks = 0
        
        for key, value in checks.items():
            if key.endswith("_ok") or key.endswith("_fluent") or key.endswith("_consistency") or key.endswith("_artifacts"):
                total_checks += 1
                if value:
                    passed_checks += 1
        
        if total_checks == 0:
            return 0.0
        
        return passed_checks / total_checks
    
    def batch_quality_check(self, samples: Optional[List[TestSample]] = None, n: int = 50) -> Dict[str, Any]:
        """
        批量质量检查
        
        Args:
            samples: 要检查的样本列表（None则随机抽取）
            n: 随机抽取数量（当samples为None时有效）
            
        Returns:
            批量检查结果
        """
        if samples is None:
            samples = self.random_sample(n)
        
        results = []
        total_score = 0
        
        for sample in samples:
            checks = self.check_quality(sample)
            results.append({
                "sample": sample.question[:50] + "..." if len(sample.question) > 50 else sample.question,
                "checks": checks,
                "quality_score": checks["quality_score"]
            })
            total_score += checks["quality_score"]
        
        avg_score = total_score / len(samples) if samples else 0
        
        return {
            "total_samples": len(samples),
            "average_quality_score": avg_score,
            "results": results[:10],  # 只保留前10个详细结果
        }
    
    def print_statistics(self):
        """打印样本统计信息"""
        if not self.samples:
            print("没有样本可分析")
            return
        
        print("\n" + "="*80)
        print("测试样本统计信息")
        print("="*80)
        
        # 问题类型分布
        type_counts = {}
        for sample in self.samples:
            q_type = sample.question_type
            type_counts[q_type] = type_counts.get(q_type, 0) + 1
        
        print(f"\n问题类型分布 (共 {len(self.samples)} 个样本):")
        for q_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(self.samples) * 100
            print(f"  {q_type}: {count} ({percentage:.1f}%)")
        
        # 难度分布
        difficulty_counts = {}
        for sample in self.samples:
            diff = sample.difficulty
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        
        print(f"\n难度分布:")
        for diff, count in sorted(difficulty_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(self.samples) * 100
            print(f"  {diff}: {count} ({percentage:.1f}%)")
        
        # 策略分布
        strategy_counts = {}
        for sample in self.samples:
            strategy = sample.expected_strategy or "unknown"
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        print(f"\n预期策略分布:")
        for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(self.samples) * 100
            print(f"  {strategy}: {count} ({percentage:.1f}%)")
        
        # 问题长度统计
        question_lengths = [len(sample.question) for sample in self.samples]
        if question_lengths:
            avg_len = sum(question_lengths) / len(question_lengths)
            min_len = min(question_lengths)
            max_len = max(question_lengths)
            print(f"\n问题长度统计:")
            print(f"  平均长度: {avg_len:.1f} 字符")
            print(f"  最短: {min_len} 字符")
            print(f"  最长: {max_len} 字符")
        
        # 答案长度统计
        answer_lengths = [len(sample.ground_truth) for sample in self.samples]
        if answer_lengths:
            avg_len = sum(answer_lengths) / len(answer_lengths)
            min_len = min(answer_lengths)
            max_len = max(answer_lengths)
            print(f"\n答案长度统计:")
            print(f"  平均长度: {avg_len:.1f} 字符")
            print(f"  最短: {min_len} 字符")
            print(f"  最长: {max_len} 字符")
        
        # 质量检查（抽样）
        if len(self.samples) > 0:
            quality_result = self.batch_quality_check(n=min(20, len(self.samples)))
            print(f"\n抽样质量检查 ({quality_result['total_samples']} 个样本):")
            print(f"  平均质量分数: {quality_result['average_quality_score']:.2f}")
        
        print("\n" + "="*80)
    
    def export_review_report(self, output_path: str = "review_report.json"):
        """
        导出审查报告
        
        Args:
            output_path: 输出文件路径
        """
        report = {
            "total_samples": len(self.samples),
            "statistics": {},
            "random_samples": [],
            "quality_check": self.batch_quality_check(n=min(20, len(self.samples)))
        }
        
        # 统计信息
        type_counts = {}
        difficulty_counts = {}
        strategy_counts = {}
        
        for sample in self.samples:
            type_counts[sample.question_type] = type_counts.get(sample.question_type, 0) + 1
            difficulty_counts[sample.difficulty] = difficulty_counts.get(sample.difficulty, 0) + 1
            strategy_counts[sample.expected_strategy or "unknown"] = strategy_counts.get(sample.expected_strategy or "unknown", 0) + 1
        
        report["statistics"] = {
            "type_distribution": type_counts,
            "difficulty_distribution": difficulty_counts,
            "strategy_distribution": strategy_counts
        }
        
        # 随机样本（用于人工审查）
        random_samples = self.random_sample(min(20, len(self.samples)))
        report["random_samples"] = [asdict(sample) for sample in random_samples]
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"审查报告已保存到 {output_path}")
        return report

def main():
    """命令行入口点"""
    import argparse
    
    parser = argparse.ArgumentParser(description="测试样本审查工具")
    parser.add_argument("--input", type=str, required=True, help="输入测试集文件路径（JSONL/JSON）")
    parser.add_argument("--num_samples", type=int, default=10, help="随机抽取样本数量")
    parser.add_argument("--output_report", type=str, help="输出审查报告路径")
    parser.add_argument("--check_quality", action="store_true", help="执行质量检查")
    
    args = parser.parse_args()
    
    # 加载测试集
    from .utils import load_testset
    samples = load_testset(args.input)
    
    # 创建审查器
    reviewer = SampleReviewer(samples)
    
    # 打印统计信息
    reviewer.print_statistics()
    
    # 打印随机样本以供审查
    reviewer.print_samples(n=args.num_samples)
    
    # 执行质量检查
    if args.check_quality:
        result = reviewer.batch_quality_check(n=min(20, len(samples)))
        print(f"\n质量检查结果:")
        print(f"  平均质量分数: {result['average_quality_score']:.2f}")
    
    # 导出报告
    if args.output_report:
        reviewer.export_review_report(args.output_report)

if __name__ == "__main__":
    main()