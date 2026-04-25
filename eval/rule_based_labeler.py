"""
基于规则的兜底标注函数
当自动生成 question_type / difficulty 不稳定时，使用规则进行标注
"""

import re
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class QuestionType(Enum):
    """问题类型枚举"""
    SIMPLE_FACT = "simple_fact"  # 简单事实型问题（单跳）
    ATTRIBUTE_QUERY = "attribute_query"  # 菜品/食材属性查询
    STEP_BY_STEP = "step_by_step"  # 制作步骤型问题
    ENTITY_RELATION = "entity_relation"  # 实体关系型问题
    MULTI_HOP = "multi_hop"  # 多跳推理问题
    COMPARISON = "comparison"  # 对比型问题
    CAUSAL = "causal"  # 原因/为什么类问题

class DifficultyLevel(Enum):
    """难度等级枚举"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class ExpectedStrategy(Enum):
    """预期检索策略枚举"""
    TRADITIONAL = "traditional"  # 传统混合检索
    GRAPH_RAG = "graph_rag"  # 图RAG检索
    COMBINED = "combined"  # 组合检索

class RuleBasedLabeler:
    """基于规则的标注器"""
    
    def __init__(self):
        # 问题类型关键词
        self.type_keywords = {
            QuestionType.SIMPLE_FACT: [
                r"什么是", r"是什么", r"属于", r"分类", r"类型", r"哪个菜系",
                r"哪里", r"何时", r"谁", r"多少", r"几[个种]", r"简单介绍",
                r"简述", r"概括", r"特点"
            ],
            QuestionType.ATTRIBUTE_QUERY: [
                r"食材", r"配料", r"原料", r"需要.*什么", r"准备.*什么",
                r"主要.*是", r"含有", r"包含", r"成分", r"用量", r"多少克",
                r"几[斤两克]", r"时间", r"分钟", r"小时", r"难度", r"星级",
                r"份量", r"几人份", r"准备时间", r"烹饪时间"
            ],
            QuestionType.STEP_BY_STEP: [
                r"怎么做", r"如何制作", r"制作步骤", r"做法", r"步骤",
                r"流程", r"过程", r"详细步骤", r"分步", r"一步一步",
                r"具体做法", r"操作步骤", r"制作方法", r"烹饪方法"
            ],
            QuestionType.ENTITY_RELATION: [
                r"搭配", r"配什么", r"和.*一起", r"与.*搭配", r"关系",
                r"关联", r"联系", r"配合", r"组合", r"配对", r"适合配",
                r"可以和.*一起", r"常见搭配", r"传统搭配"
            ],
            QuestionType.MULTI_HOP: [
                r"并且", r"而且", r"同时", r"还要", r"既要.*又要",
                r"不仅.*还", r"除了.*还要", r"适合.*并且", r"要求.*并且",
                r"需要.*并且", r"既要.*也要", r"既要.*还要", r"如果.*那么",
                r"假如.*可以", r"要是.*能", r"哪些.*又.*", r"什么.*还.*"
            ],
            QuestionType.COMPARISON: [
                r"比较", r"对比", r"区别", r"不同", r"差异", r"相比",
                r"相较于", r"差别", r"异同", r"哪个更好", r"哪个更",
                r"更好", r"更优", r"优劣", r"优缺点", r"优势", r"劣势"
            ],
            QuestionType.CAUSAL: [
                r"为什么", r"为何", r"原因", r"缘故", r"为何要",
                r"为什么要", r"为何需要", r"原因是什么", r"为什么用",
                r"为何用", r"为什么选择", r"为何选择", r"为什么是",
                r"为何是", r"原理", r"科学依据", r"道理"
            ]
        }
        
        # 难度判断规则
        self.difficulty_rules = {
            DifficultyLevel.EASY: [
                (r"简单", 1.0),
                (r"基础", 0.8),
                (r"基本", 0.7),
                (r"入门", 0.9),
                (r"初学", 0.8),
                (r"常见", 0.6),
                (r"普通", 0.5),
                (r"一般", 0.5),
                (r"什么[是属]", 0.4),
                (r"哪[里个]", 0.4),
                (r"谁", 0.3),
                (r"何时", 0.3),
                (r"简述", 0.4),
                (r"概括", 0.4),
            ],
            DifficultyLevel.HARD: [
                (r"复杂", 1.0),
                (r"困难", 0.9),
                (r"难", 0.8),
                (r"高级", 0.8),
                (r"专业", 0.7),
                (r"深入", 0.7),
                (r"详细分析", 0.9),
                (r"对比", 0.8),
                (r"比较", 0.8),
                (r"为什么", 0.6),
                (r"原因", 0.6),
                (r"原理", 0.7),
                (r"科学依据", 0.9),
                (r"多跳", 0.9),
                (r"推理", 0.8),
                (r"推断", 0.8),
                (r"并且", 0.7),
                (r"而且", 0.7),
                (r"同时", 0.7),
                (r"还要", 0.7),
            ]
        }
        
        # 策略判断规则
        self.strategy_rules = {
            ExpectedStrategy.TRADITIONAL: [  # 简单问题 -> traditional
                (r"什么[是属]", 1.0),
                (r"哪[里个]", 0.9),
                (r"谁", 0.8),
                (r"何时", 0.8),
                (r"多少", 0.7),
                (r"几[个种]", 0.7),
                (r"简单介绍", 0.9),
                (r"简述", 0.9),
                (r"概括", 0.9),
                (r"特点", 0.8),
                (r"食材", 0.6),
                (r"配料", 0.6),
                (r"时间", 0.6),
                (r"分钟", 0.5),
                (r"小时", 0.5),
                (r"怎么做", 0.5),
                (r"做法", 0.5),
            ],
            ExpectedStrategy.GRAPH_RAG: [  # 关系/多跳 -> graph_rag
                (r"搭配", 0.9),
                (r"配什么", 0.9),
                (r"和.*一起", 0.8),
                (r"关系", 0.9),
                (r"关联", 0.9),
                (r"联系", 0.9),
                (r"多跳", 1.0),
                (r"推理", 0.9),
                (r"推断", 0.9),
                (r"并且", 0.8),
                (r"而且", 0.8),
                (r"同时", 0.8),
                (r"还要", 0.8),
                (r"既要.*又要", 0.9),
                (r"不仅.*还", 0.9),
                (r"为什么", 0.7),
                (r"原因", 0.7),
                (r"比较", 0.6),
                (r"对比", 0.6),
                (r"区别", 0.6),
            ],
            ExpectedStrategy.COMBINED: [  # 复杂综合 -> combined
                (r"复杂", 0.9),
                (r"综合", 0.9),
                (r"全面", 0.8),
                (r"详细分析", 0.8),
                (r"深入探讨", 0.9),
                (r"系统介绍", 0.8),
                (r"综合评价", 0.9),
                (r"综合分析", 0.9),
                (r"既要.*又要.*还要", 0.9),
                (r"不仅.*而且.*还", 0.9),
                (r"如果.*那么.*并且", 0.9),
            ]
        }
    
    def label_question_type(self, question: str, answer: Optional[str] = None) -> str:
        """
        标注问题类型
        
        Args:
            question: 问题文本
            answer: 答案文本（可选，用于辅助判断）
            
        Returns:
            问题类型字符串
        """
        question_lower = question.lower()
        
        # 计算每种类型的匹配分数
        type_scores = {}
        
        for q_type, keywords in self.type_keywords.items():
            score = 0
            for pattern in keywords:
                matches = re.findall(pattern, question_lower)
                if matches:
                    score += len(matches) * 0.5  # 每个匹配加0.5分
            
            # 额外基于答案的判断（如果有）
            if answer:
                # 如果答案包含步骤信息，可能是步骤型问题
                if "步骤" in answer or "第一步" in answer or "1." in answer:
                    if q_type == QuestionType.STEP_BY_STEP:
                        score += 1.0
                
                # 如果答案包含比较信息，可能是对比型问题
                if "相比" in answer or "区别" in answer or "不同" in answer:
                    if q_type == QuestionType.COMPARISON:
                        score += 1.0
            
            type_scores[q_type] = score
        
        # 找出最高分的类型
        if not type_scores:
            return QuestionType.SIMPLE_FACT.value
        
        max_type = max(type_scores.items(), key=lambda x: x[1])
        
        # 如果最高分小于阈值，返回默认类型
        if max_type[1] < 0.5:
            return QuestionType.SIMPLE_FACT.value
        
        logger.debug(f"问题类型标注: '{question}' -> {max_type[0].value} (分数: {max_type[1]:.2f})")
        return max_type[0].value
    
    def label_difficulty(self, question: str, answer: Optional[str] = None) -> str:
        """
        标注难度等级
        
        Args:
            question: 问题文本
            answer: 答案文本（可选，用于辅助判断）
            
        Returns:
            难度等级字符串
        """
        question_lower = question.lower()
        
        # 计算难度分数
        easy_score = 0
        hard_score = 0
        
        # 计算简单难度分数
        for pattern, weight in self.difficulty_rules[DifficultyLevel.EASY]:
            matches = re.findall(pattern, question_lower)
            if matches:
                easy_score += len(matches) * weight
        
        # 计算困难难度分数
        for pattern, weight in self.difficulty_rules[DifficultyLevel.HARD]:
            matches = re.findall(pattern, question_lower)
            if matches:
                hard_score += len(matches) * weight
        
        # 基于答案长度和复杂度的额外判断
        if answer:
            answer_len = len(answer)
            # 答案越长，可能越难
            if answer_len > 500:
                hard_score += 0.5
            elif answer_len > 200:
                hard_score += 0.3
            elif answer_len < 50:
                easy_score += 0.3
            
            # 答案中包含复杂结构
            if "首先" in answer and "其次" in answer:
                hard_score += 0.2
            if "一方面" in answer and "另一方面" in answer:
                hard_score += 0.3
        
        # 决策逻辑
        if hard_score > easy_score and hard_score > 0.5:
            difficulty = DifficultyLevel.HARD
        elif easy_score > hard_score and easy_score > 0.5:
            difficulty = DifficultyLevel.EASY
        else:
            difficulty = DifficultyLevel.MEDIUM
        
        logger.debug(f"难度标注: '{question}' -> {difficulty.value} (简单分: {easy_score:.2f}, 困难分: {hard_score:.2f})")
        return difficulty.value
    
    def label_expected_strategy(self, question: str) -> str:
        """
        标注预期检索策略
        
        Args:
            question: 问题文本
            
        Returns:
            预期策略字符串
        """
        question_lower = question.lower()
        
        # 计算策略分数
        strategy_scores = {}
        
        for strategy, rules in self.strategy_rules.items():
            score = 0
            for pattern, weight in rules:
                matches = re.findall(pattern, question_lower)
                if matches:
                    score += len(matches) * weight
            
            strategy_scores[strategy] = score
        
        # 找出最高分的策略
        if not strategy_scores:
            return ExpectedStrategy.TRADITIONAL.value
        
        max_strategy = max(strategy_scores.items(), key=lambda x: x[1])
        
        # 决策逻辑
        if max_strategy[1] < 0.5:
            # 分数太低，根据问题长度和复杂度判断
            if len(question) > 30 or "并且" in question or "而且" in question:
                return ExpectedStrategy.COMBINED.value
            else:
                return ExpectedStrategy.TRADITIONAL.value
        
        logger.debug(f"策略标注: '{question}' -> {max_strategy[0].value} (分数: {max_strategy[1]:.2f})")
        return max_strategy[0].value
    
    def label_all(self, question: str, answer: Optional[str] = None) -> Dict[str, str]:
        """
        一次性标注所有属性
        
        Args:
            question: 问题文本
            answer: 答案文本（可选）
            
        Returns:
            包含所有标注的字典
        """
        return {
            "question_type": self.label_question_type(question, answer),
            "difficulty": self.label_difficulty(question, answer),
            "expected_strategy": self.label_expected_strategy(question)
        }

def test_labeler():
    """测试标注器"""
    labeler = RuleBasedLabeler()
    
    test_cases = [
        ("红烧肉属于什么菜系？", "红烧肉属于川菜。"),
        ("宫保鸡丁需要哪些主要食材？", "需要鸡肉、花生、干辣椒等。"),
        ("如何制作麻婆豆腐？请分步骤说明。", "首先准备豆腐，然后..."),
        ("鸡肉通常和哪些蔬菜搭配？", "鸡肉可以和胡萝卜、土豆、青椒等搭配。"),
        ("川菜中哪些菜品适合糖尿病患者，且制作时间不超过30分钟？", "凉拌黄瓜、清炒时蔬等。"),
        ("红烧肉和糖醋排骨在制作方法上有何不同？", "红烧肉需要炖煮，糖醋排骨需要油炸。"),
        ("为什么川菜常用花椒？", "花椒能带来麻辣口感，促进食欲。"),
    ]
    
    print("测试基于规则的标注器：")
    print("=" * 80)
    
    for i, (question, answer) in enumerate(test_cases, 1):
        labels = labeler.label_all(question, answer)
        print(f"测试案例 {i}:")
        print(f"  问题: {question}")
        print(f"  答案: {answer}")
        print(f"  标注: {labels}")
        print()

if __name__ == "__main__":
    test_labeler()