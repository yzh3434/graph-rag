"""
测试集生成模块
基于 Neo4j 图结构生成菜谱问答测试集，用于评估 RAG 系统的检索、路由、生成质量。

核心思路：每个问题类型对应一个专门的 Cypher 查询，从图里捞真实存在的关系作为
种子。简单类型（事实/属性/步骤/食材）直接模板拼接；复杂类型（对比/因果）由 LLM
基于结构化种子生成自然语言问答。
"""

import os
import re
import sys
import json
import random
import logging
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from tqdm import tqdm

_C9_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _C9_DIR not in sys.path:
    sys.path.insert(0, _C9_DIR)

from config import GraphRAGConfig, DEFAULT_CONFIG

load_dotenv()
logger = logging.getLogger(__name__)


class QuestionType(Enum):
    SIMPLE_FACT = "simple_fact"
    ATTRIBUTE_QUERY = "attribute_query"
    STEP_BY_STEP = "step_by_step"
    ENTITY_RELATION = "entity_relation"
    MULTI_HOP = "multi_hop"
    COMPARISON = "comparison"
    CAUSAL = "causal"


class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ExpectedStrategy(Enum):
    TRADITIONAL = "hybrid_traditional"
    GRAPH_RAG = "graph_rag"
    COMBINED = "combined"


TYPE_TO_STRATEGY = {
    QuestionType.SIMPLE_FACT: ExpectedStrategy.TRADITIONAL,
    QuestionType.ATTRIBUTE_QUERY: ExpectedStrategy.TRADITIONAL,
    QuestionType.STEP_BY_STEP: ExpectedStrategy.TRADITIONAL,
    QuestionType.ENTITY_RELATION: ExpectedStrategy.GRAPH_RAG,
    QuestionType.MULTI_HOP: ExpectedStrategy.GRAPH_RAG,
    QuestionType.COMPARISON: ExpectedStrategy.COMBINED,
    QuestionType.CAUSAL: ExpectedStrategy.GRAPH_RAG,
}

TYPE_TO_DIFFICULTY = {
    QuestionType.SIMPLE_FACT: DifficultyLevel.EASY,
    QuestionType.ATTRIBUTE_QUERY: DifficultyLevel.EASY,
    QuestionType.STEP_BY_STEP: DifficultyLevel.MEDIUM,
    QuestionType.ENTITY_RELATION: DifficultyLevel.MEDIUM,
    QuestionType.MULTI_HOP: DifficultyLevel.HARD,
    QuestionType.COMPARISON: DifficultyLevel.HARD,
    QuestionType.CAUSAL: DifficultyLevel.HARD,
}


@dataclass
class TestSample:
    question: str
    ground_truth: str
    question_type: str
    difficulty: str
    expected_strategy: str
    source_node_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestSample":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in allowed})


class TestsetGenerator:
    """基于 Neo4j 图结构生成菜谱问答测试集"""

    def __init__(self, config: Optional[GraphRAGConfig] = None):
        self.config = config or DEFAULT_CONFIG

        self.driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_password),
        )
        with self.driver.session() as s:
            s.run("RETURN 1").single()
        logger.info(f"已连接 Neo4j: {self.config.neo4j_uri}")

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")
        self.llm = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/")
        self.model_name = self.config.llm_model

        self.type_distribution = {
            QuestionType.SIMPLE_FACT: 0.15,
            QuestionType.ATTRIBUTE_QUERY: 0.15,
            QuestionType.STEP_BY_STEP: 0.15,
            QuestionType.ENTITY_RELATION: 0.15,
            QuestionType.MULTI_HOP: 0.15,
            QuestionType.COMPARISON: 0.10,
            QuestionType.CAUSAL: 0.15,
        }

    def close(self):
        if self.driver:
            self.driver.close()

    def _build_sample(
        self,
        q_type: QuestionType,
        question: str,
        ground_truth: str,
        source_node_ids: List[str],
        metadata: Optional[Dict] = None,
    ) -> TestSample:
        return TestSample(
            question=question,
            ground_truth=ground_truth,
            question_type=q_type.value,
            difficulty=TYPE_TO_DIFFICULTY[q_type].value,
            expected_strategy=TYPE_TO_STRATEGY[q_type].value,
            source_node_ids=source_node_ids,
            metadata={"generated_by": "graph_based", **(metadata or {})},
        )

    # ========== 1. SIMPLE_FACT：菜 -> 分类 ==========
    def _gen_simple_fact(self, n: int) -> List[TestSample]:
        cypher = """
        MATCH (r:Recipe)
        WHERE r.nodeId >= '200000000'
          AND r.name IS NOT NULL
          AND r.category IS NOT NULL AND r.category <> '未知'
        RETURN r.nodeId AS node_id, r.name AS recipe_name, r.category AS category
        ORDER BY rand()
        LIMIT $n
        """
        samples = []
        with self.driver.session() as session:
            for record in session.run(cypher, n=n):
                name = record["recipe_name"]
                category = record["category"]
                node_id = record["node_id"]

                templates = [
                    (f"{name}属于什么分类？", f"{name}属于{category}。"),
                    (f"请问{name}是哪一类菜？", f"{name}是{category}。"),
                    (f"{name}这道菜的分类是什么？", f"{name}的分类是{category}。"),
                ]
                q, a = random.choice(templates)
                samples.append(
                    self._build_sample(
                        QuestionType.SIMPLE_FACT, q, a, [node_id],
                        metadata={"category": category},
                    )
                )
        return samples

    # ========== 2. ATTRIBUTE_QUERY：菜的某个属性 ==========
    def _gen_attribute_query(self, n: int) -> List[TestSample]:
        cypher = """
        MATCH (r:Recipe)
        WHERE r.nodeId >= '200000000' AND r.name IS NOT NULL
          AND (
            (r.prepTime IS NOT NULL AND r.prepTime <> '')
            OR (r.cookTime IS NOT NULL AND r.cookTime <> '')
            OR (r.servings IS NOT NULL AND r.servings <> '')
            OR r.difficulty IS NOT NULL
          )
        RETURN r.nodeId AS node_id, r.name AS recipe_name,
               r.prepTime AS prep_time, r.cookTime AS cook_time,
               r.servings AS servings, r.difficulty AS difficulty
        ORDER BY rand()
        LIMIT $n
        """
        samples = []
        with self.driver.session() as session:
            for record in session.run(cypher, n=n):
                name = record["recipe_name"]
                node_id = record["node_id"]

                choices = []
                if record["prep_time"]:
                    choices.append((
                        "prep_time",
                        f"{name}的准备时间是多久？",
                        f"{name}的准备时间是{record['prep_time']}。",
                    ))
                if record["cook_time"]:
                    choices.append((
                        "cook_time",
                        f"{name}的烹饪时间是多久？",
                        f"{name}的烹饪时间是{record['cook_time']}。",
                    ))
                if record["servings"]:
                    choices.append((
                        "servings",
                        f"{name}的份量是多少？",
                        f"{name}的份量是{record['servings']}。",
                    ))
                if record["difficulty"] is not None:
                    difficulty = int(record["difficulty"])
                    choices.append((
                        "difficulty",
                        f"{name}的难度是几星？",
                        f"{name}的难度是{difficulty}星。",
                    ))

                if not choices:
                    continue
                attr, q, a = random.choice(choices)
                samples.append(
                    self._build_sample(
                        QuestionType.ATTRIBUTE_QUERY, q, a, [node_id],
                        metadata={"attribute": attr},
                    )
                )
        return samples

    # ========== 3. STEP_BY_STEP：菜 -> 步骤列表 ==========
    def _gen_step_by_step(self, n: int) -> List[TestSample]:
        cypher = """
        MATCH (r:Recipe)-[c:CONTAINS_STEP]->(s:CookingStep)
        WHERE r.nodeId >= '200000000' AND r.name IS NOT NULL
        WITH r, collect({
            order: coalesce(c.stepOrder, s.stepNumber, 999.0),
            name: s.name,
            desc: s.description
        }) AS steps
        WHERE size(steps) >= 3
        RETURN r.nodeId AS node_id, r.name AS recipe_name, steps
        ORDER BY rand()
        LIMIT $n
        """
        samples = []
        with self.driver.session() as session:
            for record in session.run(cypher, n=n):
                name = record["recipe_name"]
                node_id = record["node_id"]
                steps_raw = record["steps"]
                steps_sorted = sorted(steps_raw, key=lambda x: x.get("order") or 999)

                lines = []
                for i, step in enumerate(steps_sorted, 1):
                    step_name = step.get("name") or f"步骤{i}"
                    step_desc = step.get("desc") or ""
                    if step_desc and step_desc != step_name:
                        lines.append(f"第{i}步（{step_name}）：{step_desc}")
                    else:
                        lines.append(f"第{i}步：{step_name}")
                answer = f"{name}的制作步骤如下：\n" + "\n".join(lines)

                templates = [
                    f"如何制作{name}？请分步骤说明。",
                    f"{name}的制作步骤是什么？",
                    f"请详细介绍{name}的做法。",
                ]
                q = random.choice(templates)
                samples.append(
                    self._build_sample(
                        QuestionType.STEP_BY_STEP, q, answer, [node_id],
                        metadata={"step_count": len(steps_sorted)},
                    )
                )
        return samples

    # ========== 4. ENTITY_RELATION：菜 -> 食材列表 ==========
    def _gen_entity_relation(self, n: int) -> List[TestSample]:
        cypher = """
        MATCH (r:Recipe)-[:REQUIRES]->(i:Ingredient)
        WHERE r.nodeId >= '200000000' AND r.name IS NOT NULL AND i.name IS NOT NULL
        WITH r, collect(DISTINCT i.name) AS ingredients
        WHERE size(ingredients) >= 2 AND size(ingredients) <= 15
        RETURN r.nodeId AS node_id, r.name AS recipe_name, ingredients
        ORDER BY rand()
        LIMIT $n
        """
        samples = []
        with self.driver.session() as session:
            for record in session.run(cypher, n=n):
                name = record["recipe_name"]
                node_id = record["node_id"]
                ingredients = record["ingredients"]

                ingredient_str = "、".join(ingredients)
                answer = f"制作{name}需要以下主要食材：{ingredient_str}。"
                templates = [
                    f"{name}需要哪些主要食材？",
                    f"制作{name}需要准备什么食材？",
                    f"请问{name}的原料有哪些？",
                ]
                q = random.choice(templates)
                samples.append(
                    self._build_sample(
                        QuestionType.ENTITY_RELATION, q, answer, [node_id],
                        metadata={"ingredient_count": len(ingredients)},
                    )
                )
        return samples

    # ========== 5. MULTI_HOP：两种食材 -> 共现的菜 ==========
    def _gen_multi_hop(self, n: int) -> List[TestSample]:
        cypher = """
        MATCH (r:Recipe)-[:REQUIRES]->(i1:Ingredient)
        MATCH (r)-[:REQUIRES]->(i2:Ingredient)
        WHERE r.nodeId >= '200000000' AND r.name IS NOT NULL
          AND i1.name IS NOT NULL AND i2.name IS NOT NULL
          AND i1.name < i2.name
        WITH i1.name AS ing1, i2.name AS ing2,
             collect(DISTINCT {id: r.nodeId, name: r.name}) AS recipes
        WHERE size(recipes) >= 2 AND size(recipes) <= 8
        RETURN ing1, ing2, recipes
        ORDER BY size(recipes) DESC, rand()
        LIMIT $n
        """
        samples = []
        with self.driver.session() as session:
            for record in session.run(cypher, n=n):
                ing1 = record["ing1"]
                ing2 = record["ing2"]
                recipes = record["recipes"]

                recipe_names = [r["name"] for r in recipes]
                node_ids = [r["id"] for r in recipes]

                names_str = "、".join(recipe_names)
                answer = f"{ing1}和{ing2}常常一起出现在这些菜中：{names_str}。"
                templates = [
                    f"{ing1}和{ing2}常常一起出现在哪些菜里？",
                    f"有哪些菜同时用到{ing1}和{ing2}？",
                    f"能同时使用{ing1}和{ing2}的菜有哪些？",
                ]
                q = random.choice(templates)
                samples.append(
                    self._build_sample(
                        QuestionType.MULTI_HOP, q, answer, node_ids,
                        metadata={
                            "ingredient_pair": [ing1, ing2],
                            "recipe_count": len(recipes),
                        },
                    )
                )
        return samples

    # ========== 6. COMPARISON：同分类两道菜对比（LLM） ==========
    def _gen_comparison(self, n: int) -> List[TestSample]:
        cypher = """
        MATCH (r:Recipe)
        WHERE r.nodeId >= '200000000' AND r.name IS NOT NULL
          AND r.category IS NOT NULL AND r.category <> '未知'
        OPTIONAL MATCH (r)-[:REQUIRES]->(i:Ingredient)
        WITH r, collect(DISTINCT i.name)[0..6] AS ingredients
        WHERE size(ingredients) >= 2
        RETURN r.nodeId AS node_id, r.name AS name, r.category AS category,
               r.difficulty AS difficulty, r.cookTime AS cook_time, ingredients
        """
        with self.driver.session() as session:
            all_recipes = [dict(record) for record in session.run(cypher)]

        from collections import defaultdict
        by_category = defaultdict(list)
        for r in all_recipes:
            by_category[r["category"]].append(r)
        pairable = [cat for cat, items in by_category.items() if len(items) >= 2]
        if not pairable:
            return []

        samples = []
        attempts = 0
        max_attempts = n * 3
        while len(samples) < n and attempts < max_attempts:
            attempts += 1
            cat = random.choice(pairable)
            r1, r2 = random.sample(by_category[cat], 2)

            context = {
                "dish1": {
                    "name": r1["name"],
                    "ingredients": r1["ingredients"],
                    "difficulty": r1.get("difficulty"),
                    "cook_time": r1.get("cook_time"),
                },
                "dish2": {
                    "name": r2["name"],
                    "ingredients": r2["ingredients"],
                    "difficulty": r2.get("difficulty"),
                    "cook_time": r2.get("cook_time"),
                },
                "category": cat,
            }
            result = self._llm_generate_comparison(context)
            if result is None:
                continue

            samples.append(
                self._build_sample(
                    QuestionType.COMPARISON,
                    result["question"],
                    result["answer"],
                    [r1["node_id"], r2["node_id"]],
                    metadata={"category": cat},
                )
            )
        return samples

    def _llm_generate_comparison(self, context: Dict) -> Optional[Dict]:
        d1 = context["dish1"]
        d2 = context["dish2"]
        d1_ing = "、".join(d1["ingredients"])
        d2_ing = "、".join(d2["ingredients"])

        prompt = f"""你是烹饪问答测试集生成助手。下面是同一分类下两道菜的信息，请基于信息生成一对对比型问答。

菜品一：{d1["name"]}
  主要食材：{d1_ing}
  难度：{d1["difficulty"]}
  烹饪时间：{d1["cook_time"]}

菜品二：{d2["name"]}
  主要食材：{d2_ing}
  难度：{d2["difficulty"]}
  烹饪时间：{d2["cook_time"]}

共同分类：{context["category"]}

要求：
1. 问题必须是对比型，询问两道菜在食材、难度、烹饪时间等某一方面的差异。
2. 答案必须严格基于上方提供的信息，不要编造不存在的内容。
3. 答案长度 50-200 字。
4. 严格返回 JSON 格式：{{"question": "...", "answer": "..."}}，不要输出其他内容。
"""
        try:
            response = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            if result.get("question") and result.get("answer"):
                return result
        except Exception as e:
            logger.warning(f"LLM 对比生成失败: {e}")
        return None

    # ========== 7. CAUSAL：为什么类问题（LLM） ==========
    def _gen_causal(self, n: int) -> List[TestSample]:
        cypher = """
        MATCH (r:Recipe)-[:CONTAINS_STEP]->(s:CookingStep)
        WHERE r.nodeId >= '200000000' AND r.name IS NOT NULL
        WITH r, count(s) AS step_count,
             collect({name: s.name, desc: s.description, methods: s.methods}) AS steps
        WHERE step_count >= 3
        OPTIONAL MATCH (r)-[:REQUIRES]->(i:Ingredient)
        WITH r, step_count, steps, collect(DISTINCT i.name)[0..8] AS ingredients
        WHERE size(ingredients) >= 2
        RETURN r.nodeId AS node_id, r.name AS recipe_name, r.description AS description,
               step_count, steps, ingredients
        ORDER BY step_count DESC, rand()
        LIMIT $n
        """
        samples = []
        attempts = 0
        with self.driver.session() as session:
            records = list(session.run(cypher, n=n * 3))

        for record in records:
            if len(samples) >= n:
                break
            attempts += 1

            steps_preview = (record["steps"] or [])[:10]
            context = {
                "recipe_name": record["recipe_name"],
                "description": record["description"] or "",
                "step_count": record["step_count"],
                "steps_preview": steps_preview,
                "ingredients": record["ingredients"],
            }
            result = self._llm_generate_causal(context)
            if result is None:
                continue
            samples.append(
                self._build_sample(
                    QuestionType.CAUSAL,
                    result["question"],
                    result["answer"],
                    [record["node_id"]],
                    metadata={"step_count": record["step_count"]},
                )
            )
        return samples

    def _llm_generate_causal(self, context: Dict) -> Optional[Dict]:
        steps_str = "\n".join(
            [
                f"  - {s.get('name', '')}: {(s.get('desc') or '')[:80]}"
                for s in context["steps_preview"]
            ]
        )
        ingredient_str = "、".join(context["ingredients"])
        recipe_name = context["recipe_name"]

        prompt = f"""你是烹饪问答测试集生成助手。请站在真实做菜用户的角度，生成一对"为什么"型问答。

菜名：{recipe_name}
描述：{context["description"]}
主要食材：{ingredient_str}
步骤总数：{context["step_count"]}
步骤详情：
{steps_str}

要求：
1. 问题必须以"为什么"开头，并且必须显式包含菜名"{recipe_name}"，让人脱离上下文也能看懂这是在问哪道菜。
2. 问题应该是普通用户做菜时真正会关心的问题，围绕食材选择、烹饪方法、火候、时间、调味、口感原理等展开。例如：
   - 为什么{recipe_name}要先腌制再下锅？
   - 为什么{recipe_name}里要加某种食材？
   - 为什么{recipe_name}适合用某种烹饪方式？
   - 为什么{recipe_name}的烹饪时间这么长／这么短？
3. **严禁**问题中出现"步骤N"、"第N步"、"步骤几"等引用具体步骤编号的表达——真实用户不会拿着步骤序号来提问。
4. 答案必须严格基于上方提供的菜谱信息推理，不要编造图中不存在的科学原理或细节数字。
5. 答案长度 50-200 字。
6. 严格返回 JSON 格式：{{"question": "...", "answer": "..."}}，不要输出其他内容。
"""
        try:
            response = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            q = result.get("question", "").strip()
            a = result.get("answer", "").strip()
            if not (q and a and q.startswith("为什么")):
                return None
            if recipe_name not in q:
                logger.debug(f"causal 样本被丢弃（问题中缺少菜名 {recipe_name}）：{q}")
                return None
            if re.search(r"步骤\s*\d+|第\s*\d+\s*步|步骤几", q):
                logger.debug(f"causal 样本被丢弃（问题引用了步骤编号）：{q}")
                return None
            return {"question": q, "answer": a}
        except Exception as e:
            logger.warning(f"LLM 因果生成失败: {e}")
        return None

    # ========== 主入口 ==========
    def generate(self, num_samples: int = 100) -> List[TestSample]:
        logger.info(f"开始生成测试集，目标样本数: {num_samples}")

        generators = {
            QuestionType.SIMPLE_FACT: self._gen_simple_fact,
            QuestionType.ATTRIBUTE_QUERY: self._gen_attribute_query,
            QuestionType.STEP_BY_STEP: self._gen_step_by_step,
            QuestionType.ENTITY_RELATION: self._gen_entity_relation,
            QuestionType.MULTI_HOP: self._gen_multi_hop,
            QuestionType.COMPARISON: self._gen_comparison,
            QuestionType.CAUSAL: self._gen_causal,
        }

        all_samples: List[TestSample] = []
        for q_type, gen_fn in generators.items():
            target = max(1, int(num_samples * self.type_distribution[q_type]))
            logger.info(f"生成 {q_type.value}，目标 {target}")
            try:
                samples = gen_fn(target)
                samples = samples[:target]
                all_samples.extend(samples)
                logger.info(f"  -> 实际生成 {len(samples)}")
            except Exception as e:
                logger.error(f"生成 {q_type.value} 失败: {e}", exc_info=True)

        random.shuffle(all_samples)
        logger.info(f"测试集生成完成，共 {len(all_samples)} 个样本")
        return all_samples


def main():
    parser = argparse.ArgumentParser(description="生成 Graph RAG 系统测试集")
    parser.add_argument("--num_samples", type=int, default=100, help="目标样本数量")
    parser.add_argument("--output_dir", type=str, default="./testset_output", help="输出目录")
    parser.add_argument("--log_level", type=str, default="INFO", help="日志级别")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    if args.seed is not None:
        random.seed(args.seed)

    generator = TestsetGenerator()
    try:
        samples = generator.generate(num_samples=args.num_samples)

        from .utils import save_testset, validate_testset
        save_testset(samples, args.output_dir)

        result = validate_testset(samples)
        if result["errors"]:
            print("\n验证发现问题:")
            for err in result["errors"][:5]:
                print(f"  - {err}")

        from .sample_reviewer import SampleReviewer
        reviewer = SampleReviewer(samples)
        reviewer.print_statistics()
    finally:
        generator.close()


if __name__ == "__main__":
    main()
