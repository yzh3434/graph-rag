"""
LLM Function Calling 路由器（小升级骨架）

目标：把现有"规则短路 + LLM 兜底(JSON 输出)"升级为"规则短路 + LLM 自主 tool calling"。

设计要点：
1. **保留规则短路（Layer 1）**——fast path 是免费的延迟收益，规则命中就直接走
2. **规则未命中**才进 LLM function calling（Layer 2 升级版）
3. LLM 看到三个 tool（traditional / graph_rag / combined）后自主决定调哪个
4. **兼容现有接口**：暴露 `route_query(query, top_k) -> (docs, QueryAnalysis)`，
   可以直接在 main.py 里替换原 IntelligentQueryRouter
5. **可解释**：LLM 调 tool 时的 reasoning 文本进入 analysis.reasoning，比纯路由器
   confidence 数字更可读

后续接手 TODO：
- [x] 在 main.py 把 IntelligentQueryRouter 切换为 LLMRouterWithToolCalling（或加 flag）
- [x] 在 config.py 加 `enable_tool_calling_router: bool = False` 开关
- [ ] 跑一次 eval：`python -m eval.eval_runner --run_id route_v3`
- [ ] 对比 retrieval_v2 vs route_v3：路由准确率、延迟、Faithfulness
- [ ] 已知风险点：DeepSeek tool_calling 输出格式偶尔不稳定，需要 fallback
"""

import json
import logging
from typing import List, Dict, Tuple, Any, Optional

from langchain_core.documents import Document

from .intelligent_query_router import (
    IntelligentQueryRouter,
    QueryAnalysis,
    SearchStrategy,
)

logger = logging.getLogger(__name__)


# ============================================================
# Tool Schemas — OpenAI / DeepSeek 兼容的 function calling 格式
# ============================================================
#
# 三个 tool 对应原路由器的三个策略；参数尽量克制（只暴露 LLM 真正需要决定的字段），
# 避免 LLM 自己幻觉出无效参数。
#
# ingredients / recipes 字段对应规则短路里 extracted_ingredients / extracted_recipes
# 的角色——LLM 可以主动把识别出的实体填进去，等价于"自己做 longest-match"。

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_hybrid_traditional",
            "description": (
                "适用于单道菜的事实/属性/步骤/原因查询。底层是 BM25(jieba) + 向量 + "
                "图键值索引三路召回 + RRF 融合。"
                "示例：'宫保鸡丁怎么做？''红烧肉的食材有哪些？''为什么要先焯水？'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户原始查询，原样传入"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_graph_rag",
            "description": (
                "适用于多个食材的共现/搭配/关系链查询。底层是 Neo4j 图遍历，会查找"
                "同时包含给定食材的菜品。"
                "示例：'鸡蛋和番茄一起出现在哪些菜里？''姜和香菜常搭配什么？'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户原始查询"
                    },
                    "ingredients": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "从查询中识别出的食材列表（≥2 个时触发 fast path 直接走"
                            "精准 2-hop Cypher，跳过图侧 LLM 意图分析）。"
                            "示例：['鸡蛋', '番茄']"
                        )
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_combined",
            "description": (
                "适用于两道菜或两类菜的对比/差异查询。底层会同时调传统混合检索和图检索，"
                "由生成层做对比。"
                "示例：'宫保鸡丁和麻婆豆腐有什么区别？''川菜和粤菜的差异？'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户原始查询"
                    },
                    "recipes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "从查询中识别出的菜名列表（≥2 个时触发 fast path 直接按名"
                            "查回 Recipe 节点，跳过图侧 LLM 意图分析）。"
                            "示例：['宫保鸡丁', '麻婆豆腐']"
                        )
                    }
                },
                "required": ["query"]
            }
        }
    },
]


# ============================================================
# System Prompt — 给 LLM 的"路由判断指南"
# ============================================================
#
# 这个 prompt 把原 _llm_route 的 few-shot 转写成 tool calling 风格。
# 重点：让 LLM **必须**调一个 tool，不要直接输出文本答案。

_SYSTEM_PROMPT = """你是一个 RAG 系统的查询路由助手。你的唯一任务是根据用户查询，从三个检索工具中选择一个最适合的并调用。

工具选择原则：
1. 单道菜的事实/属性/步骤/原因（无论用没用"为什么/如何"等词）→ search_hybrid_traditional
2. 多个食材的共现/搭配/关系链（"哪些菜""一起出现""同时含"等）→ search_graph_rag
3. 两道菜或两类菜的对比/差异（"区别""对比""哪个更"等）→ search_combined

调用 search_graph_rag 时，如果你能从查询里识别出 ≥2 个食材名，请把它们填入 ingredients 参数。
调用 search_combined 时，如果你能从查询里识别出 ≥2 个菜名，请把它们填入 recipes 参数。

注意：你必须调用一个工具，不要直接回答用户。如果不确定走哪个，默认调 search_hybrid_traditional。"""


# ============================================================
# LLMRouterWithToolCalling — 升级版路由器
# ============================================================

class LLMRouterWithToolCalling:
    """
    Function Calling 版路由器。

    分层策略：
      Layer 1（保留）：规则短路 _pattern_based_route——免费 fast path，命中直接走
      Layer 2（升级）：LLM function calling——LLM 自主选 tool 并填参数
      Layer 3（兜底）：LLM 调用失败 / tool name 异常 → 默认走 hybrid_traditional

    与原 IntelligentQueryRouter 的关系：
      - 复用其 _pattern_based_route（规则短路）
      - 复用其 traditional_retrieval / graph_rag_retrieval / _combined_search
      - 仅替换 LLM 意图分析层（原 _llm_route → tool calling）

    使用方式：和 IntelligentQueryRouter 接口完全兼容
      router = LLMRouterWithToolCalling(traditional, graph_rag, llm_client, config)
      docs, analysis = router.route_query(query, top_k=5)
    """

    def __init__(self,
                 traditional_retrieval,
                 graph_rag_retrieval,
                 llm_client,
                 config):
        self.traditional_retrieval = traditional_retrieval
        self.graph_rag_retrieval = graph_rag_retrieval
        self.llm_client = llm_client
        self.config = config

        # 复用现有路由器的规则短路逻辑（不重复实现）
        self._inner_router = IntelligentQueryRouter(
            traditional_retrieval=traditional_retrieval,
            graph_rag_retrieval=graph_rag_retrieval,
            llm_client=llm_client,
            config=config,
        )

        # 路由统计（包括 tool calling 命中各 tool 的次数）
        self.route_stats: Dict[str, int] = {
            "pattern_route_count": 0,
            "tool_calling_count": 0,
            "tool_calling_failed_count": 0,
            "traditional_count": 0,
            "graph_rag_count": 0,
            "combined_count": 0,
            "total_queries": 0,
        }

    # ==================== 主入口 ====================

    def route_query(self, query: str, top_k: int = 5) -> Tuple[List[Document], QueryAnalysis]:
        """主入口：兼容 IntelligentQueryRouter.route_query 签名"""
        self.route_stats["total_queries"] += 1
        logger.info(f"[ToolCalling Router] 开始路由: {query}")

        # —— Layer 1: 规则短路（复用 inner router）——
        pattern_result = self._inner_router._pattern_based_route(query)
        if pattern_result is not None:
            self.route_stats["pattern_route_count"] += 1
            logger.info(
                f"规则短路命中: {pattern_result.recommended_strategy.value} "
                f"({pattern_result.reasoning})"
            )
            documents = self._execute_strategy(query, top_k, pattern_result)
            self._update_strategy_stats(pattern_result.recommended_strategy)
            return documents, pattern_result

        # —— Layer 2: LLM function calling ——
        try:
            analysis = self._llm_tool_calling_route(query)
            self.route_stats["tool_calling_count"] += 1
        except Exception as e:
            logger.warning(f"Tool calling 失败，降级到 hybrid_traditional: {e}")
            self.route_stats["tool_calling_failed_count"] += 1
            # —— Layer 3: 兜底 ——
            analysis = QueryAnalysis(
                query_complexity=0.3,
                relationship_intensity=0.2,
                reasoning_required=False,
                entity_count=1,
                recommended_strategy=SearchStrategy.HYBRID_TRADITIONAL,
                confidence=0.5,
                reasoning=f"Tool calling 失败，降级到默认策略: {e}",
            )

        documents = self._execute_strategy(query, top_k, analysis)
        self._update_strategy_stats(analysis.recommended_strategy)
        return documents, analysis

    # ==================== LLM Tool Calling ====================

    def _llm_tool_calling_route(self, query: str) -> QueryAnalysis:
        """调 LLM with tools，解析返回的 tool_call，转成 QueryAnalysis"""
        response = self.llm_client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            tools=TOOL_SCHEMAS,
            tool_choice="required",  # 强制调一个工具，不允许直接回答
            temperature=0.1,
            max_tokens=300,
        )

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            raise ValueError("LLM 未返回 tool_calls（违反 tool_choice=required）")

        # 取第一个 tool_call（后续可扩展为多 tool 串行）
        tc = tool_calls[0]
        tool_name = tc.function.name
        tool_args = json.loads(tc.function.arguments or "{}")
        logger.info(f"LLM 调用 tool: {tool_name}, args: {tool_args}")

        # tool_name → SearchStrategy 映射 + 提取 hint
        if tool_name == "search_hybrid_traditional":
            return QueryAnalysis(
                query_complexity=0.4,
                relationship_intensity=0.2,
                reasoning_required=False,
                entity_count=1,
                recommended_strategy=SearchStrategy.HYBRID_TRADITIONAL,
                confidence=0.85,
                reasoning="LLM tool calling → hybrid_traditional",
            )
        elif tool_name == "search_graph_rag":
            ingredients = tool_args.get("ingredients") or None
            return QueryAnalysis(
                query_complexity=0.7,
                relationship_intensity=0.9,
                reasoning_required=True,
                entity_count=len(ingredients) if ingredients else 2,
                recommended_strategy=SearchStrategy.GRAPH_RAG,
                confidence=0.85,
                reasoning=f"LLM tool calling → graph_rag (ingredients={ingredients})",
                extracted_ingredients=ingredients[:2] if ingredients and len(ingredients) >= 2 else None,
            )
        elif tool_name == "search_combined":
            recipes = tool_args.get("recipes") or None
            return QueryAnalysis(
                query_complexity=0.7,
                relationship_intensity=0.5,
                reasoning_required=True,
                entity_count=len(recipes) if recipes else 2,
                recommended_strategy=SearchStrategy.COMBINED,
                confidence=0.85,
                reasoning=f"LLM tool calling → combined (recipes={recipes})",
                extracted_recipes=recipes[:2] if recipes and len(recipes) >= 2 else None,
            )
        else:
            raise ValueError(f"未知 tool name: {tool_name}")

    # ==================== 策略执行（复用 inner router 的能力）====================

    def _execute_strategy(self, query: str, top_k: int,
                          analysis: QueryAnalysis) -> List[Document]:
        """根据 analysis.recommended_strategy 执行检索（复用 inner router 的方法）"""
        try:
            if analysis.recommended_strategy == SearchStrategy.HYBRID_TRADITIONAL:
                docs = self.traditional_retrieval.hybrid_search(query, top_k)
            elif analysis.recommended_strategy == SearchStrategy.GRAPH_RAG:
                docs = self.graph_rag_retrieval.graph_rag_search(
                    query, top_k, ingredients_hint=analysis.extracted_ingredients
                )
            elif analysis.recommended_strategy == SearchStrategy.COMBINED:
                docs = self._inner_router._combined_search(
                    query, top_k, recipes_hint=analysis.extracted_recipes
                )
            else:
                docs = self.traditional_retrieval.hybrid_search(query, top_k)

            return self._inner_router._post_process_results(docs, analysis)

        except Exception as e:
            logger.error(f"策略执行失败，降级到 hybrid_traditional: {e}")
            return self.traditional_retrieval.hybrid_search(query, top_k)

    # ==================== 统计 ====================

    def _update_strategy_stats(self, strategy: SearchStrategy):
        if strategy == SearchStrategy.HYBRID_TRADITIONAL:
            self.route_stats["traditional_count"] += 1
        elif strategy == SearchStrategy.GRAPH_RAG:
            self.route_stats["graph_rag_count"] += 1
        elif strategy == SearchStrategy.COMBINED:
            self.route_stats["combined_count"] += 1

    def get_route_statistics(self) -> Dict[str, Any]:
        """获取路由统计（兼容 IntelligentQueryRouter 接口）"""
        total = self.route_stats["total_queries"]
        if total == 0:
            return self.route_stats
        return {
            **self.route_stats,
            "traditional_ratio": self.route_stats["traditional_count"] / total,
            "graph_rag_ratio": self.route_stats["graph_rag_count"] / total,
            "combined_ratio": self.route_stats["combined_count"] / total,
            "pattern_route_ratio": self.route_stats["pattern_route_count"] / total,
            "tool_calling_ratio": self.route_stats["tool_calling_count"] / total,
        }

    def explain_routing_decision(self, query: str) -> str:
        """解释路由决策（兼容接口）"""
        # 不真的执行检索，只跑路由判断
        pattern_result = self._inner_router._pattern_based_route(query)
        if pattern_result is not None:
            return (
                f"查询: {query}\n"
                f"路由层: Layer 1 规则短路\n"
                f"策略: {pattern_result.recommended_strategy.value}\n"
                f"理由: {pattern_result.reasoning}"
            )
        try:
            analysis = self._llm_tool_calling_route(query)
            return (
                f"查询: {query}\n"
                f"路由层: Layer 2 LLM function calling\n"
                f"策略: {analysis.recommended_strategy.value}\n"
                f"理由: {analysis.reasoning}"
            )
        except Exception as e:
            return f"查询: {query}\n路由失败: {e}"
