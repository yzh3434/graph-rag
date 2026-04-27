"""
智能查询路由器
根据查询特点自动选择最适合的检索策略：
- 传统混合检索：适合简单的信息查找
- 图RAG检索：适合复杂的关系推理和知识发现
"""

import json
import logging
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class SearchStrategy(Enum):
    """搜索策略枚举"""
    HYBRID_TRADITIONAL = "hybrid_traditional"  # 传统混合检索
    GRAPH_RAG = "graph_rag"  # 图RAG检索
    COMBINED = "combined"  # 组合策略
    
@dataclass
class QueryAnalysis:
    """查询分析结果"""
    query_complexity: float  # 查询复杂度 (0-1)
    relationship_intensity: float  # 关系密集度 (0-1)
    reasoning_required: bool  # 是否需要推理
    entity_count: int  # 实体数量
    recommended_strategy: SearchStrategy
    confidence: float  # 推荐置信度
    reasoning: str  # 推荐理由
    # 规则路由命中 multi_hop 时填入识别出的食材，graph_rag_search 据此走 fast path
    extracted_ingredients: Optional[List[str]] = None
    # 规则路由命中 comparison 时填入识别出的菜名，combined_search 据此跳过图侧 LLM 意图分析
    extracted_recipes: Optional[List[str]] = None

class IntelligentQueryRouter:
    """
    智能查询路由器
    
    核心能力：
    1. 查询复杂度分析：识别简单查找 vs 复杂推理
    2. 关系密集度评估：判断是否需要图结构优势
    3. 策略自动选择：路由到最适合的检索引擎
    4. 结果质量监控：基于反馈优化路由决策
    """
    
    # 规则路由的强意图关键词
    _MULTI_HOP_KEYWORDS = (
        "哪些菜", "哪几道", "哪几种", "共同", "同时使用", "同时含",
        "一起出现", "一起用", "都用", "搭配", "配什么", "出现在哪",
    )
    _COMPARISON_KEYWORDS = (
        "区别", "不同", "对比", "哪个更", "哪个比较", "比较", "差异", "差别",
    )

    def __init__(self,
                 traditional_retrieval,  # 传统混合检索模块
                 graph_rag_retrieval,    # 图RAG检索模块
                 llm_client,
                 config):
        self.traditional_retrieval = traditional_retrieval
        self.graph_rag_retrieval = graph_rag_retrieval
        self.llm_client = llm_client
        self.config = config

        # 路由统计
        self.route_stats = {
            "traditional_count": 0,
            "graph_rag_count": 0,
            "combined_count": 0,
            "total_queries": 0,
            "pattern_route_count": 0,  # 规则短路命中次数（顺手降 latency 指标）
        }

        # 实体词典（lazy load：首次规则路由时从 Neo4j 拉一次，常驻内存）
        self._dict_loaded = False
        self._ingredient_names: List[str] = []
        self._recipe_names: List[str] = []
        
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        分析查询特征，决定最佳检索策略。

        两层路由：
          1. 规则短路：明确的 multi_hop / comparison 模式直接路由，跳过 LLM
          2. LLM 兜底：规则不命中或信号冲突时，让 LLM 综合判断（few-shot 示例引导）
        """
        logger.info(f"分析查询特征: {query}")

        # —— Layer 1: 规则短路 ——
        pattern_result = self._pattern_based_route(query)
        if pattern_result is not None:
            self.route_stats["pattern_route_count"] += 1
            logger.info(
                f"规则短路命中: {pattern_result.recommended_strategy.value} "
                f"({pattern_result.reasoning})"
            )
            return pattern_result

        # —— Layer 2: 规则未命中 → 默认 hybrid_traditional ——
        # 规则未匹配到 multi_hop / comparison 模式时，绝大多数是单菜事实/属性/步骤/为什么类查询，
        # hybrid_traditional 已能覆盖。除非显式开启 enable_llm_routing_fallback，否则不再调 LLM。
        if not getattr(self.config, "enable_llm_routing_fallback", False):
            logger.info("规则未命中三种模式，默认走 hybrid_traditional（已跳过 LLM）")
            return QueryAnalysis(
                query_complexity=0.3,
                relationship_intensity=0.2,
                reasoning_required=False,
                entity_count=1,
                recommended_strategy=SearchStrategy.HYBRID_TRADITIONAL,
                confidence=0.9,
                reasoning="规则未命中 multi_hop / comparison 模式，默认 hybrid_traditional",
            )

        # —— Layer 3: LLM 兜底（仅在 enable_llm_routing_fallback=True 时启用）——
        analysis_prompt = f"""
作为 RAG 系统的查询分析专家，请分析以下烹饪问答查询，并选择最适合的检索策略。

【可选策略】
- hybrid_traditional: 单道菜的事实/属性/步骤/原因查询（混合检索：BM25 + 向量 + 图键值索引）
- graph_rag:          多个食材/实体的共现、搭配、关系链查询（图遍历）
- combined:           两道菜或两类菜的对比、差异查询（混合检索 + 图检索 融合）

【判断原则】
1. 查询是否涉及"多个食材一起出现/搭配"→ graph_rag
2. 查询是否在"对比两道菜"→ combined
3. 查询是否围绕"单一菜品"展开（无论是事实、属性、步骤还是为什么）→ hybrid_traditional
4. 不要因为查询用了"为什么/如何"等词就路由到 graph_rag——单菜的"为什么"也属于 hybrid_traditional

【参考示例】
查询: "姜和香菜常出现在哪些菜里？"
判断: 两个食材 + 共现查询，需要图遍历
输出: {{"recommended_strategy": "graph_rag", "query_complexity": 0.7, "relationship_intensity": 0.9, "reasoning_required": true, "entity_count": 2, "confidence": 0.9, "reasoning": "多食材共现查询"}}

查询: "宫保鸡丁和麻婆豆腐有什么区别？"
判断: 两道菜对比
输出: {{"recommended_strategy": "combined", "query_complexity": 0.7, "relationship_intensity": 0.5, "reasoning_required": true, "entity_count": 2, "confidence": 0.9, "reasoning": "两菜对比，需融合两路结果"}}

查询: "茭白炒肉的原料有哪些？"
判断: 单道菜的食材属性
输出: {{"recommended_strategy": "hybrid_traditional", "query_complexity": 0.2, "relationship_intensity": 0.2, "reasoning_required": false, "entity_count": 1, "confidence": 0.95, "reasoning": "单菜属性查询，混合检索足够"}}

查询: "为什么红烧肉要先焯水再炖？"
判断: 单道菜的烹饪推理，菜谱内信息可解答
输出: {{"recommended_strategy": "hybrid_traditional", "query_complexity": 0.6, "relationship_intensity": 0.2, "reasoning_required": true, "entity_count": 1, "confidence": 0.85, "reasoning": "单菜内部因果推理"}}

【请分析以下查询】
查询: {query}

请严格返回一个 JSON 对象，字段同上方示例（recommended_strategy / query_complexity / relationship_intensity / reasoning_required / entity_count / confidence / reasoning），不要输出任何其他内容。
"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=800
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            
            analysis = QueryAnalysis(
                query_complexity=result.get("query_complexity", 0.5),
                relationship_intensity=result.get("relationship_intensity", 0.5),
                reasoning_required=result.get("reasoning_required", False),
                entity_count=result.get("entity_count", 1),
                recommended_strategy=SearchStrategy(result.get("recommended_strategy", "hybrid_traditional")),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", "默认分析")
            )
            
            logger.info(f"查询分析完成: {analysis.recommended_strategy.value} (置信度: {analysis.confidence:.2f})")
            return analysis
            
        except Exception as e:
            logger.error(f"查询分析失败: {e}")
            # 降级方案：基于规则的简单分析
            return self._rule_based_analysis(query)
    
    def _rule_based_analysis(self, query: str) -> QueryAnalysis:
        """基于规则的降级分析"""
        # 简单的规则判断
        complexity_keywords = ["为什么", "如何", "关系", "影响", "原因", "比较", "区别"]
        relation_keywords = ["配", "搭配", "组合", "相关", "联系", "连接"]
        
        complexity = sum(1 for kw in complexity_keywords if kw in query) / len(complexity_keywords)
        relation_intensity = sum(1 for kw in relation_keywords if kw in query) / len(relation_keywords)
        
        if complexity > 0.3 or relation_intensity > 0.3:
            strategy = SearchStrategy.GRAPH_RAG
        else:
            strategy = SearchStrategy.HYBRID_TRADITIONAL
            
        return QueryAnalysis(
            query_complexity=complexity,
            relationship_intensity=relation_intensity,
            reasoning_required=complexity > 0.3,
            entity_count=len(query.split()),
            recommended_strategy=strategy,
            confidence=0.6,
            reasoning="基于规则的简单分析"
        )

    # ==================== 规则短路（Layer 1）====================

    def _pattern_based_route(self, query: str) -> Optional[QueryAnalysis]:
        """
        规则路由：明确的 multi_hop / comparison 模式直接路由，跳过 LLM。
        不命中返回 None，由调用方走 LLM 兜底。

        判定规则：
          - query 含 ≥2 个食材名 + multi_hop 关键词 → graph_rag
          - query 含 ≥2 个菜名   + comparison 关键词 → combined
          - 两类信号同时出现（冲突）→ 让 LLM 判断
          - 词典未加载/加载失败    → 让 LLM 判断（fail-open 而非 fail-closed）
        """
        self._load_entity_dictionaries()
        if not self._dict_loaded:
            return None

        has_multi_hop_kw  = any(kw in query for kw in self._MULTI_HOP_KEYWORDS)
        has_comparison_kw = any(kw in query for kw in self._COMPARISON_KEYWORDS)

        if has_multi_hop_kw and has_comparison_kw:
            return None  # 信号冲突，LLM 判

        if has_multi_hop_kw:
            ingredients = self._count_matches(query, self._ingredient_names)
            if len(ingredients) >= 2:
                return QueryAnalysis(
                    query_complexity=0.7,
                    relationship_intensity=0.9,
                    reasoning_required=True,
                    entity_count=len(ingredients),
                    recommended_strategy=SearchStrategy.GRAPH_RAG,
                    confidence=0.95,
                    reasoning=f"规则匹配：多食材共现查询（{ingredients[:3]}）",
                    extracted_ingredients=ingredients[:2],  # 取前两个传给 fast path
                )

        if has_comparison_kw:
            recipes = self._count_matches(query, self._recipe_names)
            if len(recipes) >= 2:
                return QueryAnalysis(
                    query_complexity=0.7,
                    relationship_intensity=0.5,
                    reasoning_required=True,
                    entity_count=len(recipes),
                    recommended_strategy=SearchStrategy.COMBINED,
                    confidence=0.95,
                    reasoning=f"规则匹配：两菜对比查询（{recipes[:3]}）",
                    extracted_recipes=recipes[:2],  # 取前两个传给 combined fast path
                )

        return None

    def _load_entity_dictionaries(self):
        """从 Neo4j lazy load 食材名 + 菜名词典，按长度倒序便于 longest-match"""
        if self._dict_loaded:
            return

        driver = getattr(self.graph_rag_retrieval, "driver", None)
        if driver is None:
            logger.debug("graph_rag_retrieval.driver 未就绪，规则路由暂时跳过")
            return

        try:
            with driver.session() as session:
                ing_result = session.run(
                    "MATCH (i:Ingredient) WHERE i.name IS NOT NULL "
                    "RETURN DISTINCT i.name AS name"
                )
                self._ingredient_names = [r["name"] for r in ing_result if r["name"]]

                rec_result = session.run(
                    "MATCH (r:Recipe) WHERE r.name IS NOT NULL "
                    "RETURN DISTINCT r.name AS name"
                )
                self._recipe_names = [r["name"] for r in rec_result if r["name"]]

            self._ingredient_names.sort(key=len, reverse=True)
            self._recipe_names.sort(key=len, reverse=True)
            self._dict_loaded = True
            logger.info(
                f"路由器实体词典加载完成: "
                f"{len(self._ingredient_names)} 食材 / {len(self._recipe_names)} 菜品"
            )
        except Exception as e:
            logger.warning(f"路由器实体词典加载失败: {e}")

    @staticmethod
    def _count_matches(text: str, dictionary: List[str]) -> List[str]:
        """
        从 text 中找出 dictionary 里出现的词。
        dictionary 必须按长度倒序排好。命中后用占位符消去，避免 "鸡蛋" 命中后 "鸡" 又被算一次。
        """
        found: List[str] = []
        remaining = text
        for term in dictionary:
            if term and term in remaining:
                found.append(term)
                remaining = remaining.replace(term, "□" * len(term))
        return found

    # ==================== 路由执行 ====================

    def route_query(self, query: str, top_k: int = 5) -> Tuple[List[Document], QueryAnalysis]:
        """
        智能路由查询到最适合的检索引擎
        """
        logger.info(f"开始智能路由: {query}")
        
        # 1. 分析查询特征
        analysis = self.analyze_query(query)
        
        # 2. 更新统计
        self._update_route_stats(analysis.recommended_strategy)
        
        # 3. 根据策略执行检索
        documents = []
        
        try:
            if analysis.recommended_strategy == SearchStrategy.HYBRID_TRADITIONAL:
                logger.info("使用传统混合检索")
                documents = self.traditional_retrieval.hybrid_search(query, top_k)
                
            elif analysis.recommended_strategy == SearchStrategy.GRAPH_RAG:
                logger.info("🕸️ 使用图RAG检索")
                documents = self.graph_rag_retrieval.graph_rag_search(
                    query, top_k, ingredients_hint=analysis.extracted_ingredients
                )
                
            elif analysis.recommended_strategy == SearchStrategy.COMBINED:
                logger.info("🔄 使用组合检索策略")
                documents = self._combined_search(
                    query, top_k, recipes_hint=analysis.extracted_recipes
                )
            
            # 4. 结果后处理
            documents = self._post_process_results(documents, analysis)
            
            logger.info(f"路由完成，返回 {len(documents)} 个结果")
            return documents, analysis
            
        except Exception as e:
            logger.error(f"查询路由失败: {e}")
            # 降级到传统检索
            documents = self.traditional_retrieval.hybrid_search(query, top_k)
            return documents, analysis
    
    def _combined_search(self, query: str, top_k: int,
                         recipes_hint: Optional[List[str]] = None) -> List[Document]:
        """
        组合搜索策略：结合传统检索和图RAG的优势

        recipes_hint：上游路由器规则识别出的菜名列表（comparison 模式专用）。
        透传给 graph_rag_search → 触发 fast path 跳过图侧 LLM 意图分析。
        """
        # 分配结果数量
        traditional_k = max(1, top_k // 2)
        graph_k = top_k - traditional_k

        # 执行两种检索
        traditional_docs = self.traditional_retrieval.hybrid_search(query, traditional_k)
        graph_docs = self.graph_rag_retrieval.graph_rag_search(
            query, graph_k, recipes_hint=recipes_hint
        )
        
        # 合并和去重
        combined_docs = []
        seen_contents = set()
        
        # 交替添加结果（Round-robin）
        max_len = max(len(traditional_docs), len(graph_docs))
        for i in range(max_len):
            # 先添加图RAG结果（通常质量更高）
            if i < len(graph_docs):
                doc = graph_docs[i]
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    doc.metadata["search_source"] = "graph_rag"
                    combined_docs.append(doc)
            
            # 再添加传统检索结果
            if i < len(traditional_docs):
                doc = traditional_docs[i]
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    doc.metadata["search_source"] = "traditional"
                    combined_docs.append(doc)
        
        return combined_docs[:top_k]
    
    def _post_process_results(self, documents: List[Document], analysis: QueryAnalysis) -> List[Document]:
        """
        结果后处理：根据查询分析优化结果
        """
        for doc in documents:
            # 添加路由信息到元数据
            doc.metadata.update({
                "route_strategy": analysis.recommended_strategy.value,
                "query_complexity": analysis.query_complexity,
                "route_confidence": analysis.confidence
            })
        
        return documents
    
    def _update_route_stats(self, strategy: SearchStrategy):
        """更新路由统计"""
        self.route_stats["total_queries"] += 1
        
        if strategy == SearchStrategy.HYBRID_TRADITIONAL:
            self.route_stats["traditional_count"] += 1
        elif strategy == SearchStrategy.GRAPH_RAG:
            self.route_stats["graph_rag_count"] += 1
        elif strategy == SearchStrategy.COMBINED:
            self.route_stats["combined_count"] += 1
    
    def get_route_statistics(self) -> Dict[str, Any]:
        """获取路由统计信息"""
        total = self.route_stats["total_queries"]
        if total == 0:
            return self.route_stats
        
        return {
            **self.route_stats,
            "traditional_ratio": self.route_stats["traditional_count"] / total,
            "graph_rag_ratio": self.route_stats["graph_rag_count"] / total,
            "combined_ratio": self.route_stats["combined_count"] / total
        }
    
    def explain_routing_decision(self, query: str) -> str:
        """解释路由决策过程"""
        analysis = self.analyze_query(query)
        
        explanation = f""" 
        查询路由分析报告
        
        查询：{query}
        
        特征分析：
        - 复杂度：{analysis.query_complexity:.2f} ({'简单' if analysis.query_complexity < 0.4 else '中等' if analysis.query_complexity < 0.8 else '复杂'})
        - 关系密集度：{analysis.relationship_intensity:.2f} ({'单一实体' if analysis.relationship_intensity < 0.4 else '实体关系' if analysis.relationship_intensity < 0.8 else '复杂关系网络'})
        - 推理需求：{'是' if analysis.reasoning_required else '否'}
        - 实体数量：{analysis.entity_count}
        
        推荐策略：{analysis.recommended_strategy.value}
        置信度：{analysis.confidence:.2f}
        
        决策理由：{analysis.reasoning}
        """
        
        return explanation

 