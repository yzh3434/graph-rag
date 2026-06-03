"""
Corrective-RAG 生成编排（crag_v1）

B 路线最小侵入：包住现成 GenerationIntegrationModule，不改其 generate_adaptive_answer。
- LLM① 自评：够则直接返回答案；不足则返回 {"insufficient": true, "search_query": ...}
- 不足 → Tavily 网络检索 → 把网络上下文当一篇 Document 追加 → 复用原生成做 LLM②
- 一轮硬上限；任何失败都降级，绝不硬失败
"""

import os
import re
import json
import time
import logging
from typing import List, Dict, Tuple, Optional, Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_JSON_OBJ_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_sufficiency_response(text: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    解析 LLM① 输出。

    返回 (is_insufficient, search_query, answer)：
      - 有效不足信号：(True, search_query, None)
      - 普通答案：     (False, None, answer_text)
      - 纯 JSON 但非有效不足信号：(False, None, None)  ← 交上层兜底重新生成
    """
    if text is None:
        return False, None, None
    raw = text.strip()
    if not raw:
        return False, None, None

    # 去 ```json ... ``` 围栏
    fenced = raw
    if fenced.startswith("```"):
        fenced = re.sub(r"^```[a-zA-Z]*\s*", "", fenced)
        fenced = re.sub(r"\s*```$", "", fenced).strip()

    def _try_signal(s: str) -> Optional[Tuple[bool, Optional[str], Optional[str]]]:
        try:
            data = json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict) or "insufficient" not in data:
            return None
        if data.get("insufficient") is True:
            sq = data.get("search_query")
            if isinstance(sq, str) and sq.strip():
                return True, sq.strip(), None
            return False, None, None  # 缺 search_query，无效信号
        # insufficient=false：是纯 JSON，无自然语言答案
        return False, None, None

    # 整体就是 JSON
    sig = _try_signal(fenced)
    if sig is not None:
        return sig

    # 文本里嵌了一个 JSON 对象（宽松）
    m = _JSON_OBJ_RE.search(fenced)
    if m:
        sig = _try_signal(m.group(0))
        if sig is not None and sig[0] is True:
            return sig

    # 默认：当答案（保护领域内，不误触发）
    return False, None, raw


# 与原 generate_adaptive_answer 一致的上下文拼接（小段复制，换取零侵入隔离）
def _build_kb_context(documents: List[Document]) -> str:
    parts = []
    for doc in documents:
        content = (doc.page_content or "").strip()
        if not content:
            continue
        level = doc.metadata.get("retrieval_level", "")
        parts.append(f"[{level.upper()}] {content}" if level else content)
    return "\n\n".join(parts)


def _default_tavily_search(query: str, max_results: int = 5, timeout: int = 10) -> List[str]:
    """返回 snippet 文本列表；无 key / 报错 / 空结果由调用方处理（这里照常抛/返空）"""
    from tavily import TavilyClient
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 TAVILY_API_KEY")
    client = TavilyClient(api_key=api_key)
    resp = client.search(query=query, max_results=max_results, search_depth="basic")
    results = resp.get("results", []) if isinstance(resp, dict) else []
    return [r.get("content", "").strip() for r in results if r.get("content", "").strip()]


_SUFFICIENCY_PROMPT = """作为一位专业的烹饪助手，请基于以下信息回答用户的问题。

检索到的相关信息：
{context}

用户问题：{question}

判断规则（务必先判断）：
- 若上述检索信息明显**不足以**回答该问题（例如问的菜在信息里根本没有），
  请**只**输出一行 JSON，不要输出其它任何内容：
  {{"insufficient": true, "search_query": "<你认为最有效的中文网络检索关键词>"}}
- 否则，请**不要**输出任何 JSON，直接给出准确、实用的回答。根据问题性质：
  - 询问多个菜品 → 清晰列表
  - 询问具体做法 → 详细步骤
  - 一般性咨询 → 综合性回答

回答："""


class CRAGGenerator:
    """Corrective-RAG 编排器。接口：generate(question, documents) -> (answer, meta)"""

    def __init__(self, generation_module, config, _tavily_fn=None):
        self.gen = generation_module
        self.config = config
        # 依赖注入便于测试；默认用真实 Tavily
        self._tavily_fn = _tavily_fn or _default_tavily_search

    def _empty_meta(self) -> Dict[str, Any]:
        return {"triggered": False, "web_query": None, "web_context": "",
                "n_web_results": 0, "web_failed": False}

    def generate(self, question: str, documents: List[Document]) -> Tuple[str, Dict[str, Any]]:
        meta = self._empty_meta()
        kb_context = _build_kb_context(documents)
        prompt = _SUFFICIENCY_PROMPT.format(context=kb_context, question=question)

        # —— LLM 调用① ——
        try:
            resp = self.gen.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=getattr(self.gen, "temperature", 0.1),
                max_tokens=getattr(self.gen, "max_tokens", 2048),
            )
            text = resp.choices[0].message.content
        except Exception as e:
            logger.error(f"CRAG LLM① 失败，降级原生成: {e}")
            return self.gen.generate_adaptive_answer(question, documents), meta

        is_insufficient, search_query, answer = parse_sufficiency_response(text)

        # 充分 → 直接用 LLM① 的答案（happy path：1 次调用）
        if not is_insufficient and answer is not None:
            return answer, meta

        # 纯 JSON 非有效信号（answer is None 且非不足）→ 兜底走原生成
        if not is_insufficient:
            return self.gen.generate_adaptive_answer(question, documents), meta

        # —— 不足：网络检索 ——
        meta["triggered"] = True
        meta["web_query"] = search_query
        snippets: List[str] = []
        try:
            snippets = self._tavily_fn(
                search_query,
                max_results=self.config.tavily_max_results,
                timeout=self.config.tavily_timeout_seconds,
            ) or []
        except Exception as e:
            logger.warning(f"Tavily 失败，降级: {e}")
            meta["web_failed"] = True

        if not snippets:
            meta["web_failed"] = True
            note = "（注：知识库未收录该内容，以下为基于通用知识的参考。）"
            return self.gen.generate_adaptive_answer(
                f"{question}\n{note}", documents), meta

        web_context = "\n\n".join(snippets)
        meta["web_context"] = web_context
        meta["n_web_results"] = len(snippets)

        # —— LLM 调用②：网络内容当一篇 Document 追加，复用原生成（不改其代码）——
        web_doc = Document(
            page_content=f"以下内容来自网络检索（用户问题在本地知识库未收录）：\n{web_context}",
            metadata={"retrieval_level": "web", "source": "tavily"},
        )
        answer = self.gen.generate_adaptive_answer(question, list(documents) + [web_doc])
        return answer, meta
