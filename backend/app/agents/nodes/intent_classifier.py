"""
Intent classifier: rule-based first, LLM fallback.
"""
import re

import structlog

from app.agents.state import AgentState

logger = structlog.get_logger()

INTENT_LABELS = ["DOCUMENT_QA", "DOCUMENT_SUMMARY", "DOCUMENT_COMPARISON", "DATABASE_QUERY", "GENERAL_QUERY"]

COMPARISON_PATTERNS = [r'\bcompare\b', r'\bvs\b', r'\bversus\b', r'\bdifference\b', r'\bdifferences\b', r'\bcontrast\b']
SUMMARY_PATTERNS = [r'\bsummar', r'\boverview\b', r'\bbrief\b', r'\bdescribe the document\b']
DATABASE_PATTERNS = [r'\bhow many\b', r'\bcount\b', r'\blist all\b', r'\bshow me all\b', r'\btotal number\b', r'\bwhen was.*uploaded\b', r'\brecent.*document\b']

def _rule_based_classify(query: str) -> str | None:
    q = query.lower()
    for pattern in COMPARISON_PATTERNS:
        if re.search(pattern, q):
            return "DOCUMENT_COMPARISON"
    for pattern in SUMMARY_PATTERNS:
        if re.search(pattern, q):
            return "DOCUMENT_SUMMARY"
    for pattern in DATABASE_PATTERNS:
        if re.search(pattern, q):
            return "DATABASE_QUERY"
    return None

async def classify_intent(state: AgentState) -> AgentState:
    query = state["query"]

    # Step 1: Rule-based
    intent = _rule_based_classify(query)

    # Step 2: LLM fallback
    if intent is None:
        try:
            from app.core.prompt_manager import prompt_manager
            from app.llm.factory import LLMProviderFactory

            provider = LLMProviderFactory.get_provider()
            doc_list = ", ".join(state.get("document_ids", [])) or "none"
            prompt = prompt_manager.render("classification", "intent", query=query, document_list=doc_list)

            response = await provider.chat([{"role": "user", "content": prompt}])
            response = response.strip().upper()
            if response in INTENT_LABELS:
                intent = response
            else:
                intent = "DOCUMENT_QA"
        except Exception as e:
            logger.warning("intent_llm_fallback_failed", error=str(e))
            intent = "DOCUMENT_QA"

    logger.info("intent_classified", query=query[:50], intent=intent)
    return {**state, "intent": intent}
