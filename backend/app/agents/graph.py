"""
LangGraph StateGraph — main orchestration for InsightFlow AI.
"""
import structlog
from langgraph.graph import END, StateGraph

from app.agents.nodes.comparison_agent import run_comparison_agent
from app.agents.nodes.intent_classifier import classify_intent
from app.agents.nodes.rag_agent import run_rag_agent
from app.agents.nodes.response_generator import run_response_generator
from app.agents.nodes.sql_agent import run_sql_agent
from app.agents.nodes.summary_agent import run_summary_agent
from app.agents.nodes.verification import run_verification
from app.agents.state import AgentState

logger = structlog.get_logger()

def _route_by_intent(state: AgentState) -> str:
    intent = state.get("intent", "DOCUMENT_QA")
    routes = {
        "DOCUMENT_QA": "rag_agent",
        "DOCUMENT_SUMMARY": "summary_agent",
        "DOCUMENT_COMPARISON": "comparison_agent",
        "DATABASE_QUERY": "sql_agent",
        "GENERAL_QUERY": "rag_agent",
    }
    return routes.get(intent, "rag_agent")

def build_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("rag_agent", run_rag_agent)
    workflow.add_node("sql_agent", run_sql_agent)
    workflow.add_node("summary_agent", run_summary_agent)
    workflow.add_node("comparison_agent", run_comparison_agent)
    workflow.add_node("generate_response", run_response_generator)
    workflow.add_node("verify", run_verification)

    # Entry point
    workflow.set_entry_point("classify_intent")

    # Conditional routing after classification
    workflow.add_conditional_edges(
        "classify_intent",
        _route_by_intent,
        {
            "rag_agent": "rag_agent",
            "sql_agent": "sql_agent",
            "summary_agent": "summary_agent",
            "comparison_agent": "comparison_agent",
        }
    )

    # All agents → response generator → verification → END
    for node in ["rag_agent", "sql_agent", "summary_agent", "comparison_agent"]:
        workflow.add_edge(node, "generate_response")

    workflow.add_edge("generate_response", "verify")
    workflow.add_edge("verify", END)

    return workflow.compile()

# Compiled graph singleton
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph

async def run_graph(
    query: str,
    user_id: str,
    db,
    conversation_id: str = None,
    document_ids: list[str] = None,
    messages: list[dict] = None
) -> AgentState:
    graph = get_graph()
    initial_state = AgentState(
        query=query,
        user_id=user_id,
        conversation_id=conversation_id,
        db=db,
        document_ids=document_ids or [],
        messages=messages or [],
        intent=None,
        retrieved_chunks=[],
        reranked_chunks=[],
        context="",
        answer=None,
        sources=[],
        confidence=0.0,
        error=None,
        metadata={},
        sql_query=None,
        sql_result=None
    )

    result = await graph.ainvoke(initial_state)
    return result
