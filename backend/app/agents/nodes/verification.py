from app.agents.state import AgentState

NO_INFO_PHRASES = [
    "could not find",
    "not found in the provided",
    "no information",
    "insufficient information",
    "unable to find",
    "not mentioned"
]

def run_verification(state: AgentState) -> AgentState:
    """Basic grounding check — detect likely hallucination or missing context."""
    answer = state.get("answer", "")
    sources = state.get("sources", [])
    confidence = state.get("confidence", 0.5)

    answer_lower = answer.lower()

    # If answer explicitly says no info found
    if any(phrase in answer_lower for phrase in NO_INFO_PHRASES):
        confidence = min(confidence, 0.2)

    # If answer has content but no sources
    if answer and len(answer) > 100 and not sources:
        intent = state.get("intent", "")
        if intent in ("DOCUMENT_QA", "DOCUMENT_COMPARISON"):
            confidence = min(confidence, 0.4)

    return {**state, "confidence": round(confidence, 4)}
