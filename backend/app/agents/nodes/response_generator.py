from app.agents.state import AgentState


def run_response_generator(state: AgentState) -> AgentState:
    """Assemble and format the final response."""
    answer = state.get("answer", "I was unable to generate a response.")
    sources = state.get("sources", [])

    # Deduplicate sources by document + page
    seen = set()
    deduped = []
    for s in sources:
        key = (s.get("document", ""), s.get("page", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    # Clean answer
    answer = answer.strip() if answer else "No response generated."

    return {**state, "answer": answer, "sources": deduped}
