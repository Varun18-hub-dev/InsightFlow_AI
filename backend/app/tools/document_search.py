from langchain.tools import Tool

from app.retrieval.hybrid_retriever import HybridRetriever


def create_document_search_tool(user_id: str, db):
    async def search(query: str) -> str:
        retriever = HybridRetriever(db)
        chunks = await retriever.retrieve(query, filters={"user_id": user_id}, final_top_k=5)
        if not chunks:
            return "No relevant documents found."
        return "\n---\n".join(f"[{c.metadata.get('filename','doc')} p.{c.metadata.get('page_number','?')}]\n{c.content}" for c in chunks)
    return Tool(name="document_search", func=search, description="Search uploaded documents for relevant information", coroutine=search)
