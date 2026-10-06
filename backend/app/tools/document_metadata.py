import json

from langchain.tools import Tool
from sqlalchemy import select

from app.db.models import Document


def create_document_metadata_tool(user_id: str, db):
    async def get_metadata(document_id: str) -> str:
        stmt = select(Document).where(Document.id == document_id, Document.user_id == user_id)
        result = await db.execute(stmt)
        doc = result.scalar_one_or_none()
        if not doc:
            return "Document not found."
        return json.dumps({
            "id": str(doc.id), "filename": doc.filename, "type": doc.document_type,
            "status": doc.status, "pages": doc.page_count, "chunks": doc.chunk_count,
            "created": str(doc.created_at)
        }, indent=2)
    return Tool(name="document_metadata", func=get_metadata, description="Get metadata for a document by ID", coroutine=get_metadata)
