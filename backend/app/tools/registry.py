from app.tools.document_metadata import create_document_metadata_tool
from app.tools.document_search import create_document_search_tool


def get_all_tools(user_id: str, db):
    return [
        create_document_search_tool(user_id, db),
        create_document_metadata_tool(user_id, db),
    ]
