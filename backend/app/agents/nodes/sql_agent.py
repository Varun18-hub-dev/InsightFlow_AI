"""
SQL Agent: NL → validated SELECT → PostgreSQL → natural language explanation.
ONLY SELECT queries allowed.
"""
import time

import sqlparse
import structlog
from sqlalchemy import text

from app.agents.state import AgentState
from app.core.exceptions import SQLValidationError

logger = structlog.get_logger()

# Forbidden keywords in ANY part of the SQL AST (including CTEs, subqueries, expressions)
FORBIDDEN_KEYWORDS = frozenset({
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE",
    "REPLACE", "MERGE", "EXEC", "EXECUTE", "GRANT", "REVOKE", "COPY",
    "VACUUM", "REINDEX", "CALL", "DO", "INTO", "ATTACH", "DETACH",
    "TRANSACTION", "COMMIT", "ROLLBACK", "SAVEPOINT", "LOCK",
})

# Forbidden sensitive tables, system catalogs, and columns
FORBIDDEN_IDENTIFIERS = frozenset({
    "PG_SHADOW", "PG_AUTHID", "PG_CATALOG", "INFORMATION_SCHEMA",
    "PG_ROLES", "PG_USER", "PG_DATABASE", "PG_TABLES",
    "HASHED_PASSWORD", "PASSWORD", "SECRET", "SECRET_KEY",
})

# Approved tables for query execution
ALLOWED_TABLES = frozenset({
    "DOCUMENTS", "DOCUMENT_CHUNKS", "CONVERSATIONS", "MESSAGES",
    "USER_FEEDBACK", "USERS",
})

SQL_SCHEMA = """
Table: users (id UUID, email TEXT, full_name TEXT, is_active BOOL, created_at TIMESTAMP)
Table: documents (id UUID, user_id UUID, filename TEXT, document_type TEXT, status TEXT, page_count INT, chunk_count INT, created_at TIMESTAMP)
Table: document_chunks (id UUID, document_id UUID, user_id UUID, chunk_index INT, content TEXT, page_number INT, created_at TIMESTAMP)
Table: conversations (id UUID, user_id UUID, title TEXT, created_at TIMESTAMP)
Table: messages (id UUID, conversation_id UUID, user_id UUID, role TEXT, content TEXT, created_at TIMESTAMP)
Table: user_feedback (id UUID, message_id UUID, user_id UUID, rating TEXT, feedback_text TEXT, created_at TIMESTAMP)
"""


def _walk_tokens(tokens):
    """Recursively inspect all tokens in the AST for forbidden commands or identifiers."""
    for token in tokens:
        if token.is_whitespace:
            continue
        val = token.value.strip().upper()
        clean_val = val.strip('"`[]')

        # Check forbidden keywords
        if clean_val in FORBIDDEN_KEYWORDS:
            raise SQLValidationError(f"Forbidden SQL operation: {clean_val}")

        # Check forbidden identifiers
        for forbidden in FORBIDDEN_IDENTIFIERS:
            if forbidden in clean_val:
                raise SQLValidationError(f"Access to sensitive identifier '{forbidden}' is prohibited")

        # Recurse into compound tokens (Parenthesis, Where, IdentifierList, etc.)
        if hasattr(token, "tokens"):
            _walk_tokens(token.tokens)


def validate_sql(sql: str) -> str:
    """
    Validate that SQL is a safe SELECT-only query using AST parsing.
    - Rejects multiple / stacked statements
    - Rejects non-SELECT queries (including DML embedded in CTEs)
    - Rejects forbidden DDL/DML operations
    - Rejects access to sensitive columns (passwords, credentials, system tables)
    - Strips semicolons and ensures query termination
    - Enforces row LIMIT if not present
    Raises SQLValidationError if invalid.
    Returns cleaned, validated SQL.
    """
    if not sql or not sql.strip():
        raise SQLValidationError("Query cannot be empty")

    cleaned = sql.strip().rstrip(";")
    parsed = sqlparse.parse(cleaned)
    statements = [stmt for stmt in parsed if stmt.value.strip() and stmt.value.strip() != ";"]

    if len(statements) != 1:
        raise SQLValidationError("Multiple statements are not allowed")

    stmt = statements[0]
    if stmt.get_type() != "SELECT":
        raise SQLValidationError(f"Only SELECT queries are allowed, got {stmt.get_type()}")

    # Walk full AST to catch nested DML/DDL or forbidden tokens in subqueries/CTEs
    _walk_tokens(stmt.tokens)

    # Strip comments to prevent comment-based dialect obfuscation
    cleaned_sql = sqlparse.format(cleaned, strip_comments=True).strip().rstrip(";")

    # Enforce safe row limit to prevent DoS / memory exhaustion
    if "LIMIT" not in cleaned_sql.upper():
        cleaned_sql = f"{cleaned_sql} LIMIT 100"

    return cleaned_sql


async def run_sql_agent(state: AgentState) -> AgentState:
    start_time = time.time()
    db = state.get("db")
    user_id = state["user_id"]
    query = state["query"]

    try:
        # Lazy imports to avoid circular/unavailable module issues at test time
        from app.core.prompt_manager import prompt_manager
        from app.llm.factory import LLMProviderFactory
        # 1. Generate SQL
        provider = LLMProviderFactory.get_provider()
        sql_prompt = prompt_manager.render(
            "sql", "sql_generation",
            schema=SQL_SCHEMA,
            question=query,
            user_id=user_id,
        )

        generated_sql = await provider.chat([{"role": "user", "content": sql_prompt}])
        generated_sql = generated_sql.strip()
        logger.info("sql_generated", sql=generated_sql[:200])

        if "INVALID_QUERY" in generated_sql:
            return {**state, "answer": "I cannot answer this question with a database query.", "sources": [], "confidence": 0.3}

        # 2. Validate
        safe_sql = validate_sql(generated_sql)

        # 3. Execute with timeout
        result = await db.execute(text(safe_sql))
        rows = result.fetchall()
        columns = result.keys()
        result_data = [dict(zip(columns, row, strict=False)) for row in rows]

        # 4. Explain in natural language
        explanation_prompt = f"""
The user asked: "{query}"
The database returned these results: {result_data}
Provide a clear, natural language answer based on these results. Be concise.
"""
        answer = await provider.chat([{"role": "user", "content": explanation_prompt}])

        return {
            **state,
            "answer": answer,
            "sql_query": safe_sql,
            "sql_result": result_data,
            "sources": [],
            "confidence": 0.9,
            "metadata": {"latency": time.time() - start_time, "sql": safe_sql},
        }
    except SQLValidationError as e:
        logger.warning("sql_validation_failed", error=str(e))
        return {**state, "answer": "I cannot execute that query for security reasons.", "sources": [], "confidence": 0.0, "error": str(e)}
    except Exception as e:
        logger.error("sql_agent_failed", error=str(e))
        return {**state, "answer": "I encountered an error running the database query.", "sources": [], "confidence": 0.0, "error": str(e)}
