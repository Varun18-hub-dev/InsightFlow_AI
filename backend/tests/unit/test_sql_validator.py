"""Unit tests for SQL validator in sql_agent."""
import pytest

from app.agents.nodes.sql_agent import validate_sql
from app.core.exceptions import SQLValidationError


def test_valid_select_passes():
    sql = "SELECT * FROM documents WHERE user_id = 'abc'"
    result = validate_sql(sql)
    assert result.upper().startswith("SELECT")
    assert "LIMIT" in result.upper()


def test_valid_select_with_join():
    sql = "SELECT d.filename, COUNT(dc.id) FROM documents d JOIN document_chunks dc ON d.id = dc.document_id GROUP BY d.filename"
    result = validate_sql(sql)
    assert "SELECT" in result.upper()


def test_blocks_insert():
    with pytest.raises(SQLValidationError):
        validate_sql("INSERT INTO users VALUES ('1', 'test@test.com')")


def test_blocks_update():
    with pytest.raises(SQLValidationError):
        validate_sql("UPDATE users SET email = 'x' WHERE id = '1'")


def test_blocks_delete():
    with pytest.raises(SQLValidationError):
        validate_sql("DELETE FROM users WHERE id = '1'")


def test_blocks_drop():
    with pytest.raises(SQLValidationError):
        validate_sql("DROP TABLE users")


def test_blocks_alter():
    with pytest.raises(SQLValidationError):
        validate_sql("ALTER TABLE users ADD COLUMN foo TEXT")


def test_blocks_truncate():
    with pytest.raises(SQLValidationError):
        validate_sql("TRUNCATE TABLE users")


def test_case_insensitive_blocking():
    with pytest.raises(SQLValidationError):
        validate_sql("insert into users values ('1')")


def test_non_select_raises():
    with pytest.raises(SQLValidationError):
        validate_sql("EXEC sp_something")


def test_semicolon_stripped():
    sql = "SELECT 1;"
    result = validate_sql(sql)
    assert not result.endswith(";")


def test_blocks_stacked_queries():
    """Security: prevent SQL injection via multiple stacked queries."""
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT * FROM documents; DROP TABLE users;")


def test_blocks_cte_dml_injection():
    """Security: prevent DML embedded in Common Table Expressions (WITH clause)."""
    with pytest.raises(SQLValidationError):
        validate_sql("WITH deleted AS (DELETE FROM users RETURNING *) SELECT * FROM deleted")


def test_blocks_sensitive_auth_columns():
    """Security: prevent reading sensitive user passwords/credentials."""
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT id, email, hashed_password FROM users")


def test_blocks_system_catalogs():
    """Security: prevent reading PostgreSQL system catalog tables."""
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT * FROM pg_shadow")

    with pytest.raises(SQLValidationError):
        validate_sql("SELECT * FROM information_schema.tables")


def test_enforces_limit_on_unbounded_queries():
    """Security: prevent DoS by automatically appending LIMIT 100 if none specified."""
    result = validate_sql("SELECT filename FROM documents")
    assert "LIMIT 100" in result.upper()


def test_preserves_existing_limit():
    """Ensure user-specified LIMIT is preserved."""
    result = validate_sql("SELECT filename FROM documents LIMIT 10")
    assert "LIMIT 10" in result.upper()
