"""Unit tests for PromptManager."""
import pytest
from pathlib import Path
from unittest.mock import patch


def test_render_with_variables(tmp_path):
    """Test that variables are substituted correctly."""
    rag_dir = tmp_path / "rag"
    rag_dir.mkdir()
    (rag_dir / "test.txt").write_text("Hello {name}, your query is: {query}")

    with patch("app.core.prompt_manager.PROMPTS_DIR", tmp_path):
        from app.core.prompt_manager import PromptManager
        pm = PromptManager()
        result = pm.render("rag", "test", name="Alice", query="what is RAG?")

    assert "Alice" in result
    assert "what is RAG?" in result


def test_missing_prompt_returns_empty_string(tmp_path):
    """Missing prompt file should return empty string, not raise."""
    with patch("app.core.prompt_manager.PROMPTS_DIR", tmp_path):
        from app.core.prompt_manager import PromptManager
        pm = PromptManager()
        result = pm.get_prompt("nonexistent", "prompt")

    assert result == ""


def test_prompt_caching(tmp_path):
    """Second access to same prompt should use cache."""
    cat_dir = tmp_path / "test_cat"
    cat_dir.mkdir()
    (cat_dir / "my_prompt.txt").write_text("Cached prompt content")

    with patch("app.core.prompt_manager.PROMPTS_DIR", tmp_path):
        from app.core.prompt_manager import PromptManager
        pm = PromptManager()
        first = pm.get_prompt("test_cat", "my_prompt")
        second = pm.get_prompt("test_cat", "my_prompt")

    assert first == second == "Cached prompt content"
    assert "test_cat/my_prompt" in pm._cache


def test_render_with_missing_variable(tmp_path):
    """Render with missing variable should return template (not raise)."""
    cat_dir = tmp_path / "rag"
    cat_dir.mkdir()
    (cat_dir / "answer.txt").write_text("Hello {name}, context: {context}")

    with patch("app.core.prompt_manager.PROMPTS_DIR", tmp_path):
        from app.core.prompt_manager import PromptManager
        pm = PromptManager()
        # Missing 'context' variable — should not raise
        result = pm.render("rag", "answer", name="Bob")

    # Should return original template when variable missing
    assert "Bob" in result or "{name}" in result  # Depends on partial format
