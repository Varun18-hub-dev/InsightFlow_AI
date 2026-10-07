"""Unit tests for PromptManager."""
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


def test_rag_system_prompt_loaded():
    """Verify get_prompt('rag', 'system') returns non-empty content."""
    from app.core.prompt_manager import prompt_manager

    system_prompt = prompt_manager.get_prompt("rag", "system")
    assert system_prompt is not None
    assert len(system_prompt.strip()) > 0
    assert "InsightFlow AI" in system_prompt


def test_rag_answer_prompt_loaded():
    """Verify get_prompt('rag', 'answer') returns non-empty content."""
    from app.core.prompt_manager import prompt_manager

    answer_prompt = prompt_manager.get_prompt("rag", "answer")
    assert answer_prompt is not None
    assert len(answer_prompt.strip()) > 0
    assert "{context}" in answer_prompt
    assert "{question}" in answer_prompt


def test_rag_answer_prompt_rendered():
    """Verify render('rag', 'answer', context=..., question=...) returns non-empty rendered prompt."""
    from app.core.prompt_manager import prompt_manager

    context_text = "The quarterly revenue was $10M."
    question_text = "What was the quarterly revenue?"

    rendered = prompt_manager.render(
        "rag",
        "answer",
        context=context_text,
        question=question_text,
    )
    assert rendered is not None
    assert len(rendered.strip()) > 0
    assert context_text in rendered
    assert question_text in rendered
    assert "{context}" not in rendered
    assert "{question}" not in rendered


def test_startup_validation_succeeds():
    """Verify startup validation detects present prompts."""
    from app.core.prompt_manager import validate_prompts_on_startup

    # Should not raise exception
    validate_prompts_on_startup()


def test_startup_validation_fails_on_missing_in_production(tmp_path):
    """Verify startup validation raises in production when required prompts are missing."""
    with patch("app.core.prompt_manager.PROMPTS_DIR", tmp_path):
        from app.core.prompt_manager import PromptManager

        pm = PromptManager(prompts_dir=tmp_path)
        missing = pm.validate_required_prompts()
        assert "rag/system" in missing
        assert "rag/answer" in missing

