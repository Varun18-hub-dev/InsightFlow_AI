"""
PromptManager - loads prompts from the prompts/ directory at project root.
"""
from pathlib import Path

import structlog

logger = structlog.get_logger()

PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts"


class PromptManager:
    """Loads and renders prompt templates from the prompts/ directory."""

    def __init__(self):
        self._cache: dict[str, str] = {}

    def get_prompt(self, category: str, prompt_name: str) -> str:
        """Load a prompt from prompts/{category}/{prompt_name}.txt"""
        key = f"{category}/{prompt_name}"
        if key not in self._cache:
            path = PROMPTS_DIR / category / f"{prompt_name}.txt"
            if path.exists():
                self._cache[key] = path.read_text(encoding="utf-8")
            else:
                logger.warning("prompt_not_found", path=str(path))
                self._cache[key] = ""
        return self._cache[key]

    def render(self, category: str, prompt_name: str, **variables) -> str:
        """Load and render a prompt with variable substitution.

        Uses str.format_map to substitute {variable} placeholders.
        Missing variables leave the placeholder unchanged (no KeyError).
        """
        template = self.get_prompt(category, prompt_name)
        if not template:
            return ""
        try:
            # format_map with a default-returning dict handles missing keys gracefully
            class SafeDict(dict):
                def __missing__(self, key):
                    return "{" + key + "}"

            return template.format_map(SafeDict(**variables))
        except Exception as e:
            logger.warning("prompt_render_error", error=str(e))
            return template


prompt_manager = PromptManager()
