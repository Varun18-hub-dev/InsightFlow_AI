import os
from pathlib import Path

import structlog

logger = structlog.get_logger()


def resolve_prompts_dir() -> Path:
    """Resolve the prompts directory across development and Docker environments."""
    if os.environ.get("PROMPTS_DIR"):
        p = Path(os.environ["PROMPTS_DIR"])
        if p.exists() and p.is_dir():
            return p

    candidates = [
        # 1. backend/prompts (inside backend folder or /app/prompts in Docker container)
        Path(__file__).resolve().parent.parent.parent / "prompts",
        # 2. repo root prompts/
        Path(__file__).resolve().parent.parent.parent.parent / "prompts",
        # 3. app/prompts (inside package)
        Path(__file__).resolve().parent.parent / "prompts",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    return candidates[0]


PROMPTS_DIR = resolve_prompts_dir()


class PromptManager:
    """Loads and renders prompt templates from the prompts/ directory."""

    def __init__(self, prompts_dir: Path | None = None):
        self._prompts_dir = prompts_dir
        self._cache: dict[str, str] = {}

    @property
    def base_dir(self) -> Path:
        return self._prompts_dir or PROMPTS_DIR

    def get_prompt(self, category: str, prompt_name: str) -> str:
        """Load a prompt from prompts/{category}/{prompt_name}.txt"""
        key = f"{category}/{prompt_name}"
        if key not in self._cache:
            path = self.base_dir / category / f"{prompt_name}.txt"
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

    def validate_required_prompts(self) -> list[str]:
        """Validate that all core production prompts exist and are non-empty."""
        required = [
            ("rag", "system"),
            ("rag", "answer"),
        ]
        missing = []
        for cat, name in required:
            content = self.get_prompt(cat, name)
            if not content.strip():
                missing.append(f"{cat}/{name}")
        return missing


prompt_manager = PromptManager()


def validate_prompts_on_startup() -> None:
    """Verify core prompt templates are loaded on startup."""
    from app.core.config import settings

    missing = prompt_manager.validate_required_prompts()
    if missing:
        msg = f"Missing or empty required prompt files: {', '.join(missing)} in {prompt_manager.base_dir}"
        logger.error("missing_required_prompts", missing=missing, prompts_dir=str(prompt_manager.base_dir))
        if settings.ENVIRONMENT == "production":
            raise RuntimeError(msg)
    else:
        logger.info("prompts_validated", prompts_dir=str(prompt_manager.base_dir))

