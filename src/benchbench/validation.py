"""
Validation interface for benchmark tasks.

Each task folder can contain a `validate.py` with an async `validate` function:

    from benchbench.validation import ValidationResult
    from benchbench.models import get_async_client, Model

    async def validate(output: str) -> ValidationResult:
        # Simple check
        if "expected answer" in output:
            return ValidationResult(passed=True, score=1.0)
        return ValidationResult(passed=False, score=0.0, reason="Missing expected answer")

    async def validate(output: str) -> ValidationResult:
        # LLM-as-judge
        client = get_async_client()
        result = await client.chat.completions.create(
            model=Model.GPT_4O_MINI,
            response_model=JudgeResponse,
            messages=[{"role": "user", "content": f"Evaluate: {output}"}]
        )
        return ValidationResult(passed=result.score > 0.5, score=result.score)
"""

from collections.abc import Awaitable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
import importlib.util
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a model output."""

    passed: bool
    score: float = 1.0  # 0.0 to 1.0
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ValidateFn(Protocol):
    """Protocol for async validation functions."""

    def __call__(self, output: str) -> Awaitable[ValidationResult]: ...


def load_validator(task_dir: Path) -> ValidateFn | None:
    """
    Load the validate function from a task's validate.py.

    Returns None if no validate.py exists or it doesn't have a validate function.
    """
    validate_path = task_dir / "validate.py"
    if not validate_path.exists():
        logger.debug(f"No validate.py found in {task_dir}")
        return None

    spec = importlib.util.spec_from_file_location("validate", validate_path)
    if spec is None or spec.loader is None:
        logger.error(f"Could not load spec for {validate_path}")
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    validate_fn = getattr(module, "validate", None)
    if validate_fn is None:
        logger.error(f"No 'validate' function found in {validate_path}")
        return None

    return validate_fn
