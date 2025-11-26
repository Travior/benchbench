"""
Validation interface for benchmark tasks.

Each task folder can contain a `validate.py` with an async `validate` function:

    from benchbench.validation import ValidationResult, pending_manual
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

    async def validate(output: str) -> ValidationResult:
        # Manual grading - will be graded via `bench grade`
        return pending_manual("Does the response correctly explain the algorithm?")
"""

from collections.abc import Awaitable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol
import importlib.util
import logging

logger = logging.getLogger(__name__)


class Score(Enum):
    """Three-point grading scale for manual validation."""
    FAIL = 0.0
    PARTIAL = 0.5
    PASS = 1.0


@dataclass
class ValidationResult:
    """Result of validating a model output."""

    passed: bool | None  # None when pending manual grading
    score: float = 1.0  # 0.0, 0.5, or 1.0
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    pending: bool = False  # True if awaiting manual grading
    rubric: str | None = None  # Grading instructions for manual review


def pending_manual(rubric: str | None = None) -> ValidationResult:
    """
    Create a validation result that requires manual grading.
    
    Use this in validate.py when human judgment is needed:
    
        async def validate(output: str) -> ValidationResult:
            return pending_manual("Does the response correctly identify the root cause?")
    
    Args:
        rubric: Instructions for the human grader explaining what to look for.
        
    Returns:
        ValidationResult with pending=True, to be graded via `bench grade`.
    """
    return ValidationResult(
        passed=None,
        score=0.0,
        pending=True,
        rubric=rubric,
    )


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
