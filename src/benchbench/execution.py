"""
Pluggable task executors: how model output is produced for a benchmark task.

Optional per-task execute.py defines async def execute(ctx: ExecutionContext) -> str.
The default path uses LiteLLM completion (same behavior as before executors existed).
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import litellm

from benchbench.models import Model, get_model_config
from benchbench.parser import Message
from benchbench.task import Task

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""

    temperature: float = 0.0
    max_tokens: int | None = None
    max_concurrency: int = 5  # Max parallel API calls


@dataclass
class ExecutionContext:
    """Everything a custom executor may need without re-parsing the task directory."""

    task: Task
    model: Model
    run_config: RunConfig
    messages: list[dict[str, str]]  # litellm-shaped roles and content


class ExecutorFn(Protocol):
    """Protocol for async executor functions."""

    def __call__(self, ctx: ExecutionContext) -> Coroutine[Any, Any, str]: ...


async def default_execute(ctx: ExecutionContext) -> str:
    """Default executor: single LiteLLM completion with RunConfig and model extras."""
    response = await litellm.acompletion(
        model=ctx.model.value,
        messages=ctx.messages,
        temperature=ctx.run_config.temperature,
        max_tokens=ctx.run_config.max_tokens,
        reasoning={"enabled": True},
        timeout=6000,
        **get_model_config(ctx.model),
    )
    return response.choices[0].message.content or ""  # type: ignore[union-attr]


def load_executor(task_dir: Path) -> ExecutorFn | None:
    """
    Load execute function from task_dir/execute.py.

    Returns None if no execute.py or no callable execute.
    """
    execute_path = task_dir / "execute.py"
    if not execute_path.exists():
        logger.debug(f"No execute.py found in {task_dir}")
        return None

    spec = importlib.util.spec_from_file_location("execute", execute_path)
    if spec is None or spec.loader is None:
        logger.error(f"Could not load spec for {execute_path}")
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    execute_fn = getattr(module, "execute", None)
    if execute_fn is None:
        logger.error(f"No 'execute' function found in {execute_path}")
        return None

    return execute_fn


def resolve_executor_for_task(
    task: Task,
    default_executor_path: Path | None,
) -> ExecutorFn:
    """
    Choose executor for a run.

    Order: per-task execute.py > CLI default file > built-in LiteLLM.
    """
    if task.executor is not None:
        return task.executor
    if default_executor_path is not None:
        return _load_executor_from_path(default_executor_path)
    return default_execute


def _load_executor_from_path(path: Path) -> ExecutorFn:
    """Load execute() from an arbitrary path (for CLI --executor-path)."""
    spec = importlib.util.spec_from_file_location("executor_override", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load executor from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    execute_fn = getattr(module, "execute", None)
    if execute_fn is None:
        raise ValueError(f"No 'execute' function found in {path}")

    return execute_fn


def build_messages_for_litellm(messages: list[Message]) -> list[dict[str, str]]:
    return [{"role": msg.role.value, "content": msg.content} for msg in messages]
