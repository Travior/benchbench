"""
Task runner: execute benchmark tasks against LLM models.

Supports concurrent execution via asyncio.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from benchbench.execution import (
    ExecutionContext,
    RunConfig,
    build_messages_for_litellm,
    resolve_executor_for_task,
)
from benchbench.models import Model
from benchbench.task import Task, TaskRun
from benchbench.validation import ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class TaskRunner:
    """Executes benchmark tasks against LLM models."""

    config: RunConfig = field(default_factory=RunConfig)
    default_executor_path: Path | None = None

    async def run_task(self, task: Task, model: Model) -> TaskRun:
        """
        Execute a single task against a single model.

        Returns TaskRun with results or error information.
        """
        start_time = time.perf_counter()

        executor_fn = resolve_executor_for_task(task, self.default_executor_path)
        messages = build_messages_for_litellm(task.messages)
        ctx = ExecutionContext(
            task=task,
            model=model,
            run_config=self.config,
            messages=messages,
        )

        try:
            output = await executor_fn(ctx)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Run validation if available
            validation: ValidationResult | None = None
            error: str | None = None
            if task.validator is not None:
                try:
                    validation = await task.validator(output)
                except Exception as e:
                    logger.error(f"Validation failed for task {task.task_id}: {e}")
                    validation = ValidationResult(
                        passed=False, score=0.0, reason=f"Validation error: {e}"
                    )
                    error = str(e)

            return TaskRun(
                task_id=task.task_id,
                model=model.value,
                output=output,
                validation=validation,
                duration_ms=duration_ms,
                error=error,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Task execution failed for {task.task_id} on {model}: {e}")
            return TaskRun(
                task_id=task.task_id,
                model=model.value,
                output="",
                duration_ms=duration_ms,
                error=str(e),
            )

    async def run_tasks(self, tasks: list[Task], models: list[Model]) -> list[TaskRun]:
        """
        Run multiple tasks against multiple models concurrently.

        Uses a semaphore to limit concurrent API calls.
        """
        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async def run_with_semaphore(task: Task, model: Model) -> TaskRun:
            async with semaphore:
                logger.info(f"Running task {task.display_name} on {model.value}")
                return await self.run_task(task, model)

        # Create all task/model combinations
        coroutines = [
            run_with_semaphore(task, model) for task in tasks for model in models
        ]

        results = await asyncio.gather(*coroutines)
        return list(results)

    def run_tasks_sync(self, tasks: list[Task], models: list[Model]) -> list[TaskRun]:
        """Synchronous wrapper for run_tasks."""
        return asyncio.run(self.run_tasks(tasks, models))
