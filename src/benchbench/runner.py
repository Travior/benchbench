"""
Task runner: execute benchmark tasks against LLM models.

Supports concurrent execution via asyncio.
"""

import asyncio
import time
import logging
from dataclasses import dataclass

import litellm

from benchbench.task import Task, TaskRun
from benchbench.models import Model
from benchbench.validation import ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""

    temperature: float = 0.0
    max_tokens: int | None = None
    max_concurrency: int = 5  # Max parallel API calls


class TaskRunner:
    """Executes benchmark tasks against LLM models."""

    def __init__(self, config: RunConfig | None = None):
        self.config = config or RunConfig()

    async def run_task(self, task: Task, model: Model) -> TaskRun:
        """
        Execute a single task against a single model.

        Returns TaskRun with results or error information.
        """
        start_time = time.perf_counter()

        # Convert task messages to litellm format
        messages = [{"role": msg.role.value, "content": msg.content} for msg in task.messages]

        try:
            response = await litellm.acompletion(
                model=model.value,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            output = response.choices[0].message.content or ""  # type: ignore[union-attr]
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Run validation if available
            validation: ValidationResult | None = None
            if task.validator is not None:
                try:
                    validation = await task.validator(output)
                except Exception as e:
                    logger.error(f"Validation failed for task {task.task_id}: {e}")
                    validation = ValidationResult(
                        passed=False, score=0.0, reason=f"Validation error: {e}"
                    )

            return TaskRun(
                task_id=task.task_id,
                model=model.value,
                output=output,
                validation=validation,
                duration_ms=duration_ms,
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

    async def run_tasks(
        self, tasks: list[Task], models: list[Model]
    ) -> list[TaskRun]:
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

    def run_tasks_sync(
        self, tasks: list[Task], models: list[Model]
    ) -> list[TaskRun]:
        """Synchronous wrapper for run_tasks."""
        return asyncio.run(self.run_tasks(tasks, models))
