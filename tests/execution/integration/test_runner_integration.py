import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchbench.execution import RunConfig
from benchbench.models import Model
from benchbench.parser import Message, Roles
from benchbench.runner import TaskRunner
from benchbench.validation import ValidationResult
from tests.execution.utils.helpers import (
    echo_execute_source,
    minimal_task,
    raising_execute_source,
)

pytestmark = pytest.mark.integration


def test_run_task_with_task_executor():
    async def exec_fn(ctx):
        return "direct"

    task = minimal_task(executor=exec_fn)
    runner = TaskRunner()
    result = asyncio.run(runner.run_task(task, Model.GPT_51_NANO_OR))
    assert result.error is None
    assert result.output == "direct"


def test_run_task_with_default_executor_path(tmp_path):
    exec_py = tmp_path / "exec.py"
    exec_py.write_text(echo_execute_source("|r"))
    task = minimal_task(
        path=tmp_path / "taskdir",
        messages=[Message(role=Roles.user, content="ping")],
    )
    runner = TaskRunner(default_executor_path=exec_py)
    result = asyncio.run(runner.run_task(task, Model.GPT_51_NANO_OR))
    assert result.error is None
    assert result.output == "ping|r"


def test_run_task_executor_raises(tmp_path):
    exec_py = tmp_path / "exec.py"
    exec_py.write_text(raising_execute_source("intentional"))
    task = minimal_task(path=tmp_path / "t")
    runner = TaskRunner(default_executor_path=exec_py)
    result = asyncio.run(runner.run_task(task, Model.GPT_51_NANO_OR))
    assert result.output == ""
    assert result.error is not None
    assert "intentional" in result.error


async def _noop_validate(output: str) -> ValidationResult:
    return ValidationResult(passed=True, score=1.0)


def test_run_task_runs_validator():
    async def exec_fn(ctx):
        return "out"

    task = minimal_task(executor=exec_fn, validator=_noop_validate)
    runner = TaskRunner()
    result = asyncio.run(runner.run_task(task, Model.GPT_51_NANO_OR))
    assert result.error is None
    assert result.output == "out"
    assert result.validation is not None
    assert result.validation.passed is True


def test_run_tasks_concurrency_smoke():
    async def exec_fn(ctx):
        return "x"

    tasks = [
        minimal_task(executor=exec_fn, path=Path(f"/t/{i}"), id_chain=["a", str(i)])
        for i in range(2)
    ]
    runner = TaskRunner(config=RunConfig(max_concurrency=2))
    results = asyncio.run(runner.run_tasks(tasks, [Model.GPT_51_NANO_OR]))
    assert len(results) == 2
    assert {r.output for r in results} == {"x"}


def test_run_task_default_execute_mocked():
    task = minimal_task()
    runner = TaskRunner(config=RunConfig())
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "via-default"

    async def run():
        with patch(
            "benchbench.execution.litellm.acompletion", new_callable=AsyncMock
        ) as m:
            m.return_value = mock_resp
            return await runner.run_task(task, Model.GPT_51_NANO_OR)

    result = asyncio.run(run())
    assert result.error is None
    assert result.output == "via-default"
