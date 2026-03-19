import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchbench.execution import (
    ExecutionContext,
    RunConfig,
    build_messages_for_litellm,
    default_execute,
    load_executor,
    resolve_executor_for_task,
)
from benchbench.models import Model
from benchbench.parser import Message, Roles
from tests.execution.utils.helpers import echo_execute_source, minimal_task

pytestmark = pytest.mark.unit


def test_load_executor_missing_file(tmp_path):
    assert load_executor(tmp_path) is None


def test_load_executor_no_execute_attr(tmp_path):
    (tmp_path / "execute.py").write_text("x = 1\n")
    assert load_executor(tmp_path) is None


def test_load_executor_valid(tmp_path):
    (tmp_path / "execute.py").write_text(echo_execute_source("|x"))
    fn = load_executor(tmp_path)
    assert fn is not None
    task = minimal_task(
        messages=[Message(role=Roles.user, content="hi")],
        path=tmp_path,
    )
    ctx = ExecutionContext(
        task=task,
        model=Model.GPT_51_NANO_OR,
        run_config=RunConfig(),
        messages=[{"role": "user", "content": "hi"}],
    )
    assert asyncio.run(fn(ctx)) == "hi|x"


def test_resolve_executor_task_executor_wins(tmp_path):
    async def mine(ctx: ExecutionContext) -> str:
        return "mine"

    default_file = tmp_path / "def.py"
    default_file.write_text(echo_execute_source())
    task = minimal_task(executor=mine)
    fn = resolve_executor_for_task(task, default_file)
    ctx = ExecutionContext(
        task=task,
        model=Model.GPT_51_NANO_OR,
        run_config=RunConfig(),
        messages=[],
    )
    assert asyncio.run(fn(ctx)) == "mine"


def test_resolve_executor_default_path(tmp_path):
    task = minimal_task(executor=None)
    p = tmp_path / "exec.py"
    p.write_text(echo_execute_source("|d"))
    fn = resolve_executor_for_task(task, p)
    ctx = ExecutionContext(
        task=task,
        model=Model.GPT_51_NANO_OR,
        run_config=RunConfig(),
        messages=[{"role": "user", "content": "a"}],
    )
    assert asyncio.run(fn(ctx)) == "a|d"


def test_resolve_executor_fallback_default_execute():
    task = minimal_task(executor=None)
    fn = resolve_executor_for_task(task, None)
    assert fn is default_execute


def test_resolve_executor_bad_default_path(tmp_path):
    task = minimal_task(executor=None)
    # Missing file surfaces as FileNotFoundError from importlib loader
    with pytest.raises(FileNotFoundError):
        resolve_executor_for_task(task, tmp_path / "nope.py")


def test_resolve_executor_default_path_missing_execute(tmp_path):
    task = minimal_task(executor=None)
    bad = tmp_path / "bad.py"
    bad.write_text("x = 1\n")
    with pytest.raises(ValueError, match="No 'execute' function"):
        resolve_executor_for_task(task, bad)


def test_build_messages_for_litellm():
    msgs = [
        Message(role=Roles.system, content="s"),
        Message(role=Roles.user, content="u"),
    ]
    assert build_messages_for_litellm(msgs) == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]


def test_default_execute_uses_litellm_mock():
    task = minimal_task()
    ctx = ExecutionContext(
        task=task,
        model=Model.GPT_51_NANO_OR,
        run_config=RunConfig(temperature=0.1, max_tokens=42),
        messages=[{"role": "user", "content": "q"}],
    )
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "model-says"

    async def run():
        with patch(
            "benchbench.execution.litellm.acompletion", new_callable=AsyncMock
        ) as m:
            m.return_value = mock_resp
            out = await default_execute(ctx)
        m.assert_awaited_once()
        assert m.await_args is not None
        call_kw = m.await_args.kwargs
        assert call_kw["model"] == Model.GPT_51_NANO_OR.value
        assert call_kw["messages"] == ctx.messages
        assert call_kw["temperature"] == 0.1
        assert call_kw["max_tokens"] == 42
        return out

    assert asyncio.run(run()) == "model-says"
