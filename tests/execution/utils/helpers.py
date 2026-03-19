from pathlib import Path

from benchbench.parser import Message, Roles
from benchbench.task import Task


def minimal_task(
    messages: list[Message] | None = None,
    executor=None,
    path: Path | None = None,
    id_chain: list[str] | None = None,
    validator=None,
) -> Task:
    return Task(
        path=path or Path("/tmp/benchbench-test"),
        id_chain=id_chain or ["test", "task"],
        messages=messages
        or [Message(role=Roles.user, content="hello")],
        executor=executor,
        validator=validator,
    )


def echo_execute_source(suffix: str = "|ok") -> str:
    return (
        f'async def execute(ctx):\n'
        f'    return ctx.messages[-1]["content"] + "{suffix}"\n'
    )


def raising_execute_source(msg: str = "boom") -> str:
    return (
        f"async def execute(ctx):\n"
        f'    raise RuntimeError("{msg}")\n'
    )
