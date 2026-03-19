"""
Example executor: same behavior as the built-in LiteLLM path.

Copy to a task folder as `execute.py`, or pass to `bench run --executor-path`.
"""

from benchbench.execution import ExecutionContext, default_execute


async def execute(ctx: ExecutionContext) -> str:
    return await default_execute(ctx)
