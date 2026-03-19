"""
Example executor: same behavior as the built-in LiteLLM path.

Copy to a task folder as `execute.py`, or pass to `bench run --executor-path`.
"""

from benchbench.execution import default_execute


async def execute(ctx):
    return await default_execute(ctx)
