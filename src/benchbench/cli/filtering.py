"""
Glob-based filtering for task id_chains.

Supports patterns like:
    - "adv_search::*"        -> direct children of adv_search
    - "adv_search::*::*"     -> grandchildren
    - "*::next_match"        -> any next_match in any parent
    - "adv_search::next_match" -> exact match
    - "*"                    -> all top-level
    - "**"                   -> all tasks (recursive)
"""

import fnmatch

from benchbench.task import Task


def matches_filter(task: Task, pattern: str) -> bool:
    """
    Check if a task's id_chain matches the given glob pattern.

    The id_chain is joined with "::" for matching.
    Pattern uses fnmatch-style globs where:
        - * matches any single segment
        - ** is not directly supported by fnmatch, so we handle it specially

    Args:
        task: The task to check.
        pattern: Glob pattern to match against.

    Returns:
        True if the task matches the pattern.
    """
    id_chain_str = "::".join(task.id_chain)

    # Handle ** pattern (match everything)
    if pattern == "**":
        return True

    # Handle patterns with ** (match any number of segments)
    if "**" in pattern:
        # Convert ** to a regex-like match via fnmatch
        # Replace ** with a placeholder that matches multiple segments
        # fnmatch doesn't support **, so we need custom logic
        return _match_double_star(id_chain_str, pattern)

    # Standard fnmatch for single * patterns
    return fnmatch.fnmatch(id_chain_str, pattern)


def _match_double_star(id_chain_str: str, pattern: str) -> bool:
    """
    Handle patterns containing ** which matches zero or more segments.

    Examples:
        "adv_search::**" matches "adv_search::foo" and "adv_search::foo::bar"
        "**::validate" matches "foo::validate" and "foo::bar::validate"
    """
    parts = pattern.split("::")
    chain_parts = id_chain_str.split("::")

    return _match_parts(chain_parts, parts)


def _match_parts(chain: list[str], pattern: list[str]) -> bool:
    """Recursively match chain parts against pattern parts."""
    if not pattern:
        return not chain

    if not chain:
        # Only match if remaining pattern is all ** or empty
        return all(p == "**" for p in pattern)

    head = pattern[0]
    rest = pattern[1:]

    if head == "**":
        # ** can match zero or more segments
        # Try matching zero segments (skip the **)
        if _match_parts(chain, rest):
            return True
        # Try matching one segment and continue with **
        if _match_parts(chain[1:], pattern):
            return True
        return False
    else:
        # Regular segment - must match current chain part
        if fnmatch.fnmatch(chain[0], head):
            return _match_parts(chain[1:], rest)
        return False


def filter_tasks(tasks: list[Task], patterns: list[str]) -> list[Task]:
    """
    Filter tasks by one or more glob patterns.

    A task is included if it matches ANY of the patterns (OR logic).

    Args:
        tasks: List of tasks to filter.
        patterns: List of glob patterns. If empty, returns all tasks.

    Returns:
        Filtered list of tasks.
    """
    if not patterns:
        return tasks

    return [task for task in tasks if any(matches_filter(task, p) for p in patterns)]
