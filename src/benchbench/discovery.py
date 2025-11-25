"""
Task discovery: recursively walk the tasks directory and collect runnable tasks.
"""

from pathlib import Path
import logging

from benchbench.parser import parse_md
from benchbench.task import Task
from benchbench.validation import load_validator

logger = logging.getLogger(__name__)

DESCRIPTION_FILE = "description.md"


class DiscoveryError(Exception):
    """Raised when task discovery fails."""

    pass


def discover_tasks(root: Path) -> list[Task]:
    """
    Recursively traverse from root, building tasks.

    The root directory is treated as a container and does not require description.md.
    All immediate subdirectories of root must have description.md.

    Algorithm:
    1. Read description.md, parse frontmatter
    2. If content has messages -> this is a leaf task, return it
    3. Else -> recurse into subdirectories
    4. Accumulate id_chain as we descend

    Raises DiscoveryError if description.md is missing or invalid (fail fast).
    """
    if not root.is_dir():
        raise DiscoveryError(f"Root path is not a directory: {root}")

    # Root is a container - scan its subdirectories
    tasks: list[Task] = []
    subdirs = sorted([d for d in root.iterdir() if d.is_dir()])

    for subdir in subdirs:
        if subdir.name.startswith("."):
            continue
        tasks.extend(_discover_recursive(subdir, id_chain=[]))

    return tasks


def _discover_recursive(current_dir: Path, id_chain: list[str]) -> list[Task]:
    """
    Internal recursive discovery function.

    Args:
        current_dir: Current directory being processed
        id_chain: Accumulated IDs from parent directories
    """
    description_path = current_dir / DESCRIPTION_FILE

    if not description_path.exists():
        raise DiscoveryError(f"Missing {DESCRIPTION_FILE} in {current_dir}")

    md_config = parse_md(description_path)
    if md_config is None:
        raise DiscoveryError(f"Failed to parse {description_path}")

    current_id = md_config.frontmatter.id
    new_id_chain = id_chain + [current_id]

    # Check if this is a leaf task (has message content)
    if md_config.content is not None and len(md_config.content.messages) > 0:
        logger.debug(f"Found leaf task at {current_dir}: {new_id_chain}")
        validator = load_validator(current_dir)
        return [
            Task(
                path=current_dir,
                id_chain=new_id_chain,
                messages=md_config.content.messages,
                validator=validator,
            )
        ]

    # Not a leaf - recurse into subdirectories
    tasks: list[Task] = []
    subdirs = sorted([d for d in current_dir.iterdir() if d.is_dir()])

    if not subdirs:
        logger.warning(
            f"Directory {current_dir} has no content and no subdirectories - not a valid task"
        )
        return []

    for subdir in subdirs:
        # Skip hidden directories
        if subdir.name.startswith("."):
            continue
        tasks.extend(_discover_recursive(subdir, new_id_chain))

    return tasks
