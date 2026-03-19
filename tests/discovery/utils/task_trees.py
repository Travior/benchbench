"""Build example task directory trees for discovery tests."""

from pathlib import Path

DESCRIPTION_FILE = "description.md"


def leaf_task_markdown(task_id: str, system_content: str = "system prompt") -> str:
    """Minimal valid leaf: frontmatter + one system message."""
    return f"---\nid: {task_id}\n---\n\n# System\n{system_content}\n"


def parent_task_markdown(task_id: str) -> str:
    """Container task: valid frontmatter, no role sections → no messages → recurse."""
    return f"---\nid: {task_id}\n---\n\n"


def write_description(task_dir: Path, markdown: str) -> Path:
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / DESCRIPTION_FILE
    path.write_text(markdown)
    return path


def write_execute_module(task_dir: Path, body: str) -> Path:
    path = task_dir / "execute.py"
    path.write_text(body)
    return path


def write_minimal_validate(task_dir: Path) -> Path:
    path = task_dir / "validate.py"
    path.write_text(
        "from benchbench.validation import ValidationResult\n\n"
        "async def validate(output: str) -> ValidationResult:\n"
        "    return ValidationResult(passed=True, score=1.0)\n"
    )
    return path
