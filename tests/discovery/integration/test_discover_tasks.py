import asyncio
import logging

import pytest

from benchbench.discovery import DiscoveryError, discover_tasks
from benchbench.execution import ExecutionContext, RunConfig
from benchbench.models import Model
from tests.discovery.utils.task_trees import (
    leaf_task_markdown,
    parent_task_markdown,
    write_description,
    write_execute_module,
    write_minimal_validate,
)

pytestmark = pytest.mark.integration


def test_discover_empty_root(tmp_path):
    assert discover_tasks(tmp_path) == []


def test_discover_missing_description(tmp_path):
    (tmp_path / "cat").mkdir()
    with pytest.raises(DiscoveryError, match="Missing description.md"):
        discover_tasks(tmp_path)


def test_discover_leaf_extra_frontmatter(tmp_path):
    leaf = tmp_path / "meta_leaf"
    write_description(
        leaf,
        "---\nid: leaf_id\ncustom: 42\ntags: [a, b]\n---\n\n# User\nhi\n",
    )
    tasks = discover_tasks(tmp_path)
    assert len(tasks) == 1
    fm = tasks[0].frontmatter
    assert fm is not None
    assert fm.id == "leaf_id"
    dumped = fm.model_dump()
    assert dumped["custom"] == 42
    assert dumped["tags"] == ["a", "b"]


def test_discover_invalid_description(tmp_path):
    cat = tmp_path / "cat"
    cat.mkdir()
    # Valid YAML blocks but missing required `id` in frontmatter → parse_md returns None
    write_description(cat, "---\nfoo: bar\n---\n\n# System\nx\n")

    with pytest.raises(DiscoveryError, match="Failed to parse"):
        discover_tasks(tmp_path)


def test_discover_whitespace_only_id_rejected(tmp_path):
    leaf = tmp_path / "bad_id"
    write_description(leaf, "---\nid: '   '\n---\n\n# System\nx\n")
    with pytest.raises(DiscoveryError, match="Failed to parse"):
        discover_tasks(tmp_path)


def test_discover_id_trimmed_in_frontmatter(tmp_path):
    leaf = tmp_path / "trim"
    write_description(leaf, "---\nid: '  spaced  '\n---\n\n# System\nx\n")
    tasks = discover_tasks(tmp_path)
    assert len(tasks) == 1
    assert tasks[0].frontmatter is not None
    assert tasks[0].frontmatter.id == "spaced"
    assert tasks[0].id_chain == ["spaced"]


def test_discover_single_leaf(tmp_path):
    leaf = tmp_path / "my_leaf"
    write_description(leaf, leaf_task_markdown("leaf_id", "hello"))

    tasks = discover_tasks(tmp_path)
    assert len(tasks) == 1
    t = tasks[0]
    assert t.frontmatter is not None
    assert t.frontmatter.id == "leaf_id"
    assert t.frontmatter.model_dump() == {"id": "leaf_id"}
    assert t.id_chain == ["leaf_id"]
    assert t.path.resolve() == leaf.resolve()
    assert len(t.messages) == 1
    assert t.messages[0].content == "hello"
    assert t.executor is None
    assert t.validator is None


def test_discover_nested_parent_then_leaf(tmp_path):
    suite = tmp_path / "suite"
    write_description(suite, parent_task_markdown("parent_id"))
    inner = suite / "inner"
    write_description(inner, leaf_task_markdown("inner_id", "inner body"))

    tasks = discover_tasks(tmp_path)
    assert len(tasks) == 1
    t = tasks[0]
    assert t.id_chain == ["parent_id", "inner_id"]
    assert t.path.resolve() == inner.resolve()


def test_discover_leaf_with_execute(tmp_path):
    leaf = tmp_path / "leaf"
    write_description(leaf, leaf_task_markdown("e1"))
    write_execute_module(
        leaf,
        'async def execute(ctx):\n    return "from-disk"\n',
    )

    tasks = discover_tasks(tmp_path)
    assert len(tasks) == 1
    assert tasks[0].executor is not None
    out = asyncio.run(
        tasks[0].executor(
            ExecutionContext(
                task=tasks[0],
                model=Model.GPT_51_NANO_OR,
                run_config=RunConfig(),
                messages=[{"role": "user", "content": "hi"}],
            )
        )
    )
    assert out == "from-disk"


def test_discover_leaf_with_validate(tmp_path):
    leaf = tmp_path / "leaf"
    write_description(leaf, leaf_task_markdown("v1"))
    write_minimal_validate(leaf)

    tasks = discover_tasks(tmp_path)
    assert len(tasks) == 1
    assert tasks[0].validator is not None
    vr = asyncio.run(tasks[0].validator("any"))
    assert vr.passed is True
    assert vr.score == 1.0


def test_discover_parent_no_subdirs_returns_empty_branch(tmp_path, caplog):
    lonely = tmp_path / "lonely"
    write_description(lonely, parent_task_markdown("lonely_id"))

    caplog.set_level(logging.WARNING)
    tasks = discover_tasks(tmp_path)
    assert tasks == []
    assert any("no content and no subdirectories" in r.message for r in caplog.records)


def test_discover_skips_hidden_directories(tmp_path):
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    write_description(hidden, leaf_task_markdown("should_not_appear"))

    visible = tmp_path / "visible"
    write_description(visible, leaf_task_markdown("visible_id"))

    tasks = discover_tasks(tmp_path)
    assert len(tasks) == 1
    assert tasks[0].id_chain == ["visible_id"]


def test_discover_bad_execute_syntax_propagates(tmp_path):
    leaf = tmp_path / "bad_exec"
    write_description(leaf, leaf_task_markdown("bad"))
    # Unclosed paren → SyntaxError at import, not a runtime NameError
    write_execute_module(leaf, "def broken(\n")

    with pytest.raises(SyntaxError):
        discover_tasks(tmp_path)
