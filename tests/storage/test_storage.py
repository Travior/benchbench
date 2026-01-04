"""Tests for the storage module."""

import pytest
from pathlib import Path

from benchbench.storage import BenchmarkStorage
from benchbench.task import (
    Task,
    TaskRun,
    compute_execution_id,
    compute_messages_hash,
    compute_task_id,
)
from benchbench.parser import Message, Roles
from benchbench.validation import ValidationResult


@pytest.fixture
def sample_messages() -> list[Message]:
    """Sample messages for testing."""
    return [
        Message(role=Roles.system, content="You are a helpful assistant."),
        Message(role=Roles.user, content="Hello, world!"),
    ]


@pytest.fixture
def sample_task(sample_messages: list[Message]) -> Task:
    """Sample task for testing."""
    return Task(
        path=Path("/fake/path"),
        id_chain=["test_category", "test_task"],
        messages=sample_messages,
    )


@pytest.fixture
def storage() -> BenchmarkStorage:
    """In-memory storage for testing."""
    return BenchmarkStorage(":memory:")


class TestComputeMessagesHash:
    def test_deterministic(self, sample_messages: list[Message]):
        """Same messages should produce same hash."""
        hash1 = compute_messages_hash(sample_messages)
        hash2 = compute_messages_hash(sample_messages)
        assert hash1 == hash2

    def test_different_content_different_hash(self):
        """Different message content should produce different hash."""
        messages1 = [Message(role=Roles.user, content="Hello")]
        messages2 = [Message(role=Roles.user, content="Goodbye")]
        
        hash1 = compute_messages_hash(messages1)
        hash2 = compute_messages_hash(messages2)
        assert hash1 != hash2

    def test_different_role_different_hash(self):
        """Different message roles should produce different hash."""
        messages1 = [Message(role=Roles.user, content="Hello")]
        messages2 = [Message(role=Roles.system, content="Hello")]
        
        hash1 = compute_messages_hash(messages1)
        hash2 = compute_messages_hash(messages2)
        assert hash1 != hash2

    def test_order_matters(self):
        """Message order should affect hash."""
        msg1 = Message(role=Roles.user, content="First")
        msg2 = Message(role=Roles.user, content="Second")
        
        hash1 = compute_messages_hash([msg1, msg2])
        hash2 = compute_messages_hash([msg2, msg1])
        assert hash1 != hash2


class TestTaskModel:
    def test_messages_hash_cached(self, sample_task: Task):
        """messages_hash should be cached."""
        hash1 = sample_task.messages_hash
        hash2 = sample_task.messages_hash
        assert hash1 == hash2
        assert hash1 is hash2  # Same object due to caching

    def test_execution_id_method(self, sample_task: Task):
        """Task.execution_id() should compute correctly."""
        exec_id = sample_task.execution_id("model1")
        expected = compute_execution_id(sample_task.task_id, "model1")
        assert exec_id == expected

    def test_execution_id_different_models(self, sample_task: Task):
        """Different models should produce different execution IDs."""
        exec_id1 = sample_task.execution_id("model1")
        exec_id2 = sample_task.execution_id("model2")
        assert exec_id1 != exec_id2


class TestComputeExecutionId:
    def test_deterministic(self):
        """Same inputs should produce same execution ID."""
        exec_id1 = compute_execution_id("task1", "model1")
        exec_id2 = compute_execution_id("task1", "model1")
        assert exec_id1 == exec_id2

    def test_different_task_different_id(self):
        """Different task_id should produce different execution ID."""
        exec_id1 = compute_execution_id("task1", "model1")
        exec_id2 = compute_execution_id("task2", "model1")
        assert exec_id1 != exec_id2

    def test_different_model_different_id(self):
        """Different model should produce different execution ID."""
        exec_id1 = compute_execution_id("task1", "model1")
        exec_id2 = compute_execution_id("task1", "model2")
        assert exec_id1 != exec_id2


class TestComputeTaskId:
    def test_deterministic(self):
        """Same inputs should produce same task ID."""
        task_id1 = compute_task_id(["cat", "task"], "hash1")
        task_id2 = compute_task_id(["cat", "task"], "hash1")
        assert task_id1 == task_id2

    def test_different_id_chain_different_id(self):
        """Different id_chain should produce different task ID."""
        task_id1 = compute_task_id(["cat1", "task"], "hash1")
        task_id2 = compute_task_id(["cat2", "task"], "hash1")
        assert task_id1 != task_id2

    def test_different_messages_hash_different_id(self):
        """Different messages_hash should produce different task ID."""
        task_id1 = compute_task_id(["cat", "task"], "hash1")
        task_id2 = compute_task_id(["cat", "task"], "hash2")
        assert task_id1 != task_id2


class TestBenchmarkStorage:
    def test_init_creates_tables(self, storage: BenchmarkStorage):
        """Storage should create tables on init."""
        tables = storage.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        
        assert "tasks" in table_names
        assert "task_runs" in table_names

    def test_context_manager(self):
        """Storage should work as context manager."""
        with BenchmarkStorage(":memory:") as storage:
            assert storage.conn is not None

    def test_upsert_task(self, storage: BenchmarkStorage, sample_task: Task):
        """Should insert and update tasks."""
        storage.upsert_task(sample_task)
        
        result = storage.conn.execute(
            "SELECT task_id, display_name FROM tasks"
        ).fetchone()
        
        assert result is not None
        assert result[0] == sample_task.task_id
        assert result[1] == sample_task.display_name

    def test_upsert_task_updates_existing(
        self, storage: BenchmarkStorage, sample_messages: list[Message]
    ):
        """Upserting same task_id should update, not duplicate."""
        task1 = Task(
            path=Path("/path1"),
            id_chain=["cat", "task"],
            messages=sample_messages,
        )
        task2 = Task(
            path=Path("/path2"),  # Different path
            id_chain=["cat", "task"],  # Same id_chain = same task_id
            messages=sample_messages,
        )
        
        storage.upsert_task(task1)
        storage.upsert_task(task2)
        
        count_result = storage.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()
        assert count_result is not None
        count = count_result[0]
        assert count == 1
        
        path_result = storage.conn.execute("SELECT path FROM tasks").fetchone()
        assert path_result is not None
        path = path_result[0]
        assert path == "/path2"  # Updated to second task's path


class TestGetMissingExecutions:
    def test_all_missing_when_empty(
        self, storage: BenchmarkStorage, sample_task: Task
    ):
        """All executions should be missing when DB is empty."""
        models = ["model1", "model2"]
        missing = storage.get_missing_executions([sample_task], models)
        
        assert len(missing) == 2
        tasks_in_missing = {m[0].task_id for m in missing}
        models_in_missing = {m[1] for m in missing}
        
        assert tasks_in_missing == {sample_task.task_id}
        assert models_in_missing == {"model1", "model2"}

    def test_excludes_existing(
        self, storage: BenchmarkStorage, sample_task: Task
    ):
        """Should exclude executions that already exist."""
        # Save a run for model1
        task_run = TaskRun(
            task_id=sample_task.task_id,
            model="model1",
            output="test output",
            duration_ms=100.0,
        )
        storage.save_task_run(sample_task, task_run)
        
        # Check missing - should only have model2
        models = ["model1", "model2"]
        missing = storage.get_missing_executions([sample_task], models)
        
        assert len(missing) == 1
        assert missing[0][1] == "model2"

    def test_task_content_change_creates_new_execution(
        self, storage: BenchmarkStorage
    ):
        """Changing task content should require new execution."""
        messages_v1 = [Message(role=Roles.user, content="Version 1")]
        messages_v2 = [Message(role=Roles.user, content="Version 2")]
        
        task_v1 = Task(
            path=Path("/path"),
            id_chain=["cat", "task"],
            messages=messages_v1,
        )
        task_v2 = Task(
            path=Path("/path"),
            id_chain=["cat", "task"],  # Same id_chain
            messages=messages_v2,  # Different content
        )
        
        # Save run for v1
        task_run = TaskRun(
            task_id=task_v1.task_id,
            model="model1",
            output="output",
            duration_ms=100.0,
        )
        storage.save_task_run(task_v1, task_run)
        
        # Check missing with v2 - should need re-run due to content change
        missing = storage.get_missing_executions([task_v2], ["model1"])
        
        assert len(missing) == 1  # v2 needs to be run


class TestSaveTaskRun:
    def test_save_basic_run(
        self, storage: BenchmarkStorage, sample_task: Task
    ):
        """Should save basic task run."""
        task_run = TaskRun(
            task_id=sample_task.task_id,
            model="test-model",
            output="Hello, world!",
            duration_ms=150.5,
        )
        
        exec_id = storage.save_task_run(sample_task, task_run)
        
        assert exec_id is not None
        
        result = storage.conn.execute(
            "SELECT model, output, duration_ms FROM task_runs WHERE execution_id = ?",
            [exec_id]
        ).fetchone()
        assert result is not None
        
        assert result[0] == "test-model"
        assert result[1] == "Hello, world!"
        assert result[2] == 150.5

    def test_save_run_with_validation(
        self, storage: BenchmarkStorage, sample_task: Task
    ):
        """Should save task run with validation results."""
        validation = ValidationResult(
            passed=True,
            score=0.95,
            reason="Good answer",
            metadata={"key": "value"},
        )
        task_run = TaskRun(
            task_id=sample_task.task_id,
            model="test-model",
            output="output",
            duration_ms=100.0,
            validation=validation,
        )
        
        exec_id = storage.save_task_run(sample_task, task_run)
        
        result = storage.conn.execute(
            """SELECT validation_passed, validation_score, validation_reason 
               FROM task_runs WHERE execution_id = ?""",
            [exec_id]
        ).fetchone()
        assert result is not None
        
        assert result[0] is True
        assert result[1] == 0.95
        assert result[2] == "Good answer"

    def test_save_run_with_error(
        self, storage: BenchmarkStorage, sample_task: Task
    ):
        """Should save task run with error."""
        task_run = TaskRun(
            task_id=sample_task.task_id,
            model="test-model",
            output="",
            duration_ms=50.0,
            error="API timeout",
        )
        
        exec_id = storage.save_task_run(sample_task, task_run)
        
        result = storage.conn.execute(
            "SELECT error FROM task_runs WHERE execution_id = ?",
            [exec_id]
        ).fetchone()
        assert result is not None
        
        assert result[0] == "API timeout"

    def test_save_with_run_config(
        self, storage: BenchmarkStorage, sample_task: Task
    ):
        """Should save run config as JSON."""
        task_run = TaskRun(
            task_id=sample_task.task_id,
            model="test-model",
            output="output",
            duration_ms=100.0,
        )
        run_config = {"temperature": 0.7, "max_tokens": 1000}
        
        exec_id = storage.save_task_run(sample_task, task_run, run_config=run_config)
        
        result = storage.conn.execute(
            "SELECT run_config FROM task_runs WHERE execution_id = ?",
            [exec_id]
        ).fetchone()
        assert result is not None
        
        assert result[0] is not None


class TestQueryMethods:
    def test_get_task_runs_all(
        self, storage: BenchmarkStorage, sample_task: Task
    ):
        """Should retrieve all task runs."""
        for i, model in enumerate(["model1", "model2"]):
            task_run = TaskRun(
                task_id=sample_task.task_id,
                model=model,
                output=f"output {i}",
                duration_ms=100.0,
            )
            storage.save_task_run(sample_task, task_run)
        
        runs = storage.get_task_runs()
        assert len(runs) == 2

    def test_get_task_runs_by_model(
        self, storage: BenchmarkStorage, sample_task: Task
    ):
        """Should filter task runs by model."""
        for model in ["model1", "model2"]:
            task_run = TaskRun(
                task_id=sample_task.task_id,
                model=model,
                output="output",
                duration_ms=100.0,
            )
            storage.save_task_run(sample_task, task_run)
        
        runs = storage.get_task_runs(model="model1")
        assert len(runs) == 1
        assert runs[0]["model"] == "model1"

    def test_get_summary_by_model(
        self, storage: BenchmarkStorage, sample_task: Task
    ):
        """Should aggregate results by model."""
        validation = ValidationResult(passed=True, score=0.9)
        task_run = TaskRun(
            task_id=sample_task.task_id,
            model="model1",
            output="output",
            duration_ms=100.0,
            validation=validation,
        )
        storage.save_task_run(sample_task, task_run)
        
        summary = storage.get_summary_by_model()
        assert len(summary) == 1
        assert summary[0]["model"] == "model1"
        assert summary[0]["total_runs"] == 1
        assert summary[0]["avg_score"] == 0.9

    def test_get_summary_by_task_filters_by_model_substring(
        self,
        storage: BenchmarkStorage,
        sample_task: Task,
    ):
        """Task summary model filter should support partial model names."""
        storage.upsert_task(sample_task)

        for model in [
            "openrouter/openai/gpt-5.2",
            "openrouter/anthropic/claude-opus-4.6",
        ]:
            task_run = TaskRun(
                task_id=sample_task.task_id,
                model=model,
                output="output",
                duration_ms=100.0,
            )
            storage.save_task_run(sample_task, task_run)

        summary = storage.get_summary_by_task(model_filter="gpt-5.2")

        assert len(summary) == 1
        assert summary[0]["task_id"] == sample_task.task_id
        assert summary[0]["total_runs"] == 1

    def test_get_task_runs_for_display_composes_model_and_task_filters(
        self,
        storage: BenchmarkStorage,
        sample_messages: list[Message],
    ):
        """Display rows should apply model and id_chain filters together."""
        task_alpha = Task(
            path=Path("/alpha"),
            id_chain=["alpha", "first"],
            messages=sample_messages,
        )
        task_beta = Task(
            path=Path("/beta"),
            id_chain=["beta", "second"],
            messages=[Message(role=Roles.user, content="different")],
        )

        storage.upsert_task(task_alpha)
        storage.upsert_task(task_beta)

        storage.save_task_run(
            task_alpha,
            TaskRun(
                task_id=task_alpha.task_id,
                model="openrouter/openai/gpt-5.2",
                output="output",
                duration_ms=100.0,
            ),
        )
        storage.save_task_run(
            task_beta,
            TaskRun(
                task_id=task_beta.task_id,
                model="openrouter/openai/gpt-5.2",
                output="output",
                duration_ms=100.0,
            ),
        )
        storage.save_task_run(
            task_alpha,
            TaskRun(
                task_id=task_alpha.task_id,
                model="openrouter/anthropic/claude-opus-4.6",
                output="output",
                duration_ms=100.0,
            ),
        )

        rows = storage.get_task_runs_for_display(
            id_chain_patterns=["alpha::*"],
            model_filter="gpt-5.2",
        )

        assert len(rows) == 1
        assert rows[0]["task_id"] == task_alpha.task_id
        assert rows[0]["model"] == "openrouter/openai/gpt-5.2"
