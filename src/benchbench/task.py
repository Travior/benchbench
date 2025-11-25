"""
Domain models for benchmark tasks.
"""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import hashlib
import json

from benchbench.parser import Message
from benchbench.validation import ValidationResult, ValidateFn


def compute_messages_hash(messages: list[Message]) -> str:
    """Compute a stable hash of message content."""
    serialized = json.dumps(
        [{"role": msg.role.value, "content": msg.content} for msg in messages],
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def compute_task_id(id_chain: list[str], messages_hash: str) -> str:
    """
    Compute a unique task ID from id_chain and messages_hash.
    
    This ensures that changing the prompt content creates a new task,
    so old results don't mix with new ones.
    """
    combined = f"{'::'.join(id_chain)}::{messages_hash}"
    return hashlib.sha256(combined.encode()).hexdigest()


def compute_execution_id(task_id: str, model: str) -> str:
    """
    Compute unique execution ID from task_id + model.
    
    Since task_id now includes the messages_hash, we don't need
    to include it separately here.
    """
    combined = f"{task_id}::{model}"
    return hashlib.sha256(combined.encode()).hexdigest()


@dataclass
class Task:
    """A runnable benchmark task."""

    path: Path  # Filesystem path to task directory
    id_chain: list[str]  # e.g., ["adv_search", "next_match"]
    messages: list[Message]  # Prompt messages from parser
    validator: ValidateFn | None = None  # Loaded from validate.py if present

    @cached_property
    def messages_hash(self) -> str:
        """Hash of message content for change detection."""
        return compute_messages_hash(self.messages)

    @cached_property
    def task_id(self) -> str:
        """Unique identifier computed from id_chain and messages_hash."""
        return compute_task_id(self.id_chain, self.messages_hash)

    @property
    def display_name(self) -> str:
        """Human-readable name from id_chain."""
        return " / ".join(self.id_chain)

    def execution_id(self, model: str) -> str:
        """Compute execution ID for this task with a given model."""
        return compute_execution_id(self.task_id, model)


@dataclass
class TaskRun:
    """Result of running a single task against a model."""

    task_id: str  # Reference to Task.task_id
    model: str  # Model identifier string
    output: str  # Raw model response
    validation: ValidationResult | None = None
    duration_ms: float = 0.0
    error: str | None = None  # If execution failed
