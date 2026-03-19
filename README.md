# benchbench

A simple scaffolding to run LLM tests and benchmarks. Define tasks in markdown, add optional executors and validators, and run them against multiple models with results stored in DuckDB.

## Installation

```bash
uv sync
```

## Quick Start

1. Create a `tasks/` directory with your benchmark tasks
2. Run benchmarks: `bench run gpt-5-nano`
3. View results: `bench show`

## Task Structure

Tasks are organized in a hierarchical directory structure. Each task folder contains a `description.md` file with YAML frontmatter and markdown content defining the conversation.

```
tasks/
├── category_a/
│   └── description.md      # Container (no messages)
│   ├── task_1/
│   │   └── description.md  # Leaf task with messages
│   │   └── execute.py      # Optional custom executor (generation)
│   │   └── validate.py     # Optional validator (scoring)
│   └── task_2/
│       └── description.md
└── category_b/
    └── description.md
```

### description.md Format

```markdown
---
id: task_identifier
---

# System
You are a helpful assistant.

# User
What is 2 + 2?
```

Supported roles: `System`, `User`, `Assistant`

### Executors (optional)

By default, each task is run with a single LiteLLM completion using the messages from `description.md` and the CLI model.

Add an `execute.py` in a **leaf** task folder to replace that step. The runner calls **`async def execute(ctx) -> str`** with an [`ExecutionContext`](src/benchbench/execution.py) (`ctx.task`, `ctx.model`, `ctx.run_config`, `ctx.messages`). After execution, any `validate.py` still runs on the returned string (executor first, then validator).

Minimal example (equivalent to the built-in behavior); see also [`examples/default_execute_executor.py`](examples/default_execute_executor.py).

```python
from benchbench.execution import default_execute

async def execute(ctx):
    return await default_execute(ctx)
```

**Deduplication:** `execution_id` is derived from `task_id` and model only (executor choice is not part of the key). `bench run` skips task+model pairs that already have a row; saving only happens for missing pairs. To re-run after changing `execute.py` or `--executor-path`, remove the existing run(s) first (e.g. `bench db clean` or delete specific rows).

To use one shared executor file for **all** tasks that do not define their own `execute.py`:

```bash
bench run gpt-5-nano --executor-path /path/to/my_execute.py
```

### Validators

Add a `validate.py` to any task folder to automatically validate model outputs:

```python
from benchbench.validation import ValidationResult

async def validate(output: str) -> ValidationResult:
    if "4" in output:
        return ValidationResult(passed=True, score=1.0)
    return ValidationResult(passed=False, score=0.0, reason="Expected '4' in response")
```

You can also use LLM-as-judge for more complex validation:

```python
from benchbench.validation import ValidationResult
from benchbench.models import get_async_client, Model

async def validate(output: str) -> ValidationResult:
    client = get_async_client()
    # Use structured output to evaluate the response
    result = await client.chat.completions.create(
        model=Model.GPT_51_NANO_OR,
        response_model=JudgeResponse,
        messages=[{"role": "user", "content": f"Evaluate: {output}"}]
    )
    return ValidationResult(passed=result.score > 0.5, score=result.score)
```

## CLI Commands

### `bench run`

Run benchmarks against one or more models.

```bash
# Run all tasks against a model
bench run gpt-5-nano

# Run against multiple models
bench run gpt-5-nano claude-3-opus

# Filter tasks by id_chain pattern
bench run gpt-5-nano --filter "category_a::*"
bench run gpt-5-nano -f "category_a::*" -f "*::specific_task"

# Customize execution
bench run gpt-5-nano --temperature 0.7 --concurrency 10

# Default executor override for tasks without task-local execute.py
bench run gpt-5-nano --executor-path ./my_default_execute.py
```

Options:
- `-f, --filter PATTERN`: Filter tasks by glob pattern (can be repeated)
- `--db PATH`: Database file path (default: `benchmarks.duckdb`)
- `--concurrency N`: Max parallel API requests (default: 5)
- `--temperature FLOAT`: Model temperature (default: 0.0)
- `--tasks-dir PATH`: Tasks directory (default: `tasks`)
- `--executor-path PATH`: Python file with `execute(ctx)` used when a task has no `execute.py`

### `bench show`

Display benchmark results.

```bash
# Show results grouped by model (default)
bench show

# Show results grouped by task
bench show --by-task

# Filter by model
bench show --model gpt-5-nano

# Filter by task pattern (with --by-task)
bench show --by-task --filter "category_a::*"
```

### `bench tasks`

List available benchmark tasks.

```bash
# List all tasks
bench tasks

# Show as tree structure
bench tasks --tree

# Filter tasks
bench tasks --filter "category_a::*"
```

### `bench db`

Database management commands.

```bash
# Show database statistics
bench db info

# Export results to CSV
bench db export results.csv

# Remove all runs
bench db clean

# Remove only error runs
bench db clean --errors-only

# Remove runs for deleted tasks
bench db prune
```

## Filtering Patterns

Tasks are identified by their `id_chain` (e.g., `category_a::task_1`). Filter patterns support glob syntax:

| Pattern | Matches |
|---------|---------|
| `category_a::*` | Direct children of `category_a` |
| `category_a::*::*` | Grandchildren of `category_a` |
| `*::task_1` | Any `task_1` in any parent |
| `category_a::task_1` | Exact match |
| `**` | All tasks |
| `category_a::**` | All descendants of `category_a` |

## Configuration

### Environment Variables

API keys are read from environment variables via litellm:
- `OPENAI_API_KEY` for OpenAI models
- `ANTHROPIC_API_KEY` for Anthropic models
- `OPENROUTER_API_KEY` for OpenRouter models

### Adding Models

Edit `src/benchbench/models.py` to add new models:

```python
class Model(StrEnum):
    GPT_51_NANO_OR = "openrouter/openai/gpt-5-nano"
    # Add more models here
    CLAUDE_3_OPUS = "anthropic/claude-3-opus"
```

## Architecture

- **Discovery**: Recursively walks the `tasks/` directory to find benchmark tasks
- **Parser**: Parses `description.md` files with YAML frontmatter and role-based markdown sections
- **Runner**: Executes tasks against models with configurable concurrency
- **Storage**: DuckDB-backed storage with deduplication by task content hash and model
- **Execution**: Default LiteLLM completion or optional `execute.py` / `--executor-path`
- **Validation**: Optional async validators loaded from `validate.py` files (run after execution)

## Development

```bash
# Run tests
uv run pytest

# Run a specific test
uv run pytest tests/parser/test_parser.py
```

## Dependencies

- `click` - CLI framework
- `litellm` - LLM API abstraction
- `instructor` - Structured outputs
- `duckdb` - Results storage
- `pydantic` - Data validation
- `rich` - Terminal output formatting
- `pyyaml` - YAML parsing
