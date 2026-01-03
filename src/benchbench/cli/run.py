"""
bench run command: Execute benchmarks for specified models.
"""

from pathlib import Path

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from benchbench.cli.filtering import filter_tasks
from benchbench.discovery import discover_tasks
from benchbench.models import Model
from benchbench.runner import RunConfig, TaskRunner
from benchbench.storage import BenchmarkStorage

console = Console()

# Build a lookup from model name/alias to Model enum
MODEL_LOOKUP: dict[str, Model] = {}
for model in Model:
    # Add full name
    MODEL_LOOKUP[model.value.lower()] = model
    # Add short name (last part after /)
    short_name = model.value.split("/")[-1].lower()
    MODEL_LOOKUP[short_name] = model
    # Add enum name
    MODEL_LOOKUP[model.name.lower()] = model


def resolve_model(name: str) -> Model | None:
    """Resolve a model name/alias to a Model enum value."""
    name_lower = name.lower()

    # Exact match
    if name_lower in MODEL_LOOKUP:
        return MODEL_LOOKUP[name_lower]

    # Partial match (prefix)
    matches = [m for key, m in MODEL_LOOKUP.items() if key.startswith(name_lower)]
    if len(matches) == 1:
        return matches[0]

    return None


@click.command()
@click.argument("models", nargs=-1, required=True)
@click.option(
    "-f",
    "--filter",
    "filters",
    multiple=True,
    help="Filter tasks by id_chain glob pattern (can be used multiple times).",
)
@click.option(
    "--db",
    "db_path",
    default="benchmarks.duckdb",
    show_default=True,
    help="Path to the DuckDB database file.",
)
@click.option(
    "--concurrency",
    default=5,
    show_default=True,
    help="Maximum number of concurrent API requests.",
)
@click.option(
    "--temperature",
    default=0.0,
    show_default=True,
    help="Model temperature for generation.",
)
@click.option(
    "--tasks-dir",
    default="tasks",
    show_default=True,
    help="Path to the tasks directory.",
)
def run(
    models: tuple[str, ...],
    filters: tuple[str, ...],
    db_path: str,
    concurrency: int,
    temperature: float,
    tasks_dir: str,
) -> None:
    """
    Run benchmarks for the specified MODELS.

    MODELS can be full model names, short names, or partial matches.
    Use 'bench tasks' to see available tasks and 'bench run --help' for options.

    Examples:

        bench run gpt-4o claude-3-opus

        bench run gpt-5-nano --filter "adv_search::*"

        bench run gpt-4o -f "adv_search::*" -f "*::validate"
    """
    # Resolve model names
    resolved_models: list[Model] = []
    for model_name in models:
        model = resolve_model(model_name)
        if model is None:
            available = ", ".join(sorted(set(m.value.split("/")[-1] for m in Model)))
            console.print(f"[red]Error:[/red] Unknown model '{model_name}'")
            console.print(f"Available models: {available}")
            raise SystemExit(1)
        resolved_models.append(model)

    # Discover tasks
    tasks_path = Path(tasks_dir)
    if not tasks_path.exists():
        console.print(f"[red]Error:[/red] Tasks directory not found: {tasks_path}")
        raise SystemExit(1)

    console.print(f"Discovering tasks from [cyan]{tasks_path}[/cyan]...")
    all_tasks = discover_tasks(tasks_path)

    if not all_tasks:
        console.print("[yellow]No tasks found.[/yellow]")
        raise SystemExit(0)

    # Apply filters
    tasks = filter_tasks(all_tasks, list(filters))

    if not tasks:
        console.print("[yellow]No tasks match the specified filters.[/yellow]")
        raise SystemExit(0)

    console.print(f"Found [green]{len(tasks)}[/green] tasks")

    # Check what needs to be run
    with BenchmarkStorage(db_path) as storage:
        # Ensure tasks are registered
        for task in tasks:
            storage.upsert_task(task)

        missing = storage.get_missing_executions(
            tasks, [m.value for m in resolved_models]
        )

        if not missing:
            console.print("[green]All benchmarks already completed![/green]")
            console.print("Use 'bench show' to view results.")
            return

        console.print(
            f"Running [green]{len(missing)}[/green] benchmark(s) "
            f"({len(tasks)} tasks x {len(resolved_models)} models, "
            f"{len(tasks) * len(resolved_models) - len(missing)} cached)"
        )

        # Run benchmarks with progress display
        runner = TaskRunner(
            RunConfig(temperature=temperature, max_concurrency=concurrency)
        )

        # Group missing by task for cleaner execution
        tasks_to_run = []
        models_to_run = []
        for task, model_str, _ in missing:
            tasks_to_run.append(task)
            # Find the Model enum for this model string
            model_enum = next(m for m in resolved_models if m.value == model_str)
            models_to_run.append(model_enum)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Running benchmarks...", total=len(missing))

            # Run one at a time to update progress (runner handles concurrency internally)
            import asyncio

            async def run_with_progress():
                semaphore = asyncio.Semaphore(concurrency)

                async def run_single(task, model):
                    async with semaphore:
                        result = await runner.run_task(task, model)
                        storage.save_task_run(
                            task, result, {"temperature": temperature}
                        )
                        progress.advance(task_id)
                        return result

                coros = [
                    run_single(task, model)
                    for task, model in zip(tasks_to_run, models_to_run)
                ]
                return await asyncio.gather(*coros)

            results = asyncio.run(run_with_progress())

        # Summary
        passed = sum(1 for r in results if r.validation and r.validation.passed)
        failed = sum(
            1
            for r in results
            if r.validation and not r.validation.passed and not r.validation.pending
        )
        errors = sum(1 for r in results if r.error)
        no_validation = sum(1 for r in results if not r.validation and not r.error)
        pending = sum(1 for r in results if r.validation and r.validation.pending)

        console.print()
        console.print("[bold]Results:[/bold]")
        if passed:
            console.print(f"  [green]{passed} passed[/green]")
        if failed:
            console.print(f"  [red]{failed} failed[/red]")
        if errors:
            console.print(f"  [red]{errors} errors[/red]")
        if pending:
            console.print(f"  [yellow]{pending} pending manual grading[/yellow]")
        if no_validation:
            console.print(f"  [dim]{no_validation} no validation[/dim]")

        console.print()
        if pending:
            console.print("Use 'bench grade' to grade pending results.")
        console.print("Use 'bench show' to view detailed results.")
