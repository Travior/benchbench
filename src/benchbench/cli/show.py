"""
bench show command: Display benchmark results.
"""

import click
from rich.console import Console
from rich.table import Table

from benchbench.storage import BenchmarkStorage

console = Console()


@click.command()
@click.option(
    "--by-model/--by-task",
    "by_model",
    default=True,
    help="Group results by model (default) or by task.",
)
@click.option(
    "-f",
    "--filter",
    "filters",
    multiple=True,
    help="Filter by id_chain glob pattern (only applies to --by-task).",
)
@click.option(
    "--model",
    "model_filter",
    default=None,
    help="Filter results by model name.",
)
@click.option(
    "--db",
    "db_path",
    default="benchmarks.duckdb",
    show_default=True,
    help="Path to the DuckDB database file.",
)
def show(
    by_model: bool,
    filters: tuple[str, ...],
    model_filter: str | None,
    db_path: str,
) -> None:
    """
    Display benchmark results from the database.

    By default, shows aggregated results grouped by model.
    Use --by-task to group by task instead.

    Examples:

        bench show

        bench show --by-task

        bench show --model gpt-4o

        bench show --by-task --filter "adv_search::*"
    """
    try:
        storage = BenchmarkStorage(db_path)
    except Exception as e:
        console.print(f"[red]Error opening database:[/red] {e}")
        raise SystemExit(1)

    with storage:
        if by_model:
            _show_by_model(storage, model_filter)
        else:
            _show_by_task(storage, list(filters), model_filter)


def _show_by_model(storage: BenchmarkStorage, model_filter: str | None) -> None:
    """Display results grouped by model."""
    summary = storage.get_summary_by_model()

    if model_filter:
        model_filter_lower = model_filter.lower()
        summary = [
            s for s in summary
            if model_filter_lower in s["model"].lower()
        ]

    if not summary:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title="Results by Model")
    table.add_column("Model", style="cyan")
    table.add_column("Runs", justify="right")
    table.add_column("Passed", justify="right", style="green")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Avg Score", justify="right")
    table.add_column("Avg Duration", justify="right", style="dim")

    for row in summary:
        total = row["total_runs"]
        passed = row["passed"] or 0
        pass_rate = f"{100 * passed / total:.1f}%" if total > 0 else "-"
        avg_score = f"{row['avg_score']:.2f}" if row["avg_score"] is not None else "-"
        avg_duration = f"{row['avg_duration_ms']:.0f}ms" if row["avg_duration_ms"] else "-"

        # Extract short model name
        model_name = row["model"].split("/")[-1]

        table.add_row(
            model_name,
            str(total),
            str(passed),
            pass_rate,
            avg_score,
            avg_duration,
        )

    console.print(table)


def _show_by_task(
    storage: BenchmarkStorage,
    filters: list[str],
    model_filter: str | None,
) -> None:
    """Display results grouped by task."""
    # Get detailed task runs for filtering
    runs = storage.get_task_runs(model=model_filter)

    if not runs:
        console.print("[yellow]No results found.[/yellow]")
        return

    # If filters are provided, we need to filter by id_chain
    # This requires joining with the tasks table
    if filters:
        # Get task info to filter
        task_runs = storage.conn.execute("""
            SELECT 
                t.display_name,
                t.id_chain,
                tr.task_id,
                tr.model,
                tr.validation_passed,
                tr.validation_score,
                tr.duration_ms,
                tr.error
            FROM task_runs tr
            LEFT JOIN tasks t ON tr.task_id = t.task_id
            WHERE tr.error IS NULL
            ORDER BY t.display_name, tr.model
        """).fetchall()

        # Filter by id_chain patterns
        import fnmatch
        filtered_runs = []
        for row in task_runs:
            id_chain = row[1]  # id_chain array
            if id_chain:
                id_chain_str = "::".join(id_chain)
                if any(fnmatch.fnmatch(id_chain_str, p) for p in filters):
                    filtered_runs.append(row)

        if not filtered_runs:
            console.print("[yellow]No results match the specified filters.[/yellow]")
            return

        table = Table(title="Results by Task (filtered)")
        table.add_column("Task", style="cyan")
        table.add_column("Model", style="dim")
        table.add_column("Passed", justify="center")
        table.add_column("Score", justify="right")
        table.add_column("Duration", justify="right", style="dim")

        for row in filtered_runs:
            display_name = row[0] or row[2][:8]  # fallback to task_id
            model = row[3].split("/")[-1]
            passed = "[green]Yes[/green]" if row[4] else "[red]No[/red]"
            score = f"{row[5]:.2f}" if row[5] is not None else "-"
            duration = f"{row[6]:.0f}ms" if row[6] else "-"

            table.add_row(display_name, model, passed, score, duration)

        console.print(table)
    else:
        # Show aggregated by task
        summary = storage.get_summary_by_task()

        if not summary:
            console.print("[yellow]No results found.[/yellow]")
            return

        table = Table(title="Results by Task")
        table.add_column("Task", style="cyan")
        table.add_column("Runs", justify="right")
        table.add_column("Passed", justify="right", style="green")
        table.add_column("Pass Rate", justify="right")
        table.add_column("Avg Score", justify="right")

        for row in summary:
            total = row["total_runs"]
            passed = row["passed"] or 0
            pass_rate = f"{100 * passed / total:.1f}%" if total > 0 else "-"
            avg_score = f"{row['avg_score']:.2f}" if row["avg_score"] is not None else "-"
            display_name = row["display_name"] or row["task_id"][:8]

            table.add_row(
                display_name,
                str(total),
                str(passed),
                pass_rate,
                avg_score,
            )

        console.print(table)
