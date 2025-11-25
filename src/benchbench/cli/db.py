"""
bench db command: Database management subcommands.
"""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from benchbench.discovery import discover_tasks
from benchbench.storage import BenchmarkStorage

console = Console()


@click.group()
def db() -> None:
    """Database management commands."""
    pass


@db.command()
@click.option(
    "--db",
    "db_path",
    default="benchmarks.duckdb",
    show_default=True,
    help="Path to the DuckDB database file.",
)
def info(db_path: str) -> None:
    """
    Show database statistics.

    Displays counts of tasks, runs, and storage info.
    """
    path = Path(db_path)
    if not path.exists():
        console.print(f"[yellow]Database not found:[/yellow] {db_path}")
        return

    with BenchmarkStorage(db_path) as storage:
        # Get counts
        task_count = storage.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        run_count = storage.conn.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0]
        error_count = storage.conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE error IS NOT NULL"
        ).fetchone()[0]

        # Get model breakdown
        models = storage.conn.execute(
            "SELECT model, COUNT(*) FROM task_runs GROUP BY model ORDER BY COUNT(*) DESC"
        ).fetchall()

        # File size
        size_bytes = path.stat().st_size
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"

    console.print(f"[bold]Database:[/bold] {db_path}")
    console.print(f"[bold]Size:[/bold] {size_str}")
    console.print()

    table = Table(title="Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Tasks", str(task_count))
    table.add_row("Total Runs", str(run_count))
    table.add_row("Errors", str(error_count))

    console.print(table)

    if models:
        console.print()
        model_table = Table(title="Runs by Model")
        model_table.add_column("Model", style="cyan")
        model_table.add_column("Runs", justify="right")

        for model, count in models:
            model_name = model.split("/")[-1]
            model_table.add_row(model_name, str(count))

        console.print(model_table)


@db.command()
@click.option(
    "--db",
    "db_path",
    default="benchmarks.duckdb",
    show_default=True,
    help="Path to the DuckDB database file.",
)
@click.argument("output", type=click.Path())
def export(db_path: str, output: str) -> None:
    """
    Export results to a CSV file.

    OUTPUT is the path to write the CSV file.
    """
    path = Path(db_path)
    if not path.exists():
        console.print(f"[red]Error:[/red] Database not found: {db_path}")
        raise SystemExit(1)

    with BenchmarkStorage(db_path) as storage:
        # Export with task display names
        storage.conn.execute(f"""
            COPY (
                SELECT 
                    t.display_name AS task,
                    t.id_chain,
                    tr.model,
                    tr.validation_passed,
                    tr.validation_score,
                    tr.validation_reason,
                    tr.duration_ms,
                    tr.error,
                    tr.created_at
                FROM task_runs tr
                LEFT JOIN tasks t ON tr.task_id = t.task_id
                ORDER BY t.display_name, tr.model
            ) TO '{output}' (HEADER, DELIMITER ',')
        """)

    console.print(f"[green]Exported results to:[/green] {output}")


@db.command()
@click.option(
    "--db",
    "db_path",
    default="benchmarks.duckdb",
    show_default=True,
    help="Path to the DuckDB database file.",
)
@click.option(
    "--errors-only",
    is_flag=True,
    help="Only remove runs with errors.",
)
@click.option(
    "--yes", "-y",
    is_flag=True,
    help="Skip confirmation prompt.",
)
def clean(db_path: str, errors_only: bool, yes: bool) -> None:
    """
    Clean up the database.

    By default, removes all task runs. Use --errors-only to only remove failed runs.
    """
    path = Path(db_path)
    if not path.exists():
        console.print(f"[yellow]Database not found:[/yellow] {db_path}")
        return

    with BenchmarkStorage(db_path) as storage:
        if errors_only:
            count = storage.conn.execute(
                "SELECT COUNT(*) FROM task_runs WHERE error IS NOT NULL"
            ).fetchone()[0]
            action = "error runs"
            query = "DELETE FROM task_runs WHERE error IS NOT NULL"
        else:
            count = storage.conn.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0]
            action = "all runs"
            query = "DELETE FROM task_runs"

        if count == 0:
            console.print(f"[yellow]No {action} to remove.[/yellow]")
            return

        if not yes:
            if not click.confirm(f"Remove {count} {action}?"):
                console.print("[dim]Cancelled.[/dim]")
                return

        storage.conn.execute(query)
        console.print(f"[green]Removed {count} {action}.[/green]")


@db.command()
@click.option(
    "--db",
    "db_path",
    default="benchmarks.duckdb",
    show_default=True,
    help="Path to the DuckDB database file.",
)
@click.option(
    "--tasks-dir",
    default="tasks",
    show_default=True,
    help="Path to the tasks directory.",
)
@click.option(
    "--yes", "-y",
    is_flag=True,
    help="Skip confirmation prompt.",
)
def prune(db_path: str, tasks_dir: str, yes: bool) -> None:
    """
    Remove runs for tasks that no longer exist.

    Discovers tasks from the tasks directory and removes any runs
    in the database whose task_id doesn't match a current task.
    """
    db_file = Path(db_path)
    if not db_file.exists():
        console.print(f"[yellow]Database not found:[/yellow] {db_path}")
        return

    tasks_path = Path(tasks_dir)
    if not tasks_path.exists():
        console.print(f"[red]Error:[/red] Tasks directory not found: {tasks_path}")
        raise SystemExit(1)

    # Discover current tasks
    console.print(f"Discovering tasks from [cyan]{tasks_path}[/cyan]...")
    current_tasks = discover_tasks(tasks_path)
    current_task_ids = {task.task_id for task in current_tasks}

    console.print(f"Found [green]{len(current_tasks)}[/green] tasks in filesystem")

    with BenchmarkStorage(db_path) as storage:
        # Find orphaned runs (task_id not in current tasks)
        all_db_task_ids = storage.conn.execute(
            "SELECT DISTINCT task_id FROM task_runs"
        ).fetchall()
        all_db_task_ids = {row[0] for row in all_db_task_ids}

        orphaned_task_ids = all_db_task_ids - current_task_ids

        if not orphaned_task_ids:
            console.print("[green]No orphaned runs found.[/green]")
            return

        # Count runs to be removed
        orphaned_list = list(orphaned_task_ids)
        count = storage.conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id IN (SELECT UNNEST(?::VARCHAR[]))",
            [orphaned_list]
        ).fetchone()[0]

        # Show what will be removed
        console.print(f"Found [yellow]{count}[/yellow] runs for {len(orphaned_task_ids)} orphaned task(s)")

        # Try to get display names for orphaned tasks
        orphaned_info = storage.conn.execute("""
            SELECT t.display_name, t.task_id, COUNT(tr.execution_id) as run_count
            FROM tasks t
            LEFT JOIN task_runs tr ON t.task_id = tr.task_id
            WHERE t.task_id IN (SELECT UNNEST(?::VARCHAR[]))
            GROUP BY t.display_name, t.task_id
        """, [orphaned_list]).fetchall()

        if orphaned_info:
            console.print("\nOrphaned tasks:")
            for display_name, task_id, run_count in orphaned_info:
                name = display_name or task_id[:16]
                console.print(f"  [dim]-[/dim] {name} ({run_count} runs)")

        if not yes:
            if not click.confirm(f"\nRemove {count} orphaned runs?"):
                console.print("[dim]Cancelled.[/dim]")
                return

        # Delete orphaned runs
        storage.conn.execute(
            "DELETE FROM task_runs WHERE task_id IN (SELECT UNNEST(?::VARCHAR[]))",
            [orphaned_list]
        )

        # Also clean up orphaned task entries
        storage.conn.execute(
            "DELETE FROM tasks WHERE task_id IN (SELECT UNNEST(?::VARCHAR[]))",
            [orphaned_list]
        )

        console.print(f"[green]Removed {count} runs and {len(orphaned_task_ids)} task entries.[/green]")
