"""
bench tasks command: List available benchmark tasks.
"""

from pathlib import Path
from collections import defaultdict

import click
from rich.console import Console
from rich.tree import Tree
from rich.table import Table

from benchbench.discovery import discover_tasks
from benchbench.cli.filtering import filter_tasks

console = Console()


@click.command()
@click.option(
    "--tree",
    "as_tree",
    is_flag=True,
    help="Display tasks as a tree structure.",
)
@click.option(
    "-f",
    "--filter",
    "filters",
    multiple=True,
    help="Filter tasks by id_chain glob pattern (can be used multiple times).",
)
@click.option(
    "--tasks-dir",
    default="tasks",
    show_default=True,
    help="Path to the tasks directory.",
)
def tasks(
    as_tree: bool,
    filters: tuple[str, ...],
    tasks_dir: str,
) -> None:
    """
    List available benchmark tasks.

    By default, shows a flat list of all tasks.
    Use --tree to see the hierarchical structure.

    Examples:

        bench tasks

        bench tasks --tree

        bench tasks --filter "adv_search::*"
    """
    tasks_path = Path(tasks_dir)
    if not tasks_path.exists():
        console.print(f"[red]Error:[/red] Tasks directory not found: {tasks_path}")
        raise SystemExit(1)

    all_tasks = discover_tasks(tasks_path)

    if not all_tasks:
        console.print("[yellow]No tasks found.[/yellow]")
        return

    # Apply filters
    filtered = filter_tasks(all_tasks, list(filters))

    if not filtered:
        console.print("[yellow]No tasks match the specified filters.[/yellow]")
        return

    if as_tree:
        _show_tree(filtered)
    else:
        _show_list(filtered)


def _show_list(tasks) -> None:
    """Display tasks as a flat table."""
    table = Table(title=f"Available Tasks ({len(tasks)})")
    table.add_column("ID Chain", style="cyan")
    table.add_column("Has Validator", justify="center")
    table.add_column("Messages", justify="right")

    for task in sorted(tasks, key=lambda t: t.id_chain):
        id_chain_str = "::".join(task.id_chain)
        has_validator = "[green]Yes[/green]" if task.validator else "[dim]No[/dim]"
        msg_count = str(len(task.messages))

        table.add_row(id_chain_str, has_validator, msg_count)

    console.print(table)


def _show_tree(tasks) -> None:
    """Display tasks as a tree structure."""
    # Build tree structure from id_chains
    tree = Tree("[bold]Tasks[/bold]")

    # Group by first element of id_chain
    groups: dict[str, list] = defaultdict(list)
    for task in tasks:
        if task.id_chain:
            groups[task.id_chain[0]].append(task)

    for group_name in sorted(groups.keys()):
        group_tasks = groups[group_name]

        if len(group_tasks) == 1 and len(group_tasks[0].id_chain) == 1:
            # Single task at root level
            task = group_tasks[0]
            validator_mark = " [green](v)[/green]" if task.validator else ""
            tree.add(f"[cyan]{group_name}[/cyan]{validator_mark}")
        else:
            # Group with children
            branch = tree.add(f"[bold]{group_name}[/bold]")
            _add_children(branch, group_tasks, depth=1)

    console.print(tree)


def _add_children(parent, tasks, depth: int) -> None:
    """Recursively add children to tree."""
    # Group by element at current depth
    groups: dict[str, list] = defaultdict(list)
    for task in tasks:
        if len(task.id_chain) > depth:
            groups[task.id_chain[depth]].append(task)

    for name in sorted(groups.keys()):
        group_tasks = groups[name]

        # Check if any task ends at this depth
        ending_tasks = [t for t in group_tasks if len(t.id_chain) == depth + 1]
        continuing_tasks = [t for t in group_tasks if len(t.id_chain) > depth + 1]

        if ending_tasks and not continuing_tasks:
            # Leaf node(s)
            for task in ending_tasks:
                validator_mark = " [green](v)[/green]" if task.validator else ""
                parent.add(f"[cyan]{name}[/cyan]{validator_mark}")
        elif continuing_tasks:
            # Branch with more children
            branch = parent.add(f"[bold]{name}[/bold]")
            _add_children(branch, continuing_tasks, depth + 1)

            # Also add any tasks that end here
            for task in ending_tasks:
                validator_mark = " [green](v)[/green]" if task.validator else ""
                branch.add(f"[cyan](task)[/cyan]{validator_mark}")
