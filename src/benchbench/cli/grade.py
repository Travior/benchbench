"""
bench grade command: Interactively grade pending manual validations.
"""

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt

from benchbench.storage import BenchmarkStorage

console = Console()


def render_grading_ui(
    item: dict,
    current: int,
    total: int,
) -> None:
    """Render the grading interface for a single item."""
    console.clear()
    
    # Header
    header = Text()
    header.append("Manual Grading", style="bold cyan")
    header.append(f"  [{current}/{total}]", style="dim")
    console.print(Panel(header, style="cyan"))
    
    # Task info
    task_name = item.get("display_name") or "Unknown Task"
    model = item.get("model", "").split("/")[-1]  # Short model name
    
    console.print()
    console.print(f"[bold]Task:[/bold] {task_name}")
    console.print(f"[bold]Model:[/bold] {model}")
    console.print()
    
    # Rubric if present
    rubric = item.get("validation_rubric")
    if rubric:
        console.print(Panel(rubric, title="Grading Rubric", style="yellow"))
        console.print()
    
    # Model output
    output = item.get("output", "")
    # Truncate very long outputs for display
    max_lines = 30
    output_lines = output.split("\n")
    if len(output_lines) > max_lines:
        truncated = "\n".join(output_lines[:max_lines])
        truncated += f"\n\n[dim]... ({len(output_lines) - max_lines} more lines)[/dim]"
        output = truncated
    
    console.print(Panel(output, title="Model Output", style="green"))
    console.print()
    
    # Grading options
    console.print("[bold]Grade:[/bold]")
    console.print("  [green]p[/green] = Pass (1.0)    [yellow]h[/yellow] = Partial (0.5)    [red]f[/red] = Fail (0.0)")
    console.print("  [dim]s = Skip    v = View full output    q = Quit[/dim]")
    console.print()


def view_full_output(output: str) -> None:
    """Display full output with paging."""
    console.clear()
    console.print(Panel(output, title="Full Model Output", style="green"))
    console.print()
    Prompt.ask("[dim]Press Enter to continue[/dim]")


@click.command()
@click.option(
    "--db",
    "db_path",
    default="benchmarks.duckdb",
    show_default=True,
    help="Path to the DuckDB database file.",
)
@click.option(
    "--model",
    "model_filter",
    default=None,
    help="Only grade results for a specific model.",
)
def grade(db_path: str, model_filter: str | None) -> None:
    """
    Interactively grade pending manual validations.

    Use this command after running benchmarks that use `pending_manual()`
    in their validate.py files.

    Grading scale:
      - Pass (1.0): Fully correct response
      - Partial (0.5): Partially correct response  
      - Fail (0.0): Incorrect response

    Examples:

        bench grade

        bench grade --model gpt-4o
    """
    try:
        storage = BenchmarkStorage(db_path)
    except Exception as e:
        console.print(f"[red]Error opening database:[/red] {e}")
        raise SystemExit(1)

    with storage:
        pending = storage.get_pending_grades()
        
        if model_filter:
            model_filter_lower = model_filter.lower()
            pending = [
                p for p in pending
                if model_filter_lower in p.get("model", "").lower()
            ]
        
        if not pending:
            console.print("[green]No pending grades![/green]")
            return
        
        console.print(f"Found [yellow]{len(pending)}[/yellow] items pending manual grading.")
        console.print()
        
        graded = 0
        skipped = 0
        idx = 0
        
        while idx < len(pending):
            item = pending[idx]
            render_grading_ui(item, idx + 1, len(pending))
            
            choice = Prompt.ask("Enter grade").strip().lower()
            
            if choice == "q":
                break
            elif choice == "s":
                skipped += 1
                idx += 1
                continue
            elif choice == "v":
                view_full_output(item.get("output", ""))
                continue  # Re-render same item
            elif choice in ("p", "pass", "1"):
                score = 1.0
            elif choice in ("h", "half", "partial", "0.5"):
                score = 0.5
            elif choice in ("f", "fail", "0"):
                score = 0.0
            else:
                console.print("[red]Invalid choice. Use p/h/f/s/v/q[/red]")
                Prompt.ask("[dim]Press Enter to continue[/dim]")
                continue
            
            # Optional reason
            reason = Prompt.ask("Reason (optional)", default="").strip() or None
            
            # Save the grade
            storage.update_grade(
                execution_id=item["execution_id"],
                score=score,
                reason=reason,
            )
            
            graded += 1
            idx += 1
        
        # Summary
        console.clear()
        console.print()
        console.print("[bold]Grading Session Complete[/bold]")
        console.print(f"  Graded: [green]{graded}[/green]")
        console.print(f"  Skipped: [yellow]{skipped}[/yellow]")
        console.print(f"  Remaining: [dim]{len(pending) - graded - skipped}[/dim]")
        console.print()
        console.print("Use 'bench show' to view updated results.")
