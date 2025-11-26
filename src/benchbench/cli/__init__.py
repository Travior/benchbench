"""
CLI for the benchbench benchmark suite.

Usage:
    bench run <models...> [--filter PATTERN]
    bench show [--by-task] [--filter PATTERN] [--model MODEL]
    bench grade [--model MODEL]
    bench tasks [--tree] [--filter PATTERN]
    bench db <subcommand>
"""

import click

from benchbench.cli.run import run
from benchbench.cli.show import show
from benchbench.cli.grade import grade
from benchbench.cli.tasks import tasks
from benchbench.cli.db import db


@click.group()
@click.version_option(version="0.2.0", prog_name="bench")
def main() -> None:
    """Benchmark suite for LLMs."""
    pass


main.add_command(run)
main.add_command(show)
main.add_command(grade)
main.add_command(tasks)
main.add_command(db)


if __name__ == "__main__":
    main()
