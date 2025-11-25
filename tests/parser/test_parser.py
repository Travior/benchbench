from benchbench.parser import parse_md
from pathlib import Path
import pytest

def test_parse_file_valid():
    current_path = Path(__file__)
    file_path = current_path.parent / "testfile_valid.md"
    assert parse_md(file_path) is not None

def test_parse_file_invalid():
    current_path = Path(__file__)
    file_path = current_path.parent / "testfile_invalid.md"
    assert parse_md(file_path) is None

