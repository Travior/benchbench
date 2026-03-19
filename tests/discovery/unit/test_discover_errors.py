import pytest

from benchbench.discovery import DiscoveryError, discover_tasks

pytestmark = pytest.mark.unit


def test_discover_tasks_root_not_directory(tmp_path):
    file_path = tmp_path / "not_a_dir"
    file_path.write_text("x")

    with pytest.raises(DiscoveryError, match="not a directory"):
        discover_tasks(file_path)
