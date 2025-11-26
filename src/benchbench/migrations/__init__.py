"""
Database migrations for benchbench.

Each migration is a module with:
- VERSION: str - the version this migration upgrades TO
- DESCRIPTION: str - human-readable description
- migrate(conn: duckdb.DuckDBPyConnection) -> None - the migration function

Migrations are applied in order based on version number.
"""

from dataclasses import dataclass
from typing import Callable
import duckdb
from packaging.version import Version


@dataclass
class Migration:
    """A database migration."""
    version: str
    description: str
    migrate: Callable[[duckdb.DuckDBPyConnection], None]


def get_migrations() -> list[Migration]:
    """
    Get all available migrations in order.
    
    IMPORTANT: Migrations must be listed in ascending version order
    (0.2.0, then 0.3.0, then 0.4.0, etc.) to ensure correct sequential application.
    """
    from benchbench.migrations import m_0_2_0
    
    return [
        Migration(
            version=m_0_2_0.VERSION,
            description=m_0_2_0.DESCRIPTION,
            migrate=m_0_2_0.migrate,
        ),
        # Add future migrations here in version order:
        # Migration(version=m_0_3_0.VERSION, ...),
    ]


def get_pending_migrations(current_version: str | None) -> list[Migration]:
    """
    Get migrations that need to be applied, in order.
    
    Args:
        current_version: Current schema version, or None for pre-versioning databases.
        
    Returns:
        List of migrations to apply, sorted by version (ascending).
    """
    all_migrations = get_migrations()  # Already sorted
    
    if current_version is None:
        # Pre-versioning database (0.1.0) - need all migrations
        return all_migrations
    
    current = Version(current_version)
    # Filter to only pending, order is preserved from get_migrations()
    return [m for m in all_migrations if Version(m.version) > current]


def record_migration(conn: duckdb.DuckDBPyConnection, version: str) -> None:
    """Record that a migration has been applied."""
    # Ensure schema_version table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version VARCHAR PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("INSERT INTO schema_version (version) VALUES (?)", [version])


def get_current_version(conn: duckdb.DuckDBPyConnection) -> str | None:
    """
    Get the current schema version from a database connection.
    
    Returns:
        Version string or None if pre-versioning database.
    """
    try:
        result = conn.execute(
            "SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1"
        ).fetchone()
        return result[0] if result else None
    except duckdb.CatalogException:
        return None
