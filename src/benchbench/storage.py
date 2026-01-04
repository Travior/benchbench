"""
DuckDB storage for benchmark results.

Provides persistent storage for task runs with deduplication based on
execution_id = hash(task_id + messages_hash + model).

Usage:
    from benchbench.storage import BenchmarkStorage
    
    storage = BenchmarkStorage("benchmarks.duckdb")
    
    # Find what needs to be run
    missing = storage.get_missing_executions(tasks, models)
    
    # After running, save results
    storage.save_task_run(task, task_run)
"""

import json
from pathlib import Path

import duckdb

from benchbench.task import Task, TaskRun

# Current schema version - bump this when adding migrations
SCHEMA_VERSION = "0.2.0"


class BenchmarkStorage:
    """DuckDB-backed storage for benchmark results."""

    def __init__(self, db_path: str | Path = "benchmarks.duckdb"):
        """
        Initialize storage with database path.
        
        Args:
            db_path: Path to DuckDB database file. Use ":memory:" for in-memory.
        """
        self.db_path = Path(db_path) if isinstance(db_path, str) else db_path
        self.conn = duckdb.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        # Schema version tracking
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version VARCHAR PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id VARCHAR PRIMARY KEY,
                id_chain VARCHAR[],
                display_name VARCHAR,
                path VARCHAR,
                messages_hash VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS task_runs (
                execution_id VARCHAR PRIMARY KEY,
                task_id VARCHAR,
                messages_hash VARCHAR,
                model VARCHAR,
                output VARCHAR,
                duration_ms DOUBLE,
                error VARCHAR,
                validation_passed BOOLEAN,
                validation_score DOUBLE,
                validation_reason VARCHAR,
                validation_metadata JSON,
                validation_pending BOOLEAN DEFAULT FALSE,
                validation_rubric VARCHAR,
                run_config JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # For new databases, mark as current version
        existing = self.conn.execute(
            "SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1"
        ).fetchone()
        if existing is None:
            # Check if this is a fresh DB (no task_runs) or existing pre-versioning DB
            has_runs = self.conn.execute("SELECT COUNT(*) FROM task_runs").fetchone()
            if has_runs and has_runs[0] == 0:
                # Fresh database - mark as current version
                self.conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    [SCHEMA_VERSION]
                )

    def get_schema_version(self) -> str | None:
        """
        Get the current schema version from the database.
        
        Returns:
            Version string (e.g., "0.2.0") or None if pre-versioning database.
        """
        try:
            result = self.conn.execute(
                "SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1"
            ).fetchone()
            return result[0] if result else None
        except duckdb.CatalogException:
            # schema_version table doesn't exist - pre-versioning database
            return None

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __enter__(self) -> "BenchmarkStorage":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def upsert_task(self, task: Task) -> None:
        """Insert or update a task definition."""
        self.conn.execute("""
            INSERT INTO tasks (task_id, id_chain, display_name, path, messages_hash, updated_at)
            VALUES (?, ?, ?, ?, ?, now())
            ON CONFLICT (task_id) DO UPDATE SET
                id_chain = EXCLUDED.id_chain,
                display_name = EXCLUDED.display_name,
                path = EXCLUDED.path,
                messages_hash = EXCLUDED.messages_hash,
                updated_at = now()
        """, [
            task.task_id,
            task.id_chain,
            task.display_name,
            str(task.path),
            task.messages_hash,
        ])

    def get_existing_execution_ids(self, execution_ids: list[str]) -> set[str]:
        """
        Check which execution_ids already exist in the database.
        
        Args:
            execution_ids: List of execution IDs to check.
            
        Returns:
            Set of execution IDs that already exist.
        """
        if not execution_ids:
            return set()

        result = self.conn.execute("""
            SELECT execution_id 
            FROM task_runs 
            WHERE execution_id IN (SELECT UNNEST(?::VARCHAR[]))
        """, [execution_ids]).fetchall()
        
        return {row[0] for row in result}

    def get_missing_executions(
        self, tasks: list[Task], models: list[str]
    ) -> list[tuple[Task, str, str]]:
        """
        Find task+model combinations that haven't been executed yet.
        
        Args:
            tasks: List of tasks to check.
            models: List of model identifiers to check.
            
        Returns:
            List of (task, model, execution_id) tuples that need to be run.
        """
        # Build all potential executions
        potential: list[tuple[Task, str, str]] = []  # (task, model, exec_id)
        for task in tasks:
            for model in models:
                execution_id = task.execution_id(model)
                potential.append((task, model, execution_id))

        # Check which already exist
        all_exec_ids = [p[2] for p in potential]
        existing = self.get_existing_execution_ids(all_exec_ids)

        # Return missing ones
        return [
            (task, model, exec_id)
            for task, model, exec_id in potential
            if exec_id not in existing
        ]

    def save_task_run(
        self,
        task: Task,
        task_run: TaskRun,
        run_config: dict | None = None,
    ) -> str:
        """
        Save a task run result to the database.
        
        Args:
            task: The task that was run (needed for messages_hash).
            task_run: The result of running the task.
            run_config: Optional run configuration (temperature, max_tokens, etc.).
            
        Returns:
            The execution_id of the saved run.
        """
        execution_id = task.execution_id(task_run.model)

        # Extract validation fields
        validation_passed = None
        validation_score = None
        validation_reason = None
        validation_metadata = None
        validation_pending = False
        validation_rubric = None
        
        if task_run.validation is not None:
            validation_passed = task_run.validation.passed
            validation_score = task_run.validation.score
            validation_reason = task_run.validation.reason
            validation_metadata = task_run.validation.metadata
            validation_pending = task_run.validation.pending
            validation_rubric = task_run.validation.rubric

        self.conn.execute("""
            INSERT INTO task_runs (
                execution_id, task_id, messages_hash, model, output, duration_ms,
                error, validation_passed, validation_score, validation_reason,
                validation_metadata, validation_pending, validation_rubric, run_config
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (execution_id) DO UPDATE SET
                output = EXCLUDED.output,
                duration_ms = EXCLUDED.duration_ms,
                error = EXCLUDED.error,
                validation_passed = EXCLUDED.validation_passed,
                validation_score = EXCLUDED.validation_score,
                validation_reason = EXCLUDED.validation_reason,
                validation_metadata = EXCLUDED.validation_metadata,
                validation_pending = EXCLUDED.validation_pending,
                validation_rubric = EXCLUDED.validation_rubric,
                run_config = EXCLUDED.run_config
        """, [
            execution_id,
            task_run.task_id,
            task.messages_hash,
            task_run.model,
            task_run.output,
            task_run.duration_ms,
            task_run.error,
            validation_passed,
            validation_score,
            validation_reason,
            json.dumps(validation_metadata) if validation_metadata else None,
            validation_pending,
            validation_rubric,
            json.dumps(run_config) if run_config else None,
        ])

        return execution_id

    def get_task_runs(
        self,
        task_id: str | None = None,
        model: str | None = None,
    ) -> list[dict]:
        """
        Query task runs with optional filters.
        
        Args:
            task_id: Filter by task ID.
            model: Filter by model.
            
        Returns:
            List of task run records as dictionaries.
        """
        query = "SELECT * FROM task_runs WHERE 1=1"
        params: list = []

        if task_id is not None:
            query += " AND task_id = ?"
            params.append(task_id)

        if model is not None:
            query += " AND model = ?"
            params.append(model)

        query += " ORDER BY created_at DESC"

        result = self.conn.execute(query, params)
        columns = [desc[0] for desc in result.description]
        
        return [dict(zip(columns, row)) for row in result.fetchall()]

    def _get_matching_task_ids(self, id_chain_patterns: list[str]) -> list[str]:
        """Return task IDs whose id_chain matches any provided pattern."""
        from pathlib import Path

        from benchbench.cli.filtering import matches_filter

        task_rows = self.conn.execute("""
            SELECT task_id, id_chain
            FROM tasks
        """).fetchall()

        matching_task_ids: list[str] = []
        for task_id, id_chain in task_rows:
            if not id_chain:
                continue

            task = Task(path=Path(""), id_chain=id_chain, messages=[])
            if any(matches_filter(task, pattern) for pattern in id_chain_patterns):
                matching_task_ids.append(task_id)

        return matching_task_ids

    def get_summary_by_model(
        self,
        id_chain_patterns: list[str] | None = None,
        model_filter: str | None = None,
    ) -> list[dict]:
        """Get aggregated results grouped by model."""
        where_clauses = ["error IS NULL"]
        params: list = []

        if model_filter:
            where_clauses.append("LOWER(model) LIKE ?")
            params.append(f"%{model_filter.lower()}%")

        if id_chain_patterns:
            matching_task_ids = self._get_matching_task_ids(id_chain_patterns)
            if not matching_task_ids:
                return []
            where_clauses.append("task_id IN (SELECT UNNEST(?::VARCHAR[]))")
            params.append(matching_task_ids)

        where_sql = " AND ".join(where_clauses)

        result = self.conn.execute(f"""
            SELECT 
                model,
                COUNT(*) as total_runs,
                SUM(CASE WHEN validation_passed THEN 1 ELSE 0 END) as passed,
                AVG(validation_score) as avg_score,
                AVG(duration_ms) as avg_duration_ms
            FROM task_runs
            WHERE {where_sql}
            GROUP BY model
            ORDER BY avg_score DESC
        """, params)
        
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]

    def get_summary_by_task(
        self,
        id_chain_patterns: list[str] | None = None,
        model_filter: str | None = None,
    ) -> list[dict]:
        """Get aggregated results grouped by task."""
        where_clauses = ["tr.error IS NULL"]
        params: list = []

        if model_filter:
            where_clauses.append("LOWER(tr.model) LIKE ?")
            params.append(f"%{model_filter.lower()}%")

        if id_chain_patterns:
            matching_task_ids = self._get_matching_task_ids(id_chain_patterns)
            if not matching_task_ids:
                return []
            where_clauses.append("tr.task_id IN (SELECT UNNEST(?::VARCHAR[]))")
            params.append(matching_task_ids)

        where_sql = " AND ".join(where_clauses)

        result = self.conn.execute(f"""
            SELECT 
                t.display_name,
                tr.task_id,
                COUNT(*) as total_runs,
                SUM(CASE WHEN tr.validation_passed THEN 1 ELSE 0 END) as passed,
                AVG(tr.validation_score) as avg_score
            FROM task_runs tr
            LEFT JOIN tasks t ON tr.task_id = t.task_id
            WHERE {where_sql}
            GROUP BY t.display_name, tr.task_id
            ORDER BY avg_score DESC
        """, params)

        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]

    def get_task_runs_for_display(
        self,
        id_chain_patterns: list[str] | None = None,
        model_filter: str | None = None,
    ) -> list[dict]:
        """Get task runs joined with task metadata for table display."""
        where_clauses = ["tr.error IS NULL"]
        params: list = []

        if model_filter:
            where_clauses.append("LOWER(tr.model) LIKE ?")
            params.append(f"%{model_filter.lower()}%")

        if id_chain_patterns:
            matching_task_ids = self._get_matching_task_ids(id_chain_patterns)
            if not matching_task_ids:
                return []
            where_clauses.append("tr.task_id IN (SELECT UNNEST(?::VARCHAR[]))")
            params.append(matching_task_ids)

        where_sql = " AND ".join(where_clauses)

        result = self.conn.execute(f"""
            SELECT
                t.display_name,
                t.id_chain,
                tr.task_id,
                tr.model,
                tr.validation_passed,
                tr.validation_score,
                tr.duration_ms
            FROM task_runs tr
            LEFT JOIN tasks t ON tr.task_id = t.task_id
            WHERE {where_sql}
            ORDER BY t.display_name, tr.model
        """, params)

        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]

    def get_pending_grades(self) -> list[dict]:
        """
        Get all task runs awaiting manual grading.
        
        Returns:
            List of task run records with pending=True, including task metadata.
        """
        result = self.conn.execute("""
            SELECT 
                tr.execution_id,
                tr.task_id,
                tr.model,
                tr.output,
                tr.validation_rubric,
                tr.created_at,
                t.display_name,
                t.id_chain,
                t.path
            FROM task_runs tr
            LEFT JOIN tasks t ON tr.task_id = t.task_id
            WHERE tr.validation_pending = TRUE
              AND tr.error IS NULL
            ORDER BY tr.created_at ASC
        """)
        
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]

    def get_pending_count(self) -> int:
        """Get the count of task runs awaiting manual grading."""
        result = self.conn.execute("""
            SELECT COUNT(*) 
            FROM task_runs 
            WHERE validation_pending = TRUE 
              AND error IS NULL
        """).fetchone()
        return result[0] if result else 0

    def update_grade(
        self,
        execution_id: str,
        score: float,
        reason: str | None = None,
    ) -> None:
        """
        Update a pending task run with a manual grade.
        
        Args:
            execution_id: The execution ID to update.
            score: The grade (0.0, 0.5, or 1.0).
            reason: Optional reason for the grade.
        """
        if score not in (0.0, 0.5, 1.0):
            raise ValueError(f"Score must be 0.0, 0.5, or 1.0, got {score}")
        
        passed = score == 1.0
        
        self.conn.execute("""
            UPDATE task_runs
            SET validation_passed = ?,
                validation_score = ?,
                validation_reason = ?,
                validation_pending = FALSE
            WHERE execution_id = ?
        """, [passed, score, reason, execution_id])
