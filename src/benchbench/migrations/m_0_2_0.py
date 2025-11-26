"""
Migration: 0.1.0 -> 0.2.0

Adds support for manual grading with pending validation status.
"""

import duckdb

VERSION = "0.2.0"
DESCRIPTION = "Add manual grading support (validation_pending, validation_rubric columns)"


def migrate(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Add validation_pending and validation_rubric columns to task_runs table.
    
    These columns support the two-phase manual grading workflow:
    - validation_pending: True when awaiting human review
    - validation_rubric: Instructions for the human grader
    """
    # Add validation_pending column
    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute("""
            ALTER TABLE task_runs 
            ADD COLUMN IF NOT EXISTS validation_pending BOOLEAN DEFAULT FALSE
        """)
    
        # Add validation_rubric column  
        conn.execute("""
            ALTER TABLE task_runs 
            ADD COLUMN IF NOT EXISTS validation_rubric VARCHAR
        """)
    except duckdb.Error:
        conn.execute("ROLLBACK")
