"""Loads the processed table into SQLite and runs analytical SQL against it."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd


class SqliteLoader:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def load(self, df: pd.DataFrame, table: str = "daily_kpis") -> int:
        """Replace the table with the given frame and return the number of rows written."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn:
            df.to_sql(table, conn, if_exists="replace", index=False)
        return len(df)

    def query(self, sql: str) -> pd.DataFrame:
        with closing(sqlite3.connect(self.db_path)) as conn:
            return pd.read_sql_query(sql, conn)
