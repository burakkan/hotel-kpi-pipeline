"""Runs the full flow: read CSV -> clean -> KPIs -> anomalies -> SQLite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .load import SqliteLoader
from .transform import ReservationCleaner, add_kpis, flag_anomalies


class KpiPipeline:
    def __init__(
        self,
        source_csv: str | Path,
        db_path: str | Path,
        cleaner: ReservationCleaner | None = None,
        loader: SqliteLoader | None = None,
    ) -> None:
        self.source_csv = Path(source_csv)
        self.cleaner = cleaner or ReservationCleaner()
        self.loader = loader or SqliteLoader(db_path)

    def run(self) -> dict[str, int]:
        raw = pd.read_csv(self.source_csv)
        clean, rejected = self.cleaner.clean(raw)
        flagged = flag_anomalies(add_kpis(clean))
        loaded = self.loader.load(flagged)
        return {
            "rows_in": len(raw),
            "rows_rejected": rejected,
            "rows_loaded": loaded,
            "anomalies": int(flagged["is_anomaly"].sum()),
        }
