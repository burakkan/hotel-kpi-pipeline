import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hotel_kpi.load import SqliteLoader  # noqa: E402
from hotel_kpi.pipeline import KpiPipeline  # noqa: E402
from hotel_kpi.synthetic import generate_daily_rooms  # noqa: E402
from hotel_kpi.transform import (  # noqa: E402
    DataQualityError,
    ReservationCleaner,
    add_kpis,
    flag_anomalies,
)


def _raw_rows(rows):
    return pd.DataFrame(
        rows,
        columns=["date", "room_type", "rooms_available", "rooms_sold", "revenue_try"],
    )


class CleanerTests(unittest.TestCase):
    def test_drops_invalid_and_duplicate_rows(self):
        raw = _raw_rows(
            [
                ("2025-01-01", "Standard", 40, 30, 54000.0),  # valid
                ("2025-01-01", "Standard", 40, 30, 54000.0),  # duplicate key
                ("2025-01-02", "Standard", 40, 50, 60000.0),  # sold > available
                ("2025-01-03", "Standard", 0, 0, 0.0),  # no capacity
                ("not-a-date", "Standard", 40, 10, 10000.0),  # bad date
                ("2025-01-04", "Deluxe", 20, 10, -5.0),  # negative revenue
            ]
        )
        clean, rejected = ReservationCleaner().clean(raw)
        self.assertEqual(len(clean), 1)
        self.assertEqual(rejected, 5)

    def test_missing_column_raises(self):
        with self.assertRaises(DataQualityError):
            ReservationCleaner().clean(pd.DataFrame({"date": ["2025-01-01"]}))


class KpiTests(unittest.TestCase):
    def test_occupancy_adr_revpar(self):
        df = _raw_rows([("2025-01-01", "Suite", 6, 3, 12600.0)])
        df["date"] = pd.to_datetime(df["date"])
        out = add_kpis(df).iloc[0]
        self.assertAlmostEqual(out["occupancy"], 0.5)
        self.assertAlmostEqual(out["adr_try"], 4200.0)
        self.assertAlmostEqual(out["revpar_try"], 2100.0)

    def test_adr_is_nan_when_nothing_sold(self):
        df = _raw_rows([("2025-01-01", "Suite", 6, 0, 0.0)])
        df["date"] = pd.to_datetime(df["date"])
        self.assertTrue(np.isnan(add_kpis(df).iloc[0]["adr_try"]))


class AnomalyTests(unittest.TestCase):
    def test_flags_a_clear_spike_and_drop(self):
        rng = np.random.default_rng(0)
        values = rng.normal(100, 2, 60)
        values[45] = 400.0  # spike
        values[50] = 0.0  # drop
        df = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=60),
                "room_type": "Standard",
                "revpar_try": values,
            }
        )
        flagged = flag_anomalies(df)
        self.assertTrue(flagged.loc[45, "is_anomaly"])
        self.assertTrue(flagged.loc[50, "is_anomaly"])
        self.assertLessEqual(flagged["is_anomaly"].sum(), 4)

    def test_first_days_are_not_flagged_without_history(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=5),
                "room_type": "Standard",
                "revpar_try": [1.0, 500.0, 1.0, 1.0, 1.0],
            }
        )
        self.assertFalse(flag_anomalies(df)["is_anomaly"].any())


class PipelineTests(unittest.TestCase):
    def test_end_to_end_loads_sqlite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = tmp / "raw.csv"
            db_path = tmp / "warehouse" / "hotel.db"
            generate_daily_rooms(days=60, seed=1).to_csv(csv_path, index=False)

            summary = KpiPipeline(csv_path, db_path).run()

            self.assertEqual(summary["rows_in"], 60 * 3)
            self.assertEqual(summary["rows_rejected"], 0)
            self.assertEqual(summary["rows_loaded"], 60 * 3)

            result = SqliteLoader(db_path).query(
                "SELECT COUNT(*) AS n FROM daily_kpis WHERE revpar_try IS NOT NULL"
            )
            self.assertEqual(int(result["n"][0]), 60 * 3)


if __name__ == "__main__":
    unittest.main()
