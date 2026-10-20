"""Generates a synthetic daily room dataset so the pipeline can run without real hotel data."""

from __future__ import annotations

import numpy as np
import pandas as pd

# room_type: (rooms available per day, base rate in TRY)
ROOM_TYPES = {
    "Standard": (40, 1800.0),
    "Deluxe": (20, 2600.0),
    "Suite": (6, 4200.0),
}


def generate_daily_rooms(
    start: str = "2025-01-01",
    days: int = 365,
    seed: int = 42,
    anomaly_rate: float = 0.01,
) -> pd.DataFrame:
    """Return one row per date and room type.

    A small share of rows gets a corrupted revenue value (zero or a 3x spike),
    so the anomaly detection step has something real to find.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for day in pd.date_range(start, periods=days, freq="D"):
        season = 1 + 0.3 * np.sin(2 * np.pi * day.dayofyear / 365)
        weekend_boost = 1.15 if day.dayofweek >= 4 else 1.0

        for room_type, (available, base_rate) in ROOM_TYPES.items():
            demand = np.clip(rng.normal(0.72 * season * weekend_boost, 0.1), 0.05, 1.0)
            sold = int(round(available * demand))
            rate = base_rate * (1 + 0.2 * (season - 1)) * rng.normal(1.0, 0.05)
            revenue = sold * rate

            if rng.random() < anomaly_rate:
                revenue *= rng.choice([0.0, 3.0])

            rows.append(
                {
                    "date": day.strftime("%Y-%m-%d"),
                    "room_type": room_type,
                    "rooms_available": available,
                    "rooms_sold": sold,
                    "revenue_try": round(revenue, 2),
                }
            )
    return pd.DataFrame(rows)
