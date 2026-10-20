"""Cleaning, KPI calculation and anomaly detection for daily hotel room data."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["date", "room_type", "rooms_available", "rooms_sold", "revenue_try"]
NUMERIC_COLUMNS = ["rooms_available", "rooms_sold", "revenue_try"]


class DataQualityError(ValueError):
    """Raised when the input does not have the columns the pipeline needs."""


class ReservationCleaner:
    """Validates raw daily room records and drops the rows that cannot be trusted."""

    def __init__(self, required_columns: list[str] | None = None) -> None:
        self.required_columns = list(required_columns or REQUIRED_COLUMNS)

    def clean(self, raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Return the cleaned frame and the number of rows that were rejected."""
        missing = set(self.required_columns) - set(raw.columns)
        if missing:
            raise DataQualityError(f"Missing columns: {sorted(missing)}")

        df = raw[self.required_columns].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        rows_in = len(df)
        df = df.dropna()
        df = df.drop_duplicates(subset=["date", "room_type"])

        valid = (
            (df["rooms_available"] > 0)
            & (df["rooms_sold"] >= 0)
            & (df["rooms_sold"] <= df["rooms_available"])
            & (df["revenue_try"] >= 0)
        )
        df = df[valid].sort_values(["room_type", "date"]).reset_index(drop=True)
        return df, rows_in - len(df)


def add_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Add occupancy, ADR (average daily rate) and RevPAR columns."""
    out = df.copy()
    out["occupancy"] = out["rooms_sold"] / out["rooms_available"]
    out["adr_try"] = out["revenue_try"] / out["rooms_sold"].replace(0, np.nan)
    out["revpar_try"] = out["revenue_try"] / out["rooms_available"]
    return out


def _rolling_mad(values: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Median absolute deviation over a rolling window, scaled to match a std for normal data."""
    def mad(x: np.ndarray) -> float:
        return float(np.median(np.abs(x - np.median(x))))

    return values.rolling(window, min_periods=min_periods).apply(mad, raw=True) * 1.4826


def flag_anomalies(
    df: pd.DataFrame,
    value_col: str = "revpar_try",
    window: int = 28,
    min_periods: int = 7,
    threshold: float = 4.0,
) -> pd.DataFrame:
    """Flag days whose value is far from the trailing median of the same room type.

    A robust score (median and MAD) is used instead of mean and standard deviation,
    so one extreme day does not inflate the baseline and hide the next anomaly.
    The statistics are shifted by one day, so a day is never compared with a
    window that already contains it.
    """
    out = df.sort_values(["room_type", "date"]).copy()
    grouped = out.groupby("room_type")[value_col]

    trailing_median = grouped.transform(
        lambda s: s.rolling(window, min_periods=min_periods).median().shift(1)
    )
    trailing_mad = grouped.transform(
        lambda s: _rolling_mad(s, window, min_periods).shift(1)
    ).replace(0, np.nan)

    out["z_score"] = (out[value_col] - trailing_median) / trailing_mad
    out["is_anomaly"] = out["z_score"].abs() > threshold
    return out
