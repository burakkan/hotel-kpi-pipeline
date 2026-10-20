"""Command line entry point: `python -m hotel_kpi generate ...` or `python -m hotel_kpi run ...`."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import KpiPipeline
from .synthetic import generate_daily_rooms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hotel_kpi", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Write a synthetic daily room CSV")
    gen.add_argument("--out", type=Path, default=Path("data/raw/daily_rooms.csv"))
    gen.add_argument("--days", type=int, default=365)
    gen.add_argument("--seed", type=int, default=42)

    run = sub.add_parser("run", help="Run the pipeline on a CSV and load SQLite")
    run.add_argument("--source", type=Path, default=Path("data/raw/daily_rooms.csv"))
    run.add_argument("--db", type=Path, default=Path("data/warehouse/hotel.db"))

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "generate":
        df = generate_daily_rooms(days=args.days, seed=args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Wrote {len(df)} rows to {args.out}")
        return 0

    if args.command == "run":
        summary = KpiPipeline(args.source, args.db).run()
        for key, value in summary.items():
            print(f"{key}: {value}")
        return 0

    return 1
