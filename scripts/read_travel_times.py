# uv run python scripts/read_travel_times.py travel_times.parquet

import argparse
from pathlib import Path

import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)

    parser.add_argument("--route", type=str)
    parser.add_argument("--trip", type=str)
    parser.add_argument("--shape", type=str)

    parser.add_argument("--min-time", type=float)
    parser.add_argument("--max-time", type=float)

    parser.add_argument("--head", type=int, default=100)
    parser.add_argument("--sort", type=str, default="TravelTime")

    parser.add_argument(
        "--summary-by",
        choices=["route", "shape", "trip"],
    )

    args = parser.parse_args()

    df = pl.read_parquet(args.parquet)

    if args.route:
        df = df.filter(pl.col("Route").cast(pl.String) == args.route)

    if args.trip:
        df = df.filter(pl.col("Trip").cast(pl.String) == args.trip)

    if args.shape:
        df = df.filter(pl.col("Shape").cast(pl.String) == args.shape)

    if args.min_time is not None:
        df = df.filter(pl.col("travel_time_min") >= args.min_time)

    if args.max_time is not None:
        df = df.filter(pl.col("travel_time_min") <= args.max_time)

    if args.sort in df.columns:
        df = df.sort(args.sort, descending=True)

    print(f"Rows: {df.height:,}")
    print(f"Columns: {len(df.columns)}")
    print()

    display_cols = [
        c
        for c in [
            "Route",
            "Trip",
            "Shape",
            "TravelTime",
            "travel_time_min",
            "first_latitude",
            "first_longitude",
            "last_latitude",
            "last_longitude",
            "vehicle_position_count",
            "service_date",
            "trip_headsign",
        ]
        if c in df.columns
    ]

    with pl.Config(
        tbl_rows=args.head,
        tbl_cols=-1,
        fmt_str_lengths=120,
        tbl_width_chars=400,
    ):
        print(df.select(display_cols))

    print("\nTravel Time Summary")
    print("=" * 80)

    if "TravelTime" in df.columns:
        print(
            df.select(
                [
                    pl.len().alias("trips"),
                    pl.col("TravelTime").min().alias("min_sec"),
                    pl.col("TravelTime").mean().round(2).alias("avg_sec"),
                    pl.col("TravelTime").median().alias("median_sec"),
                    pl.col("TravelTime").max().alias("max_sec"),
                    pl.col("travel_time_min").mean().round(2).alias("avg_min"),
                ]
            )
        )

    if args.summary_by:
        column = {
            "route": "Route",
            "shape": "Shape",
            "trip": "Trip",
        }[args.summary_by]

        print(f"\nSummary by {column}")
        print("=" * 80)

        summary = (
            df.group_by(column)
            .agg(
                [
                    pl.len().alias("count"),
                    pl.col("TravelTime").min().alias("min_sec"),
                    pl.col("TravelTime").mean().round(2).alias("avg_sec"),
                    pl.col("TravelTime").median().alias("median_sec"),
                    pl.col("TravelTime").max().alias("max_sec"),
                ]
            )
            .sort("avg_sec", descending=True)
        )

        with pl.Config(
            tbl_rows=args.head,
            tbl_cols=-1,
            fmt_str_lengths=120,
            tbl_width_chars=400,
        ):
            print(summary)

    if "Route" in df.columns:
        print("\nTop routes by trip count")
        print("=" * 80)

        print(df.group_by("Route").agg(pl.len().alias("trip_count")).sort("trip_count", descending=True).head(20))

    print("\nLongest trips")
    print("=" * 80)

    longest_cols = [
        c
        for c in [
            "Route",
            "Trip",
            "Shape",
            "travel_time_min",
            "first_latitude",
            "first_longitude",
            "last_latitude",
            "last_longitude",
            "trip_headsign",
        ]
        if c in df.columns
    ]

    print(df.sort("travel_time_min", descending=True).head(20).select(longest_cols))


if __name__ == "__main__":
    main()
