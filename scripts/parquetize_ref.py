"""Run the parquetizer against ref/ data for all month/day pairs found in GTFS-RT.

Usage:
    uv run python scripts/parquetize_ref.py [--workers N] [--no-cache]

--workers N  Number of parallel worker processes (default: 4).
--no-cache   Re-run even if the output parquet already exists.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import polars as pl

from arpi import parquetizer
from arpi.discovery.files import FileClient
from arpi.discovery.gtfs import GtfsFetchService
from arpi.discovery.rt_parser import GtfsRtParser
from arpi.models.transit import FeedType, FetchRequest, TimePeriod, TransitAgency

REF = Path(__file__).parent.parent / "ref"
OUTPUT = REF / "output"
AGENCY = "RTL"


def iter_available_days() -> list[date]:
    base = REF / "GTFS-RT" / AGENCY
    feed_folders = ["trip_updates", "vehicle_positions"]

    days: set[date] = set()

    for feed_folder in feed_folders:
        feed_base = base / feed_folder
        if not feed_base.exists():
            continue

        for year_dir in feed_base.iterdir():
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue

            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir() or not month_dir.name.isdigit():
                    continue

                for day_dir in month_dir.iterdir():
                    if not day_dir.is_dir() or not day_dir.name.isdigit():
                        continue

                    try:
                        days.add(
                            date(
                                int(year_dir.name),
                                int(month_dir.name),
                                int(day_dir.name),
                            )
                        )
                    except ValueError:
                        continue

    return sorted(days)


def process_one(day: date, data_type: str, no_cache: bool) -> str:
    dd = f"{day.day:02d}"
    yyyy = f"{day.year:04d}"
    mm = f"{day.month:02d}"

    folder_name = "trip_updates" if data_type == "trip_update" else "vehicle_positions"
    file_stem = "trip-updates" if data_type == "trip_update" else "vehicle-positions"
    feed_type = (
        FeedType.TRIP_UPDATES
        if data_type == "trip_update"
        else FeedType.VEHICLE_POSITIONS
    )

    pb_folder = REF / "GTFS-RT" / AGENCY / folder_name / yyyy / mm / dd
    out_dir = OUTPUT / AGENCY / f"{yyyy}-{mm}"
    output_file = out_dir / f"{dd}.{file_stem}.parquet"

    out_dir.mkdir(parents=True, exist_ok=True)

    label = f"{yyyy}-{mm}-{dd} {data_type}"

    if output_file.exists() and not no_cache:
        df = pl.read_parquet(output_file)
        return f"[cached] {label}: {df.height:,} rows -> {output_file}"

    if output_file.exists():
        output_file.unlink()

    if not pb_folder.exists():
        return f"[missing] {label}: folder not found: {pb_folder}"

    transit_agency = TransitAgency(AGENCY)
    period = TimePeriod(
        start_time=datetime.combine(day, datetime.min.time()),
        end_time=datetime.combine(day, datetime.max.time()),
    )

    file_client = FileClient(local_path=str(REF))
    gtfs_fetcher = GtfsFetchService(file_client=file_client)
    rt_parser = GtfsRtParser()

    fr = FetchRequest(transit_agency, feed_type, period)

    parquetizer.run(
        data_type,
        pb_folder,
        output_file,
        fr=fr,
        gtfs_fetcher=gtfs_fetcher,
        parser=rt_parser,
    )

    df = pl.read_parquet(output_file)
    return f"[done]   {label}: {df.height:,} rows -> {output_file}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    days = iter_available_days()

    tasks = [
        (day, data_type, args.no_cache)
        for day in days
        for data_type in ("trip_update", "vehicle_position")
    ]

    print(f"Found {len(days)} day(s), {len(tasks)} parquetization task(s).")
    print(f"Running with {args.workers} worker process(es).")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_one, day, data_type, no_cache)
            for day, data_type, no_cache in tasks
        ]

        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()