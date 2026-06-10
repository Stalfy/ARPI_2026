"""Compare analyzer outputs using ref/ data.

Currently runs only the parquet analyzer for all dates found under GTFS-RT.

ref/ matches the pipeline's expected layout:
    GTFS-RT/RTL/...
    GTFS/RTL/...

Usage:
    uv run python scripts/compare_ref.py [--no-cache] [--verbose]

--no-cache   Re-run parquet analyzer even if output already exists.
--verbose    Print sample rows that differ when comparisons are enabled.

Intermediate parquets are saved under ref/output/.
"""

import argparse
import sys
import traceback
from datetime import date
from pathlib import Path

import polars as pl

from apex_transit_arpi.discovery.files import FileClient
from apex_transit_arpi.discovery.gtfs import GtfsFetchService

REF = Path(__file__).parent.parent / "ref"
OUTPUT = REF / "output"
AGENCY = "RTL"

_JOIN_KEY = ["trip_id", "stop_id", "pred_time"]
_ERROR_SEC_TOLERANCE = 1.0


def iter_available_days() -> list[date]:
    base = REF / "GTFS-RT" / AGENCY
    days: set[date] = set()

    for feed_folder in ("trip_updates", "vehicle_positions"):
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


def _paths(day: date) -> dict[str, Path]:
    yyyy, mm, dd = f"{day.year:04d}", f"{day.month:02d}", f"{day.day:02d}"

    month_dir = OUTPUT / AGENCY / f"{yyyy}-{mm}"
    month_dir.mkdir(parents=True, exist_ok=True)

    return {
        "analysis": month_dir / f"{dd}.analysis.parquet",
        "legacy": month_dir / f"{dd}.analysis-legacy.parquet",
        "duckdb": month_dir / f"{dd}.analysis-duckdb.parquet",
    }


def run_duckdb(day: date, p: dict[str, Path], force: bool) -> None:
    from apex_transit_arpi.analyzer import BenchmarkOrchestrator, DailyAnalysisStatus
    from apex_transit_arpi.analyzer.gtfs.duckdb import GtfsRtFetchService
    from apex_transit_arpi.models.transit import TransitAgency

    if force and p["duckdb"].exists():
        p["duckdb"].unlink()

    if p["duckdb"].exists():
        df = pl.read_parquet(p["duckdb"])
        print(f"  analysis cached  ({df.height:,} rows)")
        return

    transit_agency = TransitAgency(AGENCY)
    file_client = FileClient(local_path=str(REF))
    gtfs_fetcher = GtfsFetchService(file_client=file_client)
    gtfs_rt_fetcher = GtfsRtFetchService(
        file_client=file_client,
        gtfs_fetcher=gtfs_fetcher,
    )

    orchestrator = BenchmarkOrchestrator(
        file_client=file_client,
        gtfs_fetcher=gtfs_fetcher,
        gtfs_rt_fetcher=gtfs_rt_fetcher,
        output_dir=OUTPUT / AGENCY,
    )

    print("  running duckdb (reference)...")

    df = pl.DataFrame()

    try:
        status, result = orchestrator.benchmark_day(transit_agency, day)

        if status == DailyAnalysisStatus.SUCCESS:
            df = result
            print(f"  -> {df.height:,} benchmark rows")
        else:
            print(f"  -> status: {status.name} (writing empty file)")

    except Exception:
        print(f"  ERROR:\n{traceback.format_exc()}")

    df.write_parquet(p["duckdb"], compression="zstd")


def run_zero(day: date, p: dict[str, Path], force: bool) -> None:
    from apex_transit_arpi.analyzer import BenchmarkOrchestrator, DailyAnalysisStatus
    from apex_transit_arpi.analyzer.gtfs.legacy import GtfsRtFetchService
    from apex_transit_arpi.models.transit import TransitAgency

    if force and p["legacy"].exists():
        p["legacy"].unlink()

    if p["legacy"].exists():
        df = pl.read_parquet(p["legacy"])
        print(f"  analysis cached  ({df.height:,} rows)")
        return

    transit_agency = TransitAgency(AGENCY)
    file_client = FileClient(local_path=str(REF))
    gtfs_fetcher = GtfsFetchService(file_client=file_client)
    gtfs_rt_fetcher = GtfsRtFetchService(
        file_client=file_client,
        gtfs_fetcher=gtfs_fetcher,
    )

    orchestrator = BenchmarkOrchestrator(
        file_client=file_client,
        gtfs_fetcher=gtfs_fetcher,
        gtfs_rt_fetcher=gtfs_rt_fetcher,
        output_dir=OUTPUT / AGENCY,
    )

    print("  running zero (sequential Polars dedup, ground truth)...")

    df = pl.DataFrame()

    try:
        status, result = orchestrator.benchmark_day(transit_agency, day)

        if status == DailyAnalysisStatus.SUCCESS:
            df = result
            print(f"  -> {df.height:,} benchmark rows")
        else:
            print(f"  -> status: {status.name} (writing empty file)")

    except Exception:
        print(f"  ERROR:\n{traceback.format_exc()}")

    df.write_parquet(p["legacy"], compression="zstd")


def run_parquet(day: date, p: dict[str, Path], force: bool) -> None:
    from apex_transit_arpi.analyzer import BenchmarkOrchestrator, DailyAnalysisStatus
    from apex_transit_arpi.analyzer.gtfs.parquetized import GtfsRtFetchService
    from apex_transit_arpi.models.transit import TransitAgency

    if force and p["analysis"].exists():
        p["analysis"].unlink()

    if p["analysis"].exists():
        df = pl.read_parquet(p["analysis"])
        print(f"  analysis cached  ({df.height:,} rows)")
        return

    transit_agency = TransitAgency(AGENCY)
    file_client = FileClient(local_path=str(REF))
    gtfs_fetcher = GtfsFetchService(file_client=file_client)
    gtfs_rt_fetcher = GtfsRtFetchService(
        file_client=file_client,
        gtfs_fetcher=gtfs_fetcher,
        parquet_dir=OUTPUT / AGENCY,
    )

    orchestrator = BenchmarkOrchestrator(
        file_client=file_client,
        gtfs_fetcher=gtfs_fetcher,
        gtfs_rt_fetcher=gtfs_rt_fetcher,
        output_dir=OUTPUT / AGENCY,
    )

    print("  running parquet (vectorised Polars, full-fidelity TU/VP parquets)...")

    df = pl.DataFrame()

    try:
        status, result = orchestrator.benchmark_day(transit_agency, day)

        if status == DailyAnalysisStatus.SUCCESS:
            df = result
            print(f"  -> {df.height:,} benchmark rows")
        else:
            print(f"  -> status: {status.name} (writing empty file)")

    except Exception:
        print(f"  ERROR:\n{traceback.format_exc()}")

    df.write_parquet(p["analysis"], compression="zstd")


def compare(
    path_a: Path,
    label_a: str,
    path_b: Path,
    label_b: str,
    verbose: bool,
) -> None:
    df_a = pl.read_parquet(path_a) if path_a.exists() else pl.DataFrame()
    df_b = pl.read_parquet(path_b) if path_b.exists() else pl.DataFrame()

    print(f"  {label_a} rows : {df_a.height:,}")
    print(f"  {label_b} rows : {df_b.height:,}")

    if df_a.is_empty() or df_b.is_empty():
        print("  (one or both empty - skipping join)")
        return

    joined = df_a.join(df_b, on=_JOIN_KEY, how="full", suffix="_b")

    only_a = joined.filter(pl.col("error_sec_b").is_null()).height
    only_b = joined.filter(pl.col("error_sec").is_null()).height

    matched = joined.filter(pl.col("error_sec").is_not_null() & pl.col("error_sec_b").is_not_null())

    print(f"  only in {label_a}  : {only_a:,}")
    print(f"  only in {label_b}  : {only_b:,}")
    print(f"  matched rows       : {matched.height:,}")

    if matched.is_empty():
        return

    diff = (matched["error_sec"] - matched["error_sec_b"]).abs()

    print(
        "  error_sec diff     "
        f"max={diff.max():.3f}s  "
        f"mean={diff.mean():.3f}s  "
        f"within_1s={int((diff <= _ERROR_SEC_TOLERANCE).sum()):,}"
        f"/{matched.height:,}"
    )

    acc_mismatch = matched.filter(pl.col("is_accurate") != pl.col("is_accurate_b"))

    print(f"  is_accurate mismatches : {acc_mismatch.height:,}")

    if acc_mismatch.height:
        print(f"    {label_a}={int(matched['is_accurate'].sum()):,}  " f"{label_b}={int(matched['is_accurate_b'].sum()):,}  " "(on matched rows)")

    if verbose:
        if only_a:
            rows = joined.filter(pl.col("error_sec_b").is_null()).head(5)
            print(f"\n  Only in {label_a} (first 5 of {only_a}):")
            print(rows.select(["trip_id", "stop_id", "pred_time", "error_sec", "time_bucket"]))

        if only_b:
            rows = joined.filter(pl.col("error_sec").is_null()).head(5)
            print(f"\n  Only in {label_b} (first 5 of {only_b}):")
            print(
                rows.select(
                    [
                        "trip_id",
                        "stop_id",
                        "pred_time",
                        "error_sec_b",
                        "time_bucket_b",
                    ]
                )
            )

        if acc_mismatch.height:
            print(f"\n  is_accurate mismatches " f"(first 5 of {acc_mismatch.height}):")
            print(
                acc_mismatch.head(5).select(
                    [
                        "trip_id",
                        "stop_id",
                        "pred_time",
                        "time_bucket",
                        "error_sec",
                        "is_accurate",
                        "error_sec_b",
                        "is_accurate_b",
                    ]
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run analyzer variants on ref/ data")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Re-run enabled analyzers even if outputs exist",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print sample differing rows when comparisons are enabled",
    )
    args = parser.parse_args()

    if not REF.exists():
        print(f"ERROR: ref/ not found at {REF}")
        sys.exit(1)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    days = iter_available_days()

    if not days:
        print(f"ERROR: no dates found under {REF / 'GTFS-RT' / AGENCY}")
        sys.exit(1)

    print(f"Found {len(days)} day(s).\n")

    for day in days:
        print(f"=== {day} ({AGENCY}) ===")
        p = _paths(day)

        print("\n[analysis - parquet]")
        run_parquet(day, p, force=args.no_cache)

        # Keep these available for future comparison runs.
        #
        # print("\n[legacy - zero (ground truth)]")
        # run_zero(day, p, force=args.no_cache)
        #
        # print("\n[reference - duckdb]")
        # run_duckdb(day, p, force=args.no_cache)
        #
        # print("\n[comparison: parquet vs zero]")
        # compare(
        #     p["analysis"],
        #     "parquet",
        #     p["legacy"],
        #     "zero",
        #     verbose=args.verbose,
        # )
        #
        # print("\n[comparison: zero vs duckdb]")
        # compare(
        #     p["legacy"],
        #     "zero",
        #     p["duckdb"],
        #     "duckdb",
        #     verbose=args.verbose,
        # )

        if p["analysis"].exists():
            df = pl.read_parquet(p["analysis"])
            print(f"\n[output]")
            print(f"  file: {p['analysis']}")
            print(f"  rows: {df.height:,}")

        print()


if __name__ == "__main__":
    main()
