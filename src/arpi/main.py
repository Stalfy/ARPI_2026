"""Entry point: process GTFS-RT days one at a time, agency by agency."""

import argparse
import os
import sys
import traceback
import zipfile
from datetime import date, datetime
from pathlib import Path
from time import sleep

import polars as pl

from arpi.discovery import iter_day_dirs
from arpi.discovery.files import FileClient
from arpi.discovery.gtfs import GtfsFetchService
from arpi.discovery.rt_parser import GtfsRtParser
from arpi.discovery.gtfs_mapping import run as map_gtfs
from arpi.models.transit import FeedType, FetchRequest, TimePeriod

_DEFAULT_GTFS_ROOT = "GTFS"
_JOIN_KEY = ["trip_id", "stop_id", "pred_time"]
_ERROR_SEC_TOLERANCE = 1.0  # seconds


# ── Helpers ────────────────────────────────────────────────────────────────────


def _cleanup_empty(path: Path) -> None:
    """Remove *path* if it exists but is zero bytes (partial/failed write)."""
    if path.exists() and path.stat().st_size == 0:
        path.unlink()


def _zip_gtfs(gtfs_agency_dir: Path, output_file: Path) -> None:
    """Archive all files under *gtfs_agency_dir* into a deflate-compressed zip."""
    with zipfile.ZipFile(output_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(gtfs_agency_dir.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(gtfs_agency_dir))
    print(f"  -> {output_file}")


def _diff_day(label: str, modern_path: Path, legacy_path: Path) -> list[str]:
    """Compare two analysis parquet files; return human-readable mismatch descriptions."""
    modern = pl.read_parquet(modern_path)
    legacy = pl.read_parquet(legacy_path)
    issues: list[str] = []

    if modern.height != legacy.height:
        issues.append(f"row count: modern={modern.height}, legacy={legacy.height}")

    if modern.is_empty() or legacy.is_empty():
        return [f"  {label}: {issue}" for issue in issues]

    joined = modern.join(legacy, on=_JOIN_KEY, how="full", suffix="_leg")

    unmatched = joined.filter(pl.col("error_sec").is_null() | pl.col("error_sec_leg").is_null()).height
    if unmatched:
        issues.append(f"{unmatched} rows unmatched on {_JOIN_KEY}")

    matched = joined.filter(pl.col("error_sec").is_not_null() & pl.col("error_sec_leg").is_not_null())
    if matched.height:
        max_diff = matched.select((pl.col("error_sec") - pl.col("error_sec_leg")).abs().max()).item()
        if max_diff is not None and max_diff > _ERROR_SEC_TOLERANCE:
            issues.append(f"max |error_sec diff| = {max_diff:.2f}s > {_ERROR_SEC_TOLERANCE}s tolerance")
        accuracy_mismatch = matched.filter(pl.col("is_accurate") != pl.col("is_accurate_leg")).height
        if accuracy_mismatch:
            issues.append(f"{accuracy_mismatch} rows differ in is_accurate")

    return [f"  {label}: {issue}" for issue in issues]


# ── Per-day steps ──────────────────────────────────────────────────────────────


def _step_gtfs(base_dir: Path, output_dir: Path, agency: str) -> None:
    """Zip static GTFS and build mappings for *agency*. Both steps are idempotent."""
    gtfs_agency_dir = base_dir / _DEFAULT_GTFS_ROOT / agency
    if not gtfs_agency_dir.is_dir():
        return

    agency_output_dir = output_dir / agency
    agency_output_dir.mkdir(parents=True, exist_ok=True)

    zip_file = agency_output_dir / "GTFS.zip"
    if not zip_file.exists():
        print("  [gtfs] zipping...")
        _zip_gtfs(gtfs_agency_dir, zip_file)

    mappings_file = agency_output_dir / "GTFS.mappings.json.zip"
    if not mappings_file.exists():
        print("  [gtfs] building mappings...")
        try:
            map_gtfs(zip_file, mappings_file)
        except Exception:
            print(f"  ERROR (gtfs mappings):\n{traceback.format_exc()}")
        finally:
            _cleanup_empty(mappings_file)


def _step_parquetize(
    label: str,
    data_type: str,
    pb_dir: Path | None,
    output_file: Path,
    fr: FetchRequest,
    gtfs_fetcher: GtfsFetchService,
    rt_parser: GtfsRtParser,
    force: bool = False,
) -> bool:
    """Parquetize one feed type for one day. Returns True if output is ready."""
    from arpi import parquetizer

    if not force and output_file.exists():
        print(f"  [{label}] cached")
        return True

    if pb_dir is None or not pb_dir.exists():
        print(f"  [{label}] pb dir not found — skipping")
        return False

    if output_file.exists():
        output_file.unlink()

    try:
        parquetizer.run(data_type, pb_dir, output_file, fr=fr, gtfs_fetcher=gtfs_fetcher, parser=rt_parser)
        return output_file.exists()
    except Exception:
        print(f"  ERROR [{label}]:\n{traceback.format_exc()}")
        _cleanup_empty(output_file)
        return False


def _step_analyze(agency: str, day: date, orchestrator, output_file: Path, force: bool = False) -> bool:
    """Run the parquet-based analyzer for one day. Returns True if output exists when done."""
    from arpi.analyzer import DailyAnalysisStatus

    if not force and output_file.exists():
        return True
    df = pl.DataFrame()
    try:
        status, result = orchestrator.benchmark_day(agency, day)
        if status == DailyAnalysisStatus.SUCCESS:
            df = result
        else:
            print(f"  parquet: {status.name}")
    except Exception:
        print(f"  ERROR (parquet):\n{traceback.format_exc()}")
    df.write_parquet(output_file, compression="zstd")
    label = "(empty)" if df.is_empty() else f"({df.height} rows)"
    print(f"  -> {output_file} {label}")
    return True


def _step_legacy(orchestrator, agency, day: date, legacy_out: Path, force: bool = False) -> bool:
    """Run the zero-based legacy analyzer for one day. Returns True if output exists when done."""
    from arpi.analyzer import DailyAnalysisStatus

    if not force and legacy_out.exists():
        return True
    df = pl.DataFrame()
    try:
        status, result = orchestrator.benchmark_day(agency, day)
        if status == DailyAnalysisStatus.SUCCESS:
            df = result
        else:
            print(f"  legacy: {status.name}")
    except Exception:
        print(f"  ERROR (legacy):\n{traceback.format_exc()}")
    df.write_parquet(legacy_out, compression="zstd")
    label = "(empty)" if df.is_empty() else f"({df.height} rows)"
    print(f"  -> {legacy_out} {label}")
    return True


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    """ARPI GTFS-RT pipeline — one full pass per day, per agency.

    Environment variables:
        ARPI_BASE_DIR    — raw input root  (default: /app/data/input)
        ARPI_OUTPUT_DIR  — processed output root  (default: /app/data/output)
        ARPI_MAX_DAYS    — max days to process per agency; 0 = unlimited.
                           Already-processed days count toward the limit.
                           The current day always completes before stopping.
    """
    parser = argparse.ArgumentParser(description="ARPI GTFS-RT pipeline")
    parser.add_argument(
        "--compare",
        action="store_true",
        help=("Also run the legacy protobuf-based analyzer and write " "{dd}.analysis-legacy.parquet alongside {dd}.analysis.parquet."),
    )
    parser.add_argument(
        "--force-analysis",
        action="store_true",
        help="Re-run analyzers even if output files already exist (debug).",
    )
    parser.add_argument(
        "--agency",
        metavar="FOLDER",
        help="Process only this agency folder name (debug).",
    )
    args = parser.parse_args()

    base_dir = Path(os.getenv("ARPI_BASE_DIR", "/app/data/input"))
    output_dir = Path(os.getenv("ARPI_OUTPUT_DIR", "/app/data/output"))
    max_days = int(os.getenv("ARPI_MAX_DAYS", "0"))

    # Collect {agency: {(yyyy, mm, dd): {feed_name: day_dir}}} from the raw input tree.
    agency_days: dict[str, dict[tuple[str, str, str], dict[str, Path]]] = {}
    for agency, feed_name, day_dir in iter_day_dirs(base_dir=base_dir):
        key = (day_dir.parent.parent.name, day_dir.parent.name, day_dir.name)
        agency_days.setdefault(agency, {}).setdefault(key, {})[feed_name] = day_dir

    from arpi.analyzer import BenchmarkOrchestrator
    from arpi.analyzer.gtfs.parquetized import GtfsRtFetchService as ParquetFetchService
    from arpi.models.transit import TransitAgency

    file_client = FileClient(local_path=str(base_dir))

    if args.compare:
        from arpi.analyzer.gtfs.legacy import GtfsRtFetchService as ZeroFetchService

    if args.agency:
        if args.agency not in agency_days:
            print(f"Agency '{args.agency}' not found in input. Available: {sorted(agency_days)}")
            sys.exit(1)
        agency_days = {args.agency: agency_days[args.agency]}

    failures: list[str] = []

    for agency in sorted(agency_days):
        days_processed = 0

        parquet_orchestrator = None
        zero_orchestrator = None
        transit_agency = None
        try:
            transit_agency = TransitAgency(agency)
            gtfs_fetcher = GtfsFetchService(file_client=file_client)
            rt_parser = GtfsRtParser()
            parquet_fetcher = ParquetFetchService(
                file_client=file_client,
                gtfs_fetcher=gtfs_fetcher,
                parquet_dir=output_dir / agency,
            )
            parquet_orchestrator = BenchmarkOrchestrator(
                file_client=file_client,
                gtfs_fetcher=gtfs_fetcher,
                gtfs_rt_fetcher=parquet_fetcher,
                output_dir=output_dir / agency,
            )
            if args.compare:
                zero_fetcher = ZeroFetchService(file_client=file_client, gtfs_fetcher=gtfs_fetcher)
                zero_orchestrator = BenchmarkOrchestrator(
                    file_client=file_client,
                    gtfs_fetcher=gtfs_fetcher,
                    gtfs_rt_fetcher=zero_fetcher,
                    output_dir=output_dir / agency,
                )
        except ValueError:
            print(f"[{agency}] not a known TransitAgency — skipping")
            continue

        for (yyyy, mm, dd), feeds in sorted(agency_days[agency].items(), reverse=True):
            if max_days and days_processed >= max_days:
                break

            output_subdir = output_dir / agency / f"{yyyy}-{mm}"
            output_subdir.mkdir(parents=True, exist_ok=True)

            day = date(int(yyyy), int(mm), int(dd))
            period = TimePeriod(
                start_time=datetime.combine(day, datetime.min.time()),
                end_time=datetime.combine(day, datetime.max.time()),
            )

            print(f"\n[{agency}] {yyyy}/{mm}/{dd}")

            # gtfs pass
            _step_gtfs(base_dir, output_dir, agency)

            # parquetize vehicle positions
            vp_parquet = output_subdir / f"{dd}.vehicle-positions.parquet"
            _step_parquetize(
                "vp",
                "vehicle_position",
                feeds.get("vehicle_positions"),
                vp_parquet,
                FetchRequest(transit_agency, FeedType.VEHICLE_POSITIONS, period),
                gtfs_fetcher,
                rt_parser,
                force=args.force_analysis,
            )

            # parquetize trip updates
            tu_parquet = output_subdir / f"{dd}.trip-updates.parquet"
            _step_parquetize(
                "tu",
                "trip_update",
                feeds.get("trip_updates"),
                tu_parquet,
                FetchRequest(transit_agency, FeedType.TRIP_UPDATES, period),
                gtfs_fetcher,
                rt_parser,
                force=args.force_analysis,
            )

            # primary analysis: parquet variant
            analysis_out = output_subdir / f"{dd}.analysis.parquet"
            if parquet_orchestrator is not None:
                analysis_ok = _step_analyze(
                    transit_agency,
                    day,
                    parquet_orchestrator,
                    analysis_out,
                    force=args.force_analysis,
                )

                if args.compare and zero_orchestrator is not None:
                    legacy_out = output_subdir / f"{dd}.analysis-legacy.parquet"
                    legacy_ok = _step_legacy(
                        zero_orchestrator,
                        transit_agency,
                        day,
                        legacy_out,
                        force=args.force_analysis,
                    )
                    if analysis_ok and legacy_ok:
                        failures.extend(_diff_day(f"{agency} {yyyy}-{mm}/{dd}", analysis_out, legacy_out))
                    else:
                        print(
                            f"  comparison skipped — "
                            f"analysis={'ok' if analysis_ok else 'no output'}, "
                            f"legacy={'ok' if legacy_ok else 'no output'}"
                        )

            days_processed += 1

    if failures:
        print("\nCOMPARISON FAILURES:")
        for line in failures:
            print(line)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
        print("\nContainer Done.")
    except BaseException:
        print(f"Fatal error:\n{traceback.format_exc()}")
        sleep(1)  # Allow logs to flush before container exits
        raise
