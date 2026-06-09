# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.

import datetime as dt
import enum
import threading
import time
import traceback
from pathlib import Path

import polars as pl

from arpi.analyzer.gtfs.duckdb import GtfsRtFetchService
from arpi.analyzer.utils import format_duration
from arpi.discovery.files import FileClient
from arpi.discovery.gtfs import GtfsFetchService
from arpi.log import ApplicationLogger
from arpi.models.agency_settings import VehiclePositionMappingStrategy
from arpi.models.transit import FeedType, FetchRequest, TimePeriod, TransitAgency


class DailyAnalysisStatus(enum.Enum):
    """Outcome of a single-day benchmark run."""

    SUCCESS = 0
    SKIP = 1
    FAIL = 2


class BenchmarkOrchestrator:
    """Coordinate the end-to-end ETA benchmark process.

    Attributes:
        file_client: Filesystem client used to access GTFS-RT files.
        gtfs_fetcher: Service to fetch static GTFS data.
        gtfs_rt_fetcher: Service to fetch and process GTFS-RT data.
        output_dir: Directory where analysis parquet files are written.
        mapping_strategy: Vehicle position mapping strategy to use.
    """

    def __init__(
        self,
        file_client: FileClient,
        gtfs_fetcher: GtfsFetchService,
        gtfs_rt_fetcher: GtfsRtFetchService,
        output_dir: Path,
        mapping_strategy: VehiclePositionMappingStrategy = VehiclePositionMappingStrategy.AUTOMATIC,
    ) -> None:
        """Initialize orchestrator with required clients and services."""
        self._logger = ApplicationLogger(__class__.__name__)
        self.file_client = file_client
        self.gtfs_fetcher = gtfs_fetcher
        self.gtfs_rt_fetcher = gtfs_rt_fetcher
        self.output_dir = output_dir
        self.mapping_strategy = mapping_strategy

    def benchmark_period(self, agency: TransitAgency, period: TimePeriod, cancel_event: threading.Event | None = None) -> dict:
        """Run ETA benchmark day-by-day over a time period."""
        self._logger.inf(f"{agency}, Starting analysis from {period.start_time.date()} to {period.end_time.date()}.")
        period_start_time = time.time()
        period_stats = {"days_processed": 0, "days_skipped": 0, "days_failed": 0, "errors": [], "execution_time": ""}

        day = period.start_time.date()
        while day <= period.end_time.date():
            if cancel_event and cancel_event.is_set():
                self._logger.inf(f"{agency}: Cancellation requested, stopping at {day}.")
                break
            day_start_time = time.time()
            try:
                day_status, day_df = self.benchmark_day(agency, day, cancel_event=cancel_event)
                match day_status:
                    case DailyAnalysisStatus.SKIP:
                        period_stats["days_skipped"] += 1
                        self._logger.dbg(f"{agency}, {day}: Day skipped.")
                    case DailyAnalysisStatus.FAIL:
                        period_stats["days_failed"] += 1
                        self._logger.wrn(f"{agency}, {day}: Day failed with no benchmark predictions.")
                    case DailyAnalysisStatus.SUCCESS:
                        output_path = self.output_dir / str(agency) / day.strftime("%Y-%m") / f"{day}.analysis.parquet"
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        day_df.write_parquet(output_path, compression="zstd")
                        period_stats["days_processed"] += 1
                        self._logger.inf(f"{agency}, {day}: {day_df.height} benchmark rows written to {output_path}.")
                    case _:
                        raise NotImplementedError(f"Unreachable status reached ({day_status.value})")
            except Exception as e:
                period_stats["days_failed"] += 1
                period_stats["errors"].append({"date": str(day), "error": str(e), "traceback": traceback.format_exc()})
                self._logger.err(f"Failed processing {day}: {e}\n{traceback.format_exc()}")
            finally:
                self._logger.dbg(f"{agency}, {day}: Completed in {time.time() - day_start_time}.")
                day += dt.timedelta(days=1)

        period_stats["execution_time"] = format_duration(time.time() - period_start_time)
        self._logger.inf(
            f"{agency}: "
            f"Processed={period_stats['days_processed']}, "
            f"Skipped={period_stats['days_skipped']}, "
            f"Failed={period_stats['days_failed']}, "
            f"Time={period_stats['execution_time']}"
        )
        return period_stats

    def benchmark_day(
        self,
        agency: TransitAgency,
        day: dt.date,
        cancel_event: threading.Event | None = None,
    ) -> tuple[DailyAnalysisStatus, pl.DataFrame]:
        """Run the benchmark pipeline for a single day."""
        self._logger.inf(f"{agency}, {day}: Starting daily benchmark.")
        day_start = dt.datetime.combine(day, dt.datetime.min.time())
        day_end = dt.datetime.combine(day, dt.datetime.max.time())
        period = TimePeriod(start_time=day_start, end_time=day_end)

        result, _ = self._day_has_trip_updates(agency, day, cancel_event)
        if not result:
            self._logger.inf(f"{agency}, {day}: Skipping — no trip update files available.")
            return DailyAnalysisStatus.SKIP, pl.DataFrame()

        result, actuals = self._fetch_day_actual_df(agency, day, period, self.mapping_strategy, cancel_event)
        if not result:
            self._logger.inf(f"{agency}, {day}: Skipping — no vehicle position actuals available.")
            return DailyAnalysisStatus.SKIP, pl.DataFrame()

        self._logger.dbg(f"{agency}, {day}: Actuals ready; running benchmark.")

        result, day_df = self._run_day_benchmark(agency, day, period, actuals, cancel_event)
        if not result:
            self._logger.wrn(f"{agency}, {day}: Benchmark run produced no predictions; day marked as failed.")
            return DailyAnalysisStatus.FAIL, pl.DataFrame()

        self._logger.inf(f"{agency}, {day}: Day completed successfully.")
        return DailyAnalysisStatus.SUCCESS, day_df

    def _day_has_trip_updates(self, agency: TransitAgency, day: dt.date, cancel_event: threading.Event | None = None) -> tuple[bool, int]:
        trip_updates_count = self.file_client.count_files_for_date(agency, FeedType.TRIP_UPDATES, day, cancel_event=cancel_event)
        if trip_updates_count == 0:
            self._logger.wrn(f"{agency}, {day}: No trip updates found.")
            return False, trip_updates_count
        self._logger.dbg(f"{agency}, {day}: {trip_updates_count} trip update file(s) found.")
        return True, trip_updates_count

    def _fetch_day_actual_df(
        self,
        agency: TransitAgency,
        day: dt.date,
        period: TimePeriod,
        configured_strategy: VehiclePositionMappingStrategy = VehiclePositionMappingStrategy.AUTOMATIC,
        cancel_event: threading.Event | None = None,
    ):
        req = FetchRequest(agency, FeedType.VEHICLE_POSITIONS, period)
        actuals = self.gtfs_rt_fetcher.fetch_vehicle_positions_df(req, configured_strategy=configured_strategy, cancel_event=cancel_event)
        if actuals.is_empty():
            self._logger.wrn(f"{agency}, {day}: No actual data fetched.")
            return False, actuals
        self._logger.dbg(f"{agency}, {day}: Actuals fetched ({actuals.shape[0]} row(s)).")
        return True, actuals

    def _run_day_benchmark(
        self, agency: TransitAgency, day: dt.date, period: TimePeriod, actuals: pl.DataFrame, cancel_event: threading.Event | None = None
    ):
        df, stats = self.gtfs_rt_fetcher.process_trip_updates(agency, period, actuals, cancel_event=cancel_event)
        if stats["benchmark_predictions"] == 0:
            self._logger.wrn(f"{agency}, {day}: No benchmark predictions generated.")
            return False, pl.DataFrame()
        self._logger.dbg(f"{agency}, {day}: {stats['benchmark_predictions']} benchmark prediction(s) generated.")
        return True, df
