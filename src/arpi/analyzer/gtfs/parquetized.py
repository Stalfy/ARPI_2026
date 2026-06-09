# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.
#
# _parquet variant: loads pre-generated TU and VP parquets (produced by the parquetizer),
# then computes predictions and actuals vectorised with Polars instead of parsing .pb files per-file.
# Raises FileNotFoundError if the parquet for the requested day has not been generated yet.

import threading
from pathlib import Path

import polars as pl
import tqdm

from arpi.analyzer.eta_benchmark_analysis import benchmark_builder
from arpi.discovery.files import FileClient
from arpi.discovery.gtfs import GtfsFetchService
from arpi.discovery.rt_parser import GtfsRtParser
from arpi.log import ApplicationLogger
from arpi.models.transit import FetchRequest, TimePeriod, TransitAgency
from arpi.parquetizer import trip_updates, vehicle_positions


class GtfsRtFetchService:
    """Service to fetch and process GTFS-RT data into Polars DataFrames using parquet caches.

    Both VP and TU data are cached as parquet files under ``parquet_dir``.  Each is
    generated on first access from the raw ``.pb`` files and reused on subsequent calls.

    Attributes:
        file_client (FileClient): Client to interact with file storage.
        parquet_dir (Path | None): Directory where VP and TU parquets are cached.
            If None, both ``fetch_vehicle_positions_df`` and ``process_trip_updates``
            raise ``ValueError``.
    """

    def __init__(
        self,
        file_client: FileClient,
        gtfs_fetcher: GtfsFetchService,
        parquet_dir: Path | None = None,
    ) -> None:
        """Initialize the fetch service with a ``FileClient`` instance."""
        self.file_client = file_client
        self._logger = ApplicationLogger(__class__.__name__)
        self._gtfs_fetcher = gtfs_fetcher
        self._parquet_dir = parquet_dir
        self._parser = GtfsRtParser()

    # -------------------------------------------------------------------------------------------- #
    # Specifics -- Vehicle Positions retrieval  (parquet-based)
    # -------------------------------------------------------------------------------------------- #
    def fetch_vehicle_positions_df(self, fr: FetchRequest, *args, **kwargs) -> pl.DataFrame:
        _ = args, kwargs
        path = vehicle_positions.vp_parquet_path(self._parquet_dir, fr.transit_agency, fr.time_period)
        if not path.exists():
            raise FileNotFoundError(f"VP parquet not found: {path}. Run the parquetizer first.")

        return pl.read_parquet(path).select(["trip_id", "stop_id", "actual_arrival", "vehicle_id", "transit_agency"])

    # -------------------------------------------------------------------------------------------- #
    # Specifics -- Trip Updates Benchmark  (parquet-based)
    # -------------------------------------------------------------------------------------------- #
    def process_trip_updates(
        self, agency: TransitAgency, period: TimePeriod, actuals_df: pl.DataFrame, cancel_event: threading.Event | None = None
    ) -> tuple[pl.DataFrame, dict]:
        """Run the benchmark per file against the processed TU predictions parquet.

        The parquetizer is responsible for parsing raw .pb files, resolving
        arrival times, and deduplicating predictions. This method reads the
        resulting parquet and runs ``benchmark_builder.build`` per file,
        accumulating statistics across the day.

        Args:
            agency: The transit agency identifier.
            period: The time period for fetching GTFS-RT data.
            actuals_df: DataFrame containing actual vehicle positions.
            cancel_event: Optional threading event to signal cancellation.

        Returns:
            Tuple of (benchmark DataFrame, statistics dict).
        """
        if self._parquet_dir is None:
            raise ValueError("parquet_dir must be provided to use the parquet variant.")

        parquet_path = trip_updates.tu_parquet_path(self._parquet_dir, agency, period)
        if not parquet_path.exists():
            raise FileNotFoundError(f"TU parquet not found: {parquet_path}. Run the parquetizer first.")

        df_seen = pl.DataFrame([], schema={"key_hash": pl.UInt64})
        accumulated: list[pl.DataFrame] = []
        stats = {
            "total_predictions": 0,
            "predictions_horizon_over_15_min": 0,
            "merged_predictions": 0,
            "merged_predictions_over_15_min": 0,
            "benchmark_predictions": 0,
        }

        self._logger.inf(f"{agency}: Starting the daily benchmark analysis (parquet).")
        lf = pl.scan_parquet(parquet_path).select(["trip_id", "stop_id", "pred_time", "pred_arrival", "route_id", "file"])
        total_rows = lf.select(pl.len()).collect().item()
        if total_rows == 0:
            self._logger.wrn(f"{agency}, {period.start_time.date()}: TU parquet is empty.")
            return pl.DataFrame(), stats

        files = lf.select("file").unique(maintain_order=True).collect()["file"].to_list()
        self._logger.inf(f"{agency}: TU parquet — {total_rows:,} predictions from {len(files):,} file(s).")
        progress = tqdm.tqdm(files, desc="trip_updates", unit="file", leave=False)
        for file_name in progress:
            progress.set_postfix_str(file_name)
            if cancel_event and cancel_event.is_set():
                self._logger.inf("Cancellation signal received. Stopping processing.")
                break

            try:
                df = lf.filter(pl.col("file") == file_name).collect()
                if df.is_empty():
                    continue

                benchmark = benchmark_builder.build(df, actuals_df, agency, period)

                stats["total_predictions"] += benchmark.statistics["total_predictions"]
                stats["predictions_horizon_over_15_min"] += df.filter(
                    (pl.col("pred_arrival") - pl.col("pred_time")).dt.total_seconds() > 900
                ).height
                stats["merged_predictions"] += benchmark.statistics["merged_predictions"]
                stats["merged_predictions_over_15_min"] += benchmark.statistics["merged_predictions_over_15_min"]

                if not benchmark.df.is_empty():
                    bdf = benchmark.df.with_columns(pl.concat_str(["trip_id", "stop_id", "pred_time"]).hash().alias("key_hash"))
                    df_unique = bdf.join(df_seen, on="key_hash", how="anti")
                    df_seen = pl.concat([df_seen, df_unique.select("key_hash")])
                    clean = df_unique.drop("key_hash")
                    if not clean.is_empty():
                        accumulated.append(clean)
                        stats["benchmark_predictions"] += clean.height

            except Exception as e:
                self._logger.err(f"Streaming error on file {file_name}: {e}")

        result_df = pl.concat(accumulated) if accumulated else pl.DataFrame()
        return result_df, stats
