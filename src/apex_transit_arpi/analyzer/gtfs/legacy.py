# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.
#
# _zero variant: thread pool removed; sequential loop with original Polars anti-join deduplication.

import os
import threading
from typing import Any, Optional

import polars as pl
from tqdm import tqdm

from apex_transit_arpi.analyzer.eta_benchmark_analysis import benchmark_builder
from apex_transit_arpi.discovery.files import FileClient
from apex_transit_arpi.discovery.gtfs import GtfsFetchService
from apex_transit_arpi.discovery.rt_parser import GtfsRtParser
from apex_transit_arpi.log import ApplicationLogger
from apex_transit_arpi.models.agency_settings import VehiclePositionMappingStrategy
from apex_transit_arpi.models.gtfs_rt_strategy import StopIdStrategy
from apex_transit_arpi.models.gtfs_segment import ShapeSegment
from apex_transit_arpi.models.gtfs_trip import Trip
from apex_transit_arpi.models.transit import DelayUsingAgencies, FeedType, FetchRequest, TimePeriod, TransitAgency


class GtfsRtFetchService:
    """Service to fetch and parse GTFS-RT data into Polars DataFrames.

    Attributes:
        file_client (FileClient): Client to interact with file storage.
    """

    def __init__(self, file_client: FileClient, gtfs_fetcher: GtfsFetchService, batch_size: int | None = None) -> None:
        """Initialize the fetch service with a `FileClient` instance."""
        self.file_client = file_client
        self._logger = ApplicationLogger(__class__.__name__)
        self._parser = GtfsRtParser()
        self.cpu_cores = os.cpu_count() if batch_size is None else batch_size
        self._gtfs_fetcher = gtfs_fetcher

    # -------------------------------------------------------------------------------------------- #
    # Specifics -- Vechicle Positions retrieval
    # -------------------------------------------------------------------------------------------- #
    def fetch_vehicle_positions_df(
        self,
        fr: FetchRequest,
        configured_strategy: VehiclePositionMappingStrategy = VehiclePositionMappingStrategy.AUTOMATIC,
        cancel_event: threading.Event | None = None,
    ) -> pl.DataFrame:
        """Fetch GTFS-RT VehiclePositions and return a de-duplicated DataFrame."""
        files = self.file_client.retrieve_file_by_time_period(fetch_request=fr, cancel_event=cancel_event)
        if not files:
            self._logger.wrn(f"No {fr.feed_type} files found for {fr.transit_agency} in the specified time period.")
            return pl.DataFrame()

        self._logger.inf(f"{fr.transit_agency}, {fr.time_period.start_time.date()}: Processing {len(files)} {fr.feed_type} files.")

        # Minimum pour détecter la stratégie
        gtfs_stop_times = self._gtfs_fetcher.load_stop_times_df(fr.transit_agency)

        first_content = self.file_client.retrieve_file_content(files[0]["file_path"])
        if not first_content:
            self._logger.wrn(f"{fr.transit_agency}: First file content is empty; cannot determine stop_id strategy.")
            return pl.DataFrame()

        first_feed_dict = self._parser.gtfs_rt_to_dict(first_content)
        if not first_feed_dict:
            self._logger.wrn(f"{fr.transit_agency}: First feed failed to parse; cannot determine stop_id strategy.")
            return pl.DataFrame()

        stop_id_strategy = self._get_stop_id_strategy(first_feed_dict, gtfs_stop_times, configured_strategy)

        stop_ids_lookup: dict[tuple[str, int], str] | None = None
        trips: dict[str, Trip] | None = None
        shape_segments: dict[str, list[ShapeSegment]] | None = None
        projected_trip_stops_lookup: dict[str, list[tuple[str, float]]] | None = None

        if stop_id_strategy == StopIdStrategy.FROM_STOPSEQUENCE:
            stop_ids_lookup = self._parser.build_stop_id_lookup(gtfs_stop_times)
            self._logger.dbg(f"{fr.transit_agency}: stop_id lookup built ({len(stop_ids_lookup)} entry(ies)).")

        elif stop_id_strategy != StopIdStrategy.FROM_VEHICLE_POSITION:
            self._logger.inf(f"{fr.transit_agency}: Loading stops, trips and shape segments for strategy {stop_id_strategy}.")
            gtfs_stops = self._gtfs_fetcher.load_stops(fr.transit_agency)
            trips = self._gtfs_fetcher.load_trips(fr.transit_agency)
            shape_segments = self._gtfs_fetcher.load_shape_segments(fr.transit_agency)

            if stop_id_strategy == StopIdStrategy.FROM_GEOLOCALIZATION:
                trip_stops_lookup = self._parser.build_trip_stops_lookup(gtfs_stop_times, gtfs_stops)
                projected_trip_stops_lookup = self._parser.build_projected_trip_stops_lookup(
                    trip_stops_lookup,
                    trips,
                    shape_segments,
                )

        if stop_id_strategy is None:
            self._logger.wrn(f"{fr.transit_agency}: Unable to determine stop_id strategy.")
            return pl.DataFrame()

        dfs = []
        self._logger.inf(f"{fr.transit_agency}, {fr.time_period.start_time.date()}: Using stop_id strategy {stop_id_strategy.value}.")
        bar = tqdm(files, desc="vehicle_positions", unit="file", leave=False)
        for fi in bar:
            bar.set_postfix_str(fi.get("file_path", "").split("/")[-1])
            if cancel_event and cancel_event.is_set():
                self._logger.inf("Cancellation signal received. Stopping file processing.")
                return pl.DataFrame()

            df = self._process_file(
                fi,
                fr,
                None,
                stop_id_strategy=stop_id_strategy,
                trips=trips,
                shape_segments=shape_segments,
                stop_ids_lookup=stop_ids_lookup,
                projected_trip_stops_lookup=projected_trip_stops_lookup,
            )

            if not df.is_empty():
                dfs.append(df.unique(subset=["trip_id", "stop_id"], keep="first"))

        if not dfs:
            self._logger.wrn(
                f"{fr.transit_agency}, {fr.time_period.start_time.date()}: No vehicle position data produced from {len(files)} file(s)."
            )
            return pl.DataFrame()

        self._logger.inf(
            f"{fr.transit_agency}, {fr.time_period.start_time.date()}: Concatenating and de-duplicating DataFrames from {len(dfs)} files."
        )
        df: pl.DataFrame = pl.concat(dfs, how="vertical").unique(subset=["trip_id", "stop_id"], keep="first")
        df = df.with_columns(pl.lit(fr.transit_agency).alias("transit_agency"))
        self._logger.dbg(f"{fr.transit_agency}, {fr.time_period.start_time.date()}: Actuals {df.shape}.")
        return df

    def _get_stop_id_strategy(
        self,
        first_feed_dict: dict[str, Any],
        gtfs_stop_times: pl.DataFrame,
        configured_strategy: VehiclePositionMappingStrategy = VehiclePositionMappingStrategy.AUTOMATIC,
    ) -> Optional[StopIdStrategy]:
        match configured_strategy:
            case VehiclePositionMappingStrategy.AUTOMATIC:
                return self._analyze_stop_id_strategy(
                    first_feed_dict,
                    gtfs_stop_times,
                )
            case VehiclePositionMappingStrategy.FROM_VEHICLE_POSITION:
                return StopIdStrategy.FROM_VEHICLE_POSITION
            case VehiclePositionMappingStrategy.FROM_STOPSEQUENCE:
                return StopIdStrategy.FROM_STOPSEQUENCE
            case VehiclePositionMappingStrategy.FROM_GEOLOCALIZATION:
                return StopIdStrategy.FROM_GEOLOCALIZATION
            case _:
                raise ValueError(f"This configured strategy does not exist : {configured_strategy}.")

    def _analyze_stop_id_strategy(self, feed_dict: dict, gtfs_stop_times: pl.DataFrame | None = None) -> StopIdStrategy | None:
        """Analyze vehicle positions and GTFS static data to determine stop_id calculation strategy."""
        entities = feed_dict.get("entity", [])
        if not entities:
            self._logger.wrn("No entities found in the feed. Cannot determine stop_id strategy.")
            return None

        entity = entities[0]
        vehicle = entity.get("vehicle")
        if not vehicle:
            self._logger.wrn("No vehicle information found in the first feed entity. Cannot determine stop_id strategy.")
            return None

        # Direct stop_id in vehicle position
        if vehicle.get("stopId"):
            self._logger.dbg("Stop ID strategy determined: FROM_VEHICLE_POSITION")
            return StopIdStrategy.FROM_VEHICLE_POSITION

        trip_id = vehicle.get("trip", {}).get("tripId")
        current_stop_sequence = vehicle.get("currentStopSequence")

        # trip_id + current_stop_sequence
        if trip_id and current_stop_sequence is not None and gtfs_stop_times is not None:
            matching = gtfs_stop_times.filter((pl.col("trip_id") == trip_id) & (pl.col("stop_sequence") == current_stop_sequence))
            if matching.height > 0 and "stop_id" in matching.columns:
                self._logger.dbg("Stop ID strategy determined: FROM_STOPSEQUENCE")
                return StopIdStrategy.FROM_STOPSEQUENCE

        # lat/lon + trip_id + static stops available
        position = vehicle.get("position")
        if position and trip_id and gtfs_stop_times is not None:
            lat = position.get("latitude")
            lon = position.get("longitude")
            if lat is not None and lon is not None:
                self._logger.dbg("Stop ID strategy determined: FROM_GEOLOCALIZATION")
                return StopIdStrategy.FROM_GEOLOCALIZATION

        self._logger.wrn("Unable to determine stop_id strategy based on feed content and GTFS data.")
        return None

    # -------------------------------------------------------------------------------------------- #
    # Specifics -- Trip Updates Benchmark
    # -------------------------------------------------------------------------------------------- #
    def process_trip_updates(
        self, agency: TransitAgency, period: TimePeriod, actuals_df: pl.DataFrame, cancel_event: threading.Event | None = None
    ) -> tuple[pl.DataFrame, dict]:
        """Stream TripUpdate files, run benchmark per-file, and only keep benchmark results.

        Args:
            transit_agency (TransitAgency): The transit agency identifier.
            time_period (TimePeriod): The time period for fetching GTFS-RT data.
            actuals_df (pl.DataFrame): DataFrame containing actual vehicle positions.
            benchmark (BenchmarkDefinition): Instance to build benchmark results.
            postgre_client (PostgreClient): Client to persist benchmark results.

        Returns:
            dict: Aggregated daily statistics from the benchmark analysis.
        """
        req = FetchRequest(transit_agency=agency, feed_type=FeedType.TRIP_UPDATES, time_period=period)

        stops_mappings = None
        if agency in DelayUsingAgencies:
            self._logger.inf(f"{agency}: Delay-based agency detected; loading stop_times mappings.")
            stops_mappings = self._gtfs_fetcher.load_stop_times_mappings(transit_agency=agency)
            if not stops_mappings:
                self._logger.err(f"Failed to load stop_times.txt for {agency}. Cannot parse TripUpdates accurately.")
                raise ValueError(f"Failed to load stop_times.txt for {agency}. Cannot parse TripUpdates accurately.")
            self._logger.dbg(f"{agency}: {len(stops_mappings)} stop_times mapping entries available.")

        self._logger.inf(f"{agency}, : Starting the daily benchmark analysis.")

        files = self.file_client.retrieve_file_by_time_period(req, cancel_event=cancel_event)
        if not files:
            self._logger.wrn(f"No {req.feed_type} files found for {req.transit_agency} in the specified time period.")

        stats = {
            "total_predictions": 0,
            "predictions_horizon_over_15_min": 0,
            "merged_predictions": 0,
            "merged_predictions_over_15_min": 0,
            "benchmark_predictions": 0,
        }

        df_seen = pl.DataFrame([], schema={"key_hash": pl.UInt64})
        accumulated: list[pl.DataFrame] = []

        bar = tqdm(files or [], desc="trip_updates", unit="file", leave=False)
        for f in bar:
            bar.set_postfix_str(f.get("file_path", "").split("/")[-1])
            if cancel_event and cancel_event.is_set():
                self._logger.inf("Cancellation signal received. Stopping file processing.")
                break

            try:
                df = self._process_file(f, req, stops_mappings, None)
                if df.is_empty():
                    continue

                df = df.with_columns(((pl.col("pred_arrival") - pl.col("pred_time")).dt.total_seconds()).alias("prediction_horizon_sec"))
                benchmark = benchmark_builder.build(df, actuals_df, agency, period)

                stats["total_predictions"] += benchmark.statistics["total_predictions"]
                stats["predictions_horizon_over_15_min"] += df.filter(pl.col("prediction_horizon_sec") > 900).height
                stats["merged_predictions"] += benchmark.statistics["merged_predictions"]
                stats["merged_predictions_over_15_min"] += benchmark.statistics["merged_predictions_over_15_min"]

                if benchmark.df.is_empty():
                    continue

                bdf = benchmark.df.with_columns(pl.concat_str(["trip_id", "stop_id", "pred_time"]).hash().alias("key_hash"))
                df_unique = bdf.join(df_seen, on="key_hash", how="anti")
                df_seen = pl.concat([df_seen, df_unique.select("key_hash")])
                clean = df_unique.drop("key_hash")
                if not clean.is_empty():
                    accumulated.append(clean)
                stats["benchmark_predictions"] += clean.height

            except Exception as e:
                self._logger.err(f"Streaming error on file {f}: {e}")

        result_df = pl.concat(accumulated) if accumulated else pl.DataFrame()
        return result_df, stats

    # -------------------------------------------------------------------------------------------- #
    # Helpers
    # -------------------------------------------------------------------------------------------- #
    def _process_file(
        self,
        file_info,
        fetch_request,
        stops_lookup,
        stop_id_strategy: StopIdStrategy | None = None,
        trips: dict[str, Trip] | None = None,
        shape_segments: dict[str, list[ShapeSegment]] | None = None,
        stop_ids_lookup: dict[tuple[str, int], str] | None = None,
        projected_trip_stops_lookup: dict[str, list[tuple[str, float]]] | None = None,
    ) -> pl.DataFrame:
        """Process a single GCS file: download, parse, and convert to DataFrame."""
        try:
            content = self.file_client.retrieve_file_content(file_info["file_path"])
            if not content:
                self._logger.wrn(f"Empty content retrieved from {file_info['file_path']}; skipping.")
                return pl.DataFrame()

            feed_dict = self._parser.gtfs_rt_to_dict(content)
            if not feed_dict:
                self._logger.wrn(f"Feed dict is empty for {file_info['file_path']}; skipping.")
                return pl.DataFrame()

            df = pl.DataFrame()
            if fetch_request.feed_type == FeedType.TRIP_UPDATES:
                df = self._parser.parse_trip_updates_to_df(feed_dict, stops_lookup)
            elif fetch_request.feed_type == FeedType.VEHICLE_POSITIONS:
                if stop_id_strategy is None:
                    return pl.DataFrame()

                df = self._parser.parse_vehicle_positions_to_df(
                    feed_dict,
                    stop_id_strategy=stop_id_strategy,
                    trips=trips,
                    segments=shape_segments,
                    stop_ids_lookup=stop_ids_lookup,
                    projected_trip_stops_lookup=projected_trip_stops_lookup,
                )

            self._logger.dbg(f"{fetch_request.feed_type.value}: {file_info['file_path']} → {df.shape[0]} row(s) parsed.")
            return df

        except Exception as e:
            self._logger.err(f"Error processing file {file_info['file_path']}: {e}")
            return pl.DataFrame()
