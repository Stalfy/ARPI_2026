# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.

import re
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
from google.protobuf.json_format import MessageToDict
from google.transit import gtfs_realtime_pb2

from apex_transit_arpi.discovery.shapes import calculate_position_on_shape
from apex_transit_arpi.log import ApplicationLogger
from apex_transit_arpi.models.gtfs_rt_strategy import StopIdStrategy
from apex_transit_arpi.models.gtfs_segment import ShapeSegment
from apex_transit_arpi.models.gtfs_stop import Stop
from apex_transit_arpi.models.gtfs_stop_time import StopTime
from apex_transit_arpi.models.gtfs_trip import Trip


class GtfsRtParser:
    """Parse GTFS-RT payloads and convert them to Polars DataFrames."""

    def __init__(self) -> None:
        """Initialize parser with an application logger."""
        self._logger = ApplicationLogger(__class__.__name__)

    def gtfs_rt_to_dict(self, protobuf_data: bytes) -> dict[str, Any]:
        """Convert GTFS-RT protobuf bytes to a dictionary.

        Args:
            protobuf_data (bytes): Raw GTFS-RT protobuf bytes.

        Returns:
            dict[str, Any]: Dictionary representation of the GTFS-RT feed,
            or an empty dict if parsing fails.
        """
        if not protobuf_data:
            self._logger.wrn("Received empty protobuf data; skipping parse.")
            return {}
        try:
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(protobuf_data)
            result = MessageToDict(feed)
            entity_count = len(result.get("entity", []))
            self._logger.dbg(f"Protobuf parsed successfully ({entity_count} entity(ies)).")
            return result
        except Exception as e:
            self._logger.err(f"Failed to parse Protobuf data to dictionary: {e}")
            return {}

    def _compute_stop_id_from_vehicle_position(self, vehicle: dict) -> str | None:
        """Compute stop_id directly from vehicle position.

        Args:
            vehicle (dict): Vehicle position data.

        Returns:
            str | None: The stop_id or None if not found.
        """
        stop_id = vehicle.get("stopId") or vehicle.get("stop_id")
        if not stop_id:
            self._logger.dbg("No stopId field present in vehicle position.")
        return stop_id

    def _compute_stop_id_from_stopsequence(
        self,
        vehicle: dict,
        stop_id_by_trip_and_sequence: dict[tuple[str, int], str],
    ) -> str | None:
        """Compute stop_id by mapping current_stop_sequence with GTFS data."""
        trip_id = vehicle.get("trip", {}).get("tripId") or vehicle.get("trip", {}).get("trip_id")
        current_stop_sequence = vehicle.get("currentStopSequence")

        if not trip_id or current_stop_sequence is None:
            self._logger.dbg(f"Missing trip_id or stop_sequence for stop_id resolution (trip_id={trip_id}, seq={current_stop_sequence}).")
            return None

        return stop_id_by_trip_and_sequence.get((trip_id, int(current_stop_sequence)))

    def build_trip_stops_lookup(
        self,
        gtfs_stop_times: pl.DataFrame,
        stops: dict[str, Stop],
    ) -> dict[str, list[tuple[str, float, float]]]:
        lookup: dict[str, list[tuple[str, float, float]]] = {}

        if gtfs_stop_times.is_empty() or not stops:
            self._logger.wrn("Cannot build trip stops lookup: stop_times or stops data is empty.")
            return lookup

        for row in gtfs_stop_times.select(["trip_id", "stop_id"]).iter_rows(named=True):
            trip_id = row["trip_id"]
            stop_id = row["stop_id"]

            if not trip_id or not stop_id:
                continue

            stop = stops.get(stop_id)
            if stop is None or stop.stop_lat is None or stop.stop_lon is None:
                continue

            lookup.setdefault(trip_id, []).append((stop_id, stop.stop_lat, stop.stop_lon))

        self._logger.dbg(f"Trip stops lookup built ({len(lookup)} trip(s)).")
        return lookup

    def build_projected_trip_stops_lookup(
        self,
        trip_stops_lookup: dict[str, list[tuple[str, float, float]]],
        trips: dict[str, Trip],
        segments_by_shape: dict[str, list[ShapeSegment]],
    ) -> dict[str, list[tuple[str, float]]]:
        projected_lookup: dict[str, list[tuple[str, float]]] = {}

        for trip_id, stops in trip_stops_lookup.items():
            trip = trips.get(trip_id)
            if not trip or not trip.shape_id:
                continue

            segments = segments_by_shape.get(trip.shape_id)
            if not segments:
                continue

            projected_stops: list[tuple[str, float]] = []

            for stop_id, stop_lat, stop_lon in stops:
                projection = calculate_position_on_shape(stop_lon, stop_lat, segments)
                if not projection:
                    continue

                distance_from_start, _ = projection
                projected_stops.append((stop_id, distance_from_start))

            if projected_stops:
                projected_lookup[trip_id] = projected_stops

        self._logger.dbg(f"Projected trip stops lookup built ({len(projected_lookup)} trip(s) out of {len(trip_stops_lookup)} input trips).")
        return projected_lookup

    def _compute_stop_id_from_geolocalization(
        self,
        vehicle: dict,
        projected_trip_stops: list[tuple[str, float]],
        segments: list[ShapeSegment],
    ) -> str | None:
        position = vehicle.get("position")
        if not position:
            self._logger.dbg("Vehicle has no position data; cannot geolocalize stop.")
            return None

        vehicle_lat = position.get("latitude")
        vehicle_lon = position.get("longitude")
        vehicle_status = vehicle.get("currentStatus") or vehicle.get("current_status")

        if vehicle_lat is None or vehicle_lon is None:
            self._logger.dbg("Vehicle latitude or longitude is missing; cannot geolocalize stop.")
            return None

        if not projected_trip_stops:
            self._logger.wrn("Projected trip stops list is empty; cannot geolocalize stop.")
            return None

        vehicle_projection = calculate_position_on_shape(vehicle_lon, vehicle_lat, segments)
        if not vehicle_projection:
            self._logger.wrn("Vehicle position could not be projected onto the shape; skipping geolocalization.")
            return None

        vehicle_distance_from_start, _ = vehicle_projection
        self._logger.dbg(f"Vehicle projected at {vehicle_distance_from_start}m from shape start (status={vehicle_status}).")

        stop_projections: list[tuple[str, float]] = []
        for stop_id, stop_distance_from_start in projected_trip_stops:
            distance_diff = stop_distance_from_start - vehicle_distance_from_start
            stop_projections.append((stop_id, distance_diff))

        if vehicle_status == "STOPPED_AT":
            closest_stop = min(stop_projections, key=lambda x: abs(x[1]))
            self._logger.dbg(f"Vehicle status STOPPED_AT: closest stop selected is {closest_stop[0]}.")
            return closest_stop[0]

        for stop_id, distance_diff in stop_projections:
            if abs(distance_diff) <= 10.0:
                self._logger.dbg(f"Vehicle within 10m of stop {stop_id} (diff={distance_diff}m); stop selected.")
                return stop_id

        ahead_stops = [s for s in stop_projections if s[1] > 0]
        if ahead_stops:
            next_stop = min(ahead_stops, key=lambda x: x[1])
            self._logger.dbg(f"Next stop ahead is {next_stop[0]} ({next_stop[1]}m ahead); stop selected.")
            return next_stop[0]

        self._logger.wrn("No stop could be resolved via geolocalization: no match within 10m and no stop ahead.")
        return None

    def compute_stop_id(
        self,
        vehicle: dict,
        strategy: StopIdStrategy,
        segments: list[ShapeSegment] | None = None,
        stop_ids_by_trip_and_sequence: dict[tuple[str, int], str] | None = None,
        projected_stops_by_trip_id: dict[str, list[tuple[str, float]]] | None = None,
    ) -> str | None:
        """Compute stop_id based on the detected strategy."""
        match strategy:
            case StopIdStrategy.FROM_VEHICLE_POSITION:
                return self._compute_stop_id_from_vehicle_position(vehicle)

            case StopIdStrategy.FROM_STOPSEQUENCE:
                if stop_ids_by_trip_and_sequence is None:
                    self._logger.wrn("FROM_STOPSEQUENCE strategy selected but no stop_ids lookup provided.")
                    return None
                return self._compute_stop_id_from_stopsequence(vehicle, stop_ids_by_trip_and_sequence)

            case StopIdStrategy.FROM_GEOLOCALIZATION:
                trip_id = vehicle.get("trip", {}).get("tripId") or vehicle.get("trip", {}).get("trip_id")
                if not trip_id or segments is None or projected_stops_by_trip_id is None:
                    self._logger.dbg(
                        msg=(
                            f"FROM_GEOLOCALIZATION: missing required data (trip_id={trip_id}, "
                            f"segments_available={segments is not None}, "
                            f"lookup_available={projected_stops_by_trip_id is not None})."
                        )
                    )
                    return None

                projected_trip_stops = projected_stops_by_trip_id.get(trip_id)
                if not projected_trip_stops:
                    self._logger.dbg(f"FROM_GEOLOCALIZATION: no projected stops found for trip_id={trip_id}.")
                    return None

                return self._compute_stop_id_from_geolocalization(
                    vehicle,
                    projected_trip_stops,
                    segments,
                )

            case _:
                self._logger.wrn(f"Unknown stop_id strategy: {strategy}.")
                return None

    def parse_trip_updates_to_df(self, feed_dict: dict, stops_lookup: dict | None = None) -> pl.DataFrame:
        """Parse GTFS-RT trip updates into a Polars DataFrame.

        Supports both absolute arrival times and delay-based TripUpdates.

        Args:
            feed_dict (dict): The GTFS-RT feed data as a dictionary.
            stops_lookup (dict[tuple[str, str], StopTime] | None): Optional lookup for (trip_id, stop_id)
                -> StopTime objects with scheduled times.

        Returns:
            pl.DataFrame: A Polars DataFrame containing trip updates.
        """
        rows = []
        use_delay = bool(stops_lookup)

        for update in feed_dict.get("entity", []):
            trip_update = update.get("tripUpdate") or update.get("trip_update")
            if not trip_update:
                continue

            # [TODO] : verify that the id fields are consistent
            trip_id = trip_update["trip"].get("tripId") or trip_update["trip"].get("trip_id")
            route_id = trip_update["trip"].get("routeId") or trip_update["trip"].get("route_id")
            pred_time = trip_update.get("timestamp")
            if pred_time is None:
                pred_time = feed_dict.get("header").get("timestamp")

            if not trip_id:
                continue

            base_dt_utc = datetime.fromtimestamp(int(pred_time), tz=timezone.utc)
            base_dt_eastern = base_dt_utc.astimezone(ZoneInfo("America/Toronto"))

            # Determine service day: trips between 00:00-02:59 are often part of previous day's service
            # Check if trip_id contains a time >= 24:00 to confirm this
            is_service_day_carryover = False
            if base_dt_eastern.hour < 3:
                # Try to extract scheduled time from trip_id (format: ...HH:MM)
                time_match = re.search(r"_(\d{2}):(\d{2})$", trip_id or "")
                if time_match:
                    scheduled_hour = int(time_match.group(1))
                    if scheduled_hour >= 24:
                        is_service_day_carryover = True

            # Set service_midnight to previous day if this is a carryover trip
            if is_service_day_carryover:
                service_midnight = datetime(base_dt_eastern.year, base_dt_eastern.month, base_dt_eastern.day) - timedelta(days=1)
            else:
                service_midnight = datetime(base_dt_eastern.year, base_dt_eastern.month, base_dt_eastern.day)

            for stop_time_update in trip_update.get("stopTimeUpdate") or trip_update.get("stop_time_update", []):
                stop_id = stop_time_update.get("stopId") or stop_time_update.get("stop_id")
                if not stop_id:
                    continue

                arrival = stop_time_update.get("arrival", {})

                if use_delay:
                    pred_arrival = self._compute_pred_arrival_from_delay(
                        stops_lookup,
                        arrival.get("delay"),
                        trip_id,
                        stop_id,
                        service_midnight,
                    )
                else:
                    pred_arrival = self._compute_pred_arrival_from_time(arrival.get("time"))

                if pred_arrival is None:
                    continue

                rows.append(
                    {
                        "trip_id": trip_id,
                        "stop_id": stop_id,
                        "pred_time": int(pred_time) if pred_time else None,
                        "pred_arrival": int(pred_arrival),
                        "route_id": route_id,
                    }
                )

        if not rows:
            return pl.DataFrame()

        df = pl.DataFrame(rows)
        self._logger.dbg(f"{len(rows)} trip update row(s) parsed from feed.")
        return df.with_columns(
            [
                pl.from_epoch(pl.col("pred_time"), time_unit="s")
                .dt.replace_time_zone("UTC")
                .dt.convert_time_zone("America/Toronto")
                .dt.replace_time_zone(None)
                .alias("pred_time"),
                pl.from_epoch(pl.col("pred_arrival"), time_unit="s")
                .dt.replace_time_zone("UTC")
                .dt.convert_time_zone("America/Toronto")
                .dt.replace_time_zone(None)
                .alias("pred_arrival"),
                pl.col("route_id").cast(pl.Utf8),
            ]
        )

    def build_stop_id_lookup(self, gtfs_stop_times: pl.DataFrame) -> dict[tuple[str, int], str]:
        df = gtfs_stop_times.select(["trip_id", "stop_sequence", "stop_id"])

        lookup = {
            (trip_id, stop_sequence): stop_id
            for trip_id, stop_sequence, stop_id in df.iter_rows()
            if trip_id is not None and stop_sequence is not None and stop_id is not None
        }
        self._logger.dbg(f"Stop ID lookup built ({len(lookup)} entry(ies)).")
        return lookup

    def parse_vehicle_positions_to_df(
        self,
        feed_dict: dict,
        stop_id_strategy: StopIdStrategy,
        trips: dict[str, Trip] | None = None,
        segments: dict[str, list[ShapeSegment]] | None = None,
        stop_ids_lookup: dict[tuple[str, int], str] | None = None,
        projected_trip_stops_lookup: dict[str, list[tuple[str, float]]] | None = None,
    ) -> pl.DataFrame:
        """Parse GTFS-RT vehicle positions into a Polars DataFrame."""
        if stop_id_strategy is None:
            return pl.DataFrame()

        timestamp_trip_id_stop_id_mapping: dict[tuple[int, str, str | None], str | None] = {}
        last_timestamp_by_trip_vehicle_stop: dict[tuple[str, str | None, str], int] = {}

        for vehicle_position in feed_dict.get("entity", []):
            vehicle = vehicle_position.get("vehicle")
            if not vehicle:
                continue

            trip_id = vehicle.get("trip", {}).get("tripId") or vehicle.get("trip", {}).get("trip_id")
            vehicle_id = vehicle.get("vehicleId") or vehicle.get("vehicle_id") or vehicle.get("vehicle", {}).get("id")
            trip = trips.get(trip_id) if trips and trip_id else None
            shape_id = trip.shape_id if trip else None
            segments_from_shape = segments.get(shape_id) if shape_id and segments else None
            timestamp = vehicle.get("timestamp") if vehicle.get("timestamp") else feed_dict.get("header", {}).get("timestamp")

            stop_id = self.compute_stop_id(
                vehicle,
                stop_id_strategy,
                segments=segments_from_shape,
                stop_ids_by_trip_and_sequence=stop_ids_lookup,
                projected_stops_by_trip_id=projected_trip_stops_lookup,
            )

            if not trip_id or timestamp is None or stop_id is None:
                continue

            ts = int(timestamp)
            timestamp_trip_id_stop_id_mapping[(ts, trip_id, vehicle_id)] = stop_id

            key = (trip_id, vehicle_id, stop_id)
            prev = last_timestamp_by_trip_vehicle_stop.get(key)
            if prev is None or ts > prev:
                last_timestamp_by_trip_vehicle_stop[key] = ts

        rows = []
        for vehicle_position in feed_dict.get("entity", []):
            vehicle = vehicle_position.get("vehicle")
            if not vehicle:
                continue

            trip_id = vehicle.get("trip", {}).get("tripId") or vehicle.get("trip", {}).get("trip_id")
            timestamp = vehicle.get("timestamp") if vehicle.get("timestamp") else feed_dict.get("header", {}).get("timestamp")
            vehicle_id = vehicle.get("vehicleId") or vehicle.get("vehicle_id") or vehicle.get("vehicle", {}).get("id")

            if not trip_id or timestamp is None:
                continue

            ts = int(timestamp)
            stop_id = timestamp_trip_id_stop_id_mapping.get((ts, trip_id, vehicle_id))
            if not stop_id:
                continue

            is_last_at_stop = ts == last_timestamp_by_trip_vehicle_stop.get((trip_id, vehicle_id, stop_id))
            if not is_last_at_stop:
                continue

            rows.append(
                {
                    "trip_id": trip_id,
                    "stop_id": stop_id,
                    "actual_arrival": ts,
                    "vehicle_id": vehicle_id,
                }
            )

        if not rows:
            return pl.DataFrame()

        df = pl.DataFrame(rows)
        self._logger.dbg(f"{len(rows)} vehicle position row(s) parsed from feed.")
        return df.with_columns(
            [
                pl.from_epoch(pl.col("actual_arrival"), time_unit="s")
                .dt.replace_time_zone("UTC")
                .dt.convert_time_zone("America/Toronto")
                .dt.replace_time_zone(None)
                .alias("actual_arrival")
            ]
        )

    @staticmethod
    def hms_to_seconds(hms: str) -> int:
        """Convert an "HH:MM:SS" string to total seconds."""
        h, m, s = map(int, hms.split(":"))
        return h * 3600 + m * 60 + s

    def _compute_pred_arrival_from_time(self, arrival_time: Any) -> Any:
        """Return the absolute arrival timestamp unchanged or ``None`` if missing."""
        if arrival_time is None:
            self._logger.dbg("Arrival time is None; cannot compute predicted arrival.")
            return None
        return arrival_time

    def _compute_pred_arrival_from_delay(
        self,
        stops_lookup: dict,
        delay: Any,
        trip_id: str,
        stop_id: str,
        service_midnight: datetime,
    ) -> Any:
        """Compute predicted arrival timestamp from a delay and scheduled time."""
        if delay is None or not stops_lookup:
            self._logger.dbg(
                msg=(
                    f"Cannot compute delay-based arrival for trip_id={trip_id}, "
                    f"stop_id={stop_id}: delay={delay}, stops_lookup_available={bool(stops_lookup)}."
                )
            )
            return None

        scheduled: StopTime = stops_lookup.get((trip_id, stop_id))
        if not scheduled:
            self._logger.dbg(f"No scheduled stop_time for trip_id={trip_id}, stop_id={stop_id}; skipping delay computation.")
            return None

        scheduled_sec = GtfsRtParser.hms_to_seconds(scheduled.arrival_time)
        scheduled_dt = service_midnight + timedelta(seconds=scheduled_sec)
        pred_arrival_dt = scheduled_dt + timedelta(seconds=int(delay))
        pred_arrival = pred_arrival_dt.timestamp()
        return pred_arrival
