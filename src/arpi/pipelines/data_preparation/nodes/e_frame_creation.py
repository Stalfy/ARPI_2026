# nodes.py

from __future__ import annotations

import csv
import logging
from time import sleep
from typing import Any
import zipfile
from datetime import date, datetime, timedelta
from io import TextIOWrapper
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo

import polars as pl
from google.transit import gtfs_realtime_pb2
from tqdm import tqdm

logger = logging.getLogger(__name__)

MATCHES_CSV_NAME = "feed_file_sync.csv"
OUTPUT_CSV_NAME = "vehicle_position_timesteps.csv"

GTFS_ROOT = "GTFS"
GTFS_RT_ROOT = "GTFS-RT"
VEHICLE_POSITIONS = "vehicle_positions"


def _load_file_pairs(matches_zip_path: str) -> dict[str, str]:
    pairs: dict[str, str] = {}

    with zipfile.ZipFile(matches_zip_path, "r") as z:
        with z.open(MATCHES_CSV_NAME, "r") as raw:
            text = TextIOWrapper(raw, encoding="utf-8", newline="")
            reader = csv.DictReader(text)

            for row in reader:
                vp_file = row.get("vehicle_positions_file", "")
                tu_file = row.get("trip_updates_file", "")

                if vp_file and tu_file:
                    pairs[vp_file] = tu_file

    return dict(sorted(pairs.items()))


def _parse_vp_path(path: str) -> tuple[str, str] | None:
    parts = PurePosixPath(path).parts

    if len(parts) < 7:
        return None

    root, agency, feed_type, yyyy, mm, dd = parts[:6]

    if root != GTFS_RT_ROOT:
        return None

    if feed_type != VEHICLE_POSITIONS:
        return None

    date_str = f"{yyyy}-{mm}-{dd}"
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    weekday_name = dt.strftime("%A")
    is_weekend = dt.weekday() >= 5
    return agency, date_str, weekday_name, is_weekend


def _read_vehicle_positions(
    zin: zipfile.ZipFile,
    vp_file: str,
) -> dict[tuple[str, str], gtfs_realtime_pb2.VehiclePosition]:
    feed = gtfs_realtime_pb2.FeedMessage()

    with zin.open(vp_file, "r") as f:
        feed.ParseFromString(f.read())

    positions: dict[tuple[str, str], gtfs_realtime_pb2.VehiclePosition] = {}

    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue

        vp = entity.vehicle
        trip_id = vp.trip.trip_id
        vehicle_id = vp.vehicle.id

        if not trip_id or not vehicle_id:
            continue

        positions[(trip_id, vehicle_id)] = vp

    return positions


def _read_trip_updates_for_file(zin: zipfile.ZipFile, tu_file: str) -> dict[str, list[gtfs_realtime_pb2.TripUpdate.StopTimeUpdate]]:
    feed = gtfs_realtime_pb2.FeedMessage()
    with zin.open(tu_file, "r") as f:
        feed.ParseFromString(f.read())

    updates_by_trip_id: dict[str, list[gtfs_realtime_pb2.TripUpdate.StopTimeUpdate]] = {}
    for entity in feed.entity:
        if not entity.HasField("trip_update"):
            continue

        tu = entity.trip_update
        trip_id = tu.trip.trip_id
        if not trip_id:
            continue

        updates_by_trip_id.setdefault(trip_id, []).extend(tu.stop_time_update)

    return feed.header.timestamp, updates_by_trip_id


def _load_gtfs_trips(
    zin: zipfile.ZipFile,
) -> dict[tuple[str, str], dict[str, str]]:
    trips_by_key: dict[tuple[str, str], dict[str, str]] = {}

    trip_files = [name for name in zin.namelist() if name.startswith(f"{GTFS_ROOT}/") and name.endswith("/trips.txt")]

    for trips_path in tqdm(trip_files, desc="Loading GTFS trips", unit="file"):
        parts = PurePosixPath(trips_path).parts

        if len(parts) != 3:
            continue

        _root, agency, _filename = parts

        with zin.open(trips_path) as f:
            trips = pl.read_csv(f, infer_schema_length=0).with_columns(
                pl.col("trip_id").cast(pl.String),
                pl.col("route_id").cast(pl.String),
            )

        for column in [
            "service_id",
            "shape_id",
            "trip_headsign",
            "direction_id",
            "block_id",
        ]:
            if column not in trips.columns:
                trips = trips.with_columns(pl.lit("").alias(column))
            else:
                trips = trips.with_columns(pl.col(column).cast(pl.String))

        for row in trips.select(
            [
                "route_id",
                "service_id",
                "trip_id",
                "shape_id",
                "trip_headsign",
                "direction_id",
                "block_id",
            ]
        ).iter_rows(named=True):
            trip_id = row["trip_id"]

            trips_by_key[(agency, trip_id)] = {
                "route_id": row.get("route_id") or "",
                "service_id": row.get("service_id") or "",
                "trip_id": row.get("trip_id") or "",
                "shape_id": row.get("shape_id") or "",
                "trip_headsign": row.get("trip_headsign") or "",
                "direction_id": row.get("direction_id") or "",
                "block_id": row.get("block_id") or "",
            }

    logger.info("Loaded %d GTFS trip reference(s)", len(trips_by_key))

    return trips_by_key


def _load_gtfs_stops(
    zin: zipfile.ZipFile,
) -> dict[tuple[str, str], dict[str, str]]:
    stops_by_key: dict[tuple[str, str], dict[str, str]] = {}

    stop_files = [name for name in zin.namelist() if name.startswith(f"{GTFS_ROOT}/") and name.endswith("/stops.txt")]

    for stops_path in tqdm(stop_files, desc="Loading GTFS stops", unit="file"):
        parts = PurePosixPath(stops_path).parts

        if len(parts) != 3:
            continue

        _root, agency, _filename = parts

        with zin.open(stops_path) as f:
            stops = pl.read_csv(f, infer_schema_length=0).with_columns(
                pl.col("stop_id").cast(pl.String),
            )

        for column in ["stop_name"]:
            if column not in stops.columns:
                stops = stops.with_columns(pl.lit("").alias(column))
            else:
                stops = stops.with_columns(pl.col(column).cast(pl.String))

        for row in stops.select(["stop_id", "stop_name"]).iter_rows(named=True):
            stop_id = row["stop_id"]
            stops_by_key[(agency, stop_id)] = {
                "stop_id": stop_id,
                "stop_name": row.get("stop_name") or "",
            }

    logger.info("Loaded %d GTFS stop reference(s)", len(stops_by_key))

    return stops_by_key


def _load_gtfs_stop_times(zin: zipfile.ZipFile) -> tuple[
    dict[tuple[str, str, int], dict[str, str]],
    dict[tuple[str, str, str], list[dict[str, str]]],
]:
    by_trip_sequence: dict[tuple[str, str, int], dict[str, Any]] = {}
    by_trip_stop_id: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    stop_time_files = [name for name in zin.namelist() if name.startswith(f"{GTFS_ROOT}/") and name.endswith("/stop_times.txt")]

    for stop_times_path in tqdm(stop_time_files, desc="Loading GTFS stop_times", unit="file"):
        parts = PurePosixPath(stop_times_path).parts
        if len(parts) != 3:
            continue

        _root, agency, _filename = parts
        with zin.open(stop_times_path) as f:
            stop_times = pl.read_csv(f, infer_schema_length=0)

        stop_times = stop_times.with_columns(
            pl.col("trip_id").cast(pl.String), pl.col("stop_id").cast(pl.String), pl.col("stop_sequence").cast(pl.Int64)
        )

        for column in ["arrival_time", "departure_time"]:
            if column not in stop_times.columns:
                stop_times = stop_times.with_columns(pl.lit("").alias(column))
            else:
                stop_times = stop_times.with_columns(pl.col(column).cast(pl.String))

        stop_times = stop_times.sort(["trip_id", "stop_sequence"])
        for trip_id, trip_stop_times in stop_times.group_by("trip_id", maintain_order=True):
            rows = list(trip_stop_times.select(["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time", "shape_dist_traveled"]).iter_rows(named=True))
            trip_len = len(rows)

            shape_total_distance = max([float(x["shape_dist_traveled"]) for x in rows])
            for idx, row in enumerate(rows):
                stop_id = row["stop_id"]
                stop_sequence = int(row["stop_sequence"])
                arrival_time = row.get("arrival_time") or ""
                departure_time = row.get("departure_time") or ""
                arrival_seconds = _gtfs_time_to_seconds(arrival_time)
                departure_seconds = _gtfs_time_to_seconds(departure_time)

                previous_departure_seconds = None
                if idx > 0:
                    previous_departure_seconds = _gtfs_time_to_seconds(rows[idx - 1].get("departure_time") or "")

                dwell_time_seconds = None
                if arrival_seconds is not None and departure_seconds is not None:
                    dwell_time_seconds = departure_seconds - arrival_seconds

                running_time_seconds = None
                if previous_departure_seconds is not None and arrival_seconds is not None:
                    running_time_seconds = arrival_seconds - previous_departure_seconds

                item = {
                    "trip_id": str(trip_id[0]),
                    "stop_id": stop_id,
                    "stop_sequence": str(stop_sequence),
                    "arrival_time": arrival_time,
                    "departure_time": departure_time,
                    "is_first_stop": idx == 0,
                    "is_last_stop": idx == trip_len - 1,
                    "shape_dist_traveled": float(row["shape_dist_traveled"]),
                    "stops_progression": float(idx) / float(trip_len) if trip_len > 1 else 1.0,
                    "stops_count": trip_len,
                    "shape_progression": float(row["shape_dist_traveled"]) / shape_total_distance,
                    "shape_distance": shape_total_distance,
                    "dwell_time_seconds": dwell_time_seconds,
                    "running_time_seconds": running_time_seconds,
                }

                by_trip_sequence[(agency, str(trip_id[0]), stop_sequence)] = item
                by_trip_stop_id.setdefault((agency, str(trip_id[0]), stop_id), []).append(item)

    logger.info("Loaded %d GTFS stop_time sequence reference(s)", len(by_trip_sequence))
    return by_trip_sequence, by_trip_stop_id


def _load_agency_timezones(zin: zipfile.ZipFile) -> dict[str, ZoneInfo]:
    timezones: dict[str, ZoneInfo] = {}
    agency_files = [name for name in zin.namelist() if name.startswith(f"{GTFS_ROOT}/") and name.endswith("/agency.txt")]
    for agency_path in agency_files:
        parts = PurePosixPath(agency_path).parts

        if len(parts) != 3:
            continue

        _root, agency, _filename = parts
        with zin.open(agency_path) as f:
            df = pl.read_csv(f, infer_schema_length=0)

        timezone_name = "America/Montreal"
        if not df.is_empty() and "agency_timezone" in df.columns:
            timezone_name = str(df["agency_timezone"][0])

        timezones[agency] = ZoneInfo(timezone_name)

    return timezones


def _resolve_scheduled_stop_time(
    agency: str,
    trip_id: str,
    stop_sequence: int | None,
    stop_id: str | None,
    by_trip_sequence: dict[tuple[str, str, int], dict[str, str]],
    by_trip_stop_id: dict[tuple[str, str, str], list[dict[str, str]]],
) -> dict[str, str]:
    if stop_sequence is not None:
        scheduled = by_trip_sequence.get((agency, trip_id, stop_sequence))
        if scheduled:
            return scheduled

    if stop_id:
        candidates = by_trip_stop_id.get((agency, trip_id, stop_id), [])
        if len(candidates) == 1:
            return candidates[0]

    return {}


def _gtfs_time_to_seconds(value: str | None) -> int | None:
    if not value:
        return None

    h, m, s = value.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def _gtfs_time_to_posix(service_date: str, gtfs_time: str | None, agency_tz: ZoneInfo) -> int | None:
    total_seconds = _gtfs_time_to_seconds(gtfs_time)

    if total_seconds is None:
        return None

    service_day = date.fromisoformat(service_date)
    local_midnight = datetime(service_day.year, service_day.month, service_day.day, tzinfo=agency_tz)

    local_dt = local_midnight + timedelta(seconds=total_seconds)
    return int(local_dt.timestamp())


def _stop_time_event_to_posix(event, scheduled_posix: int | None) -> int | str:
    if event is None:
        return ""

    if event.HasField("time"):
        return int(event.time)

    if event.HasField("delay") and scheduled_posix is not None:
        return scheduled_posix + int(event.delay)

    return ""


def _stu_arrival_to_posix(stu, scheduled_arrival_posix: int | None) -> int | str:
    if not stu.HasField("arrival"):
        return ""

    return _stop_time_event_to_posix(stu.arrival, scheduled_arrival_posix)


def _stu_departure_to_posix(stu, scheduled_departure_posix: int | None) -> int | str:
    if not stu.HasField("departure"):
        return ""

    return _stop_time_event_to_posix(stu.departure, scheduled_departure_posix)


def _vehicle_timestamp(vp: gtfs_realtime_pb2.VehiclePosition) -> int | str:
    if vp.HasField("timestamp"):
        return int(vp.timestamp)

    return ""


def _vehicle_latitude(vp: gtfs_realtime_pb2.VehiclePosition) -> float | str:
    if vp.HasField("position"):
        return float(vp.position.latitude)

    return ""


def _vehicle_longitude(vp: gtfs_realtime_pb2.VehiclePosition) -> float | str:
    if vp.HasField("position"):
        return float(vp.position.longitude)

    return ""


def _vehicle_label(vp: gtfs_realtime_pb2.VehiclePosition) -> str:
    return vp.vehicle.label if vp.vehicle.label else ""


def write_vehicle_position_timesteps(input_zip_path: str, matches_zip_path: str, output_zip_path: str, force: bool = False) -> str:
    output_path = Path(output_zip_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        logger.info("Output already exists, skipping: %s", output_path)
        return output_zip_path

    file_pairs = _load_file_pairs(matches_zip_path)
    vehicle_position_files = list(file_pairs.keys())
    logger.info("Loaded %d matched vehicle_positions/trip_updates pair(s)", len(vehicle_position_files))
    previous_by_key: dict[tuple[str, str, str], tuple[str, gtfs_realtime_pb2.VehiclePosition]] = {}
    rows_written = 0
    observations = 0
    transitions = 0
    missing_files = 0
    skipped_files = 0
    missing_gtfs_trips = 0
    missing_stop_updates = 0
    with zipfile.ZipFile(input_zip_path, "r") as zin:
        zip_names = set(zin.namelist())
        trips_by_key = _load_gtfs_trips(zin)
        stops_by_key = _load_gtfs_stops(zin)
        by_trip_sequence, by_trip_stop_id = _load_gtfs_stop_times(zin)
        agency_timezones = _load_agency_timezones(zin)

        with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True) as zout:
            with zout.open(OUTPUT_CSV_NAME, "w") as raw:
                text = TextIOWrapper(raw, encoding="utf-8", newline="")
                writer = csv.writer(text)
                writer.writerow(
                    [
                        "gtfs_agency",
                        "gtfs_service_date",
                        "gtfs_service_day_of_week",
                        "gtfs_service_day_is_weekend",
                        "gtfs_route_id",
                        "gtfs_service_id",
                        "gtfs_trip_id",
                        "gtfs_trip_stops",
                        "gtfs_shape_id",
                        "gtfs_shape_distance",
                        "gtfs_trip_headsign",
                        "gtfs_direction_id",
                        "gtfs_block_id",
                        "gtfsrt_vp_vehicle_id",
                        "gtfsrt_vp_vehicle_label",
                        "gtfsrt_vp_vehicle_bearing",
                        "gtfsrt_vp_vehicle_speed",
                        "travel_time_seconds",
                        "gtfsrt_vp_position_timestamp_previous",
                        "gtfsrt_vp_position_latitude_previous",
                        "gtfsrt_vp_position_longitude_previous",
                        "gtfsrt_vp_position_timestamp_current",
                        "gtfsrt_vp_position_latitude_current",
                        "gtfsrt_vp_position_longitude_current",
                        "gtfs_stop_id",
                        "gtfs_stop_name",
                        "gtfs_stop_sequence",
                        "gtfs_stop_is_first_stop",
                        "gtfs_stop_is_last_stop",
                        "gtfs_stop_relative_stop_position",
                        "gtfs_stop_distance_traveled",
                        "gtfs_stop_relative_shape_progress",
                        "gtfs_stop_scheduled_dwell_time",
                        "gtfs_stop_scheduled_running_time",
                        "gtfs_stop_scheduled_arrival",
                        "gtfs_stop_scheduled_departure",
                        "gtfsrt_tu_prediction_timestamp",
                        "gtfsrt_tu_stop_predicted_arrival",
                        "gtfsrt_tu_stop_predicted_arrival_delay",
                        "gtfsrt_tu_stop_predicted_departure",
                        "gtfsrt_tu_stop_predicted_departure_delay",
                    ]
                )

                for current_file in tqdm(vehicle_position_files, desc="Writing vehicle_position timesteps", unit="file"):
                    parsed_path = _parse_vp_path(current_file)
                    if parsed_path is None:
                        logger.warning("Invalid vehicle_positions path: %s", current_file)
                        skipped_files += 1
                        continue

                    agency, service_date, weekday, is_weekend = parsed_path
                    tu_file = file_pairs[current_file]
                    if current_file not in zip_names:
                        missing_files += 1
                        continue

                    if tu_file not in zip_names:
                        missing_files += 1
                        continue

                    try:
                        current_by_key = _read_vehicle_positions(zin, current_file)
                        prediction_ts, trip_updates_by_trip_id = _read_trip_updates_for_file(zin, tu_file)
                    except Exception:
                        logger.exception("Failed to read GTFS-RT protobuf pair: vp=%s tu=%s", current_file, tu_file)
                        skipped_files += 1
                        continue

                    agency_tz = agency_timezones.get(agency, ZoneInfo("America/Montreal"))
                    for (trip_id, vehicle_id), current_vp in current_by_key.items():

                        observations += 1
                        state_key = (agency, trip_id, vehicle_id)
                        previous = previous_by_key.get(state_key)
                        if previous is None:
                            continue

                        _previous_file, previous_vp = previous
                        previous_ts = _vehicle_timestamp(previous_vp)
                        current_ts = _vehicle_timestamp(current_vp)
                        if previous_ts == "" or current_ts == "":
                            continue

                        gtfs_trip = trips_by_key.get((agency, trip_id))
                        if gtfs_trip is None:
                            missing_gtfs_trips += 1
                            continue

                        stop_time_updates = trip_updates_by_trip_id.get(trip_id, [])
                        if not stop_time_updates:
                            missing_stop_updates += 1
                            continue

                        travel_time_seconds = int(current_ts) - int(previous_ts)
                        for stu in stop_time_updates:
                            stop_sequence = int(stu.stop_sequence) if stu.HasField("stop_sequence") else None
                            stop_id = stu.stop_id or None
                            scheduled = _resolve_scheduled_stop_time(
                                agency=agency,
                                trip_id=trip_id,
                                stop_sequence=stop_sequence,
                                stop_id=stop_id,
                                by_trip_sequence=by_trip_sequence,
                                by_trip_stop_id=by_trip_stop_id,
                            )

                            resolved_stop_id = stop_id or scheduled.get("stop_id", "")
                            stop = stops_by_key.get((agency, resolved_stop_id), {}) if resolved_stop_id else {}
                            scheduled_arrival = scheduled.get("arrival_time", "")
                            scheduled_departure = scheduled.get("departure_time", "")
                            scheduled_arrival_posix = _gtfs_time_to_posix(service_date, scheduled_arrival, agency_tz)
                            scheduled_departure_posix = _gtfs_time_to_posix(service_date, scheduled_departure, agency_tz)
                            predicted_arrival = _stu_arrival_to_posix(stu, scheduled_arrival_posix)
                            predicted_departure = _stu_departure_to_posix(stu, scheduled_departure_posix)
                            arrival_delay = predicted_arrival - scheduled_arrival_posix if predicted_arrival != "" else None
                            departure_delay = predicted_departure - scheduled_departure_posix if predicted_departure != "" else None
                            writer.writerow(
                                [
                                    agency,
                                    service_date,
                                    weekday,
                                    is_weekend,
                                    gtfs_trip["route_id"],
                                    gtfs_trip["service_id"],
                                    gtfs_trip["trip_id"],
                                    scheduled["stops_count"],
                                    gtfs_trip["shape_id"],
                                    scheduled["shape_distance"],
                                    gtfs_trip["trip_headsign"],
                                    gtfs_trip["direction_id"],
                                    gtfs_trip["block_id"],
                                    vehicle_id,
                                    _vehicle_label(current_vp),
                                    current_vp.position.bearing if current_vp.position.bearing else "",
                                    current_vp.position.speed if current_vp.position.speed else "",
                                    travel_time_seconds,
                                    previous_ts,
                                    _vehicle_latitude(previous_vp),
                                    _vehicle_longitude(previous_vp),
                                    current_ts,
                                    _vehicle_latitude(current_vp),
                                    _vehicle_longitude(current_vp),
                                    resolved_stop_id,
                                    stop.get("stop_name", ""),
                                    stop_sequence,
                                    scheduled.get("is_first_stop", None),
                                    scheduled.get("is_last_stop", None),
                                    scheduled.get("stops_progression", None),
                                    scheduled.get("shape_dist_traveled", None),
                                    scheduled.get("shape_progression", None),
                                    scheduled.get("dwell_time_seconds", None),
                                    scheduled.get("running_time_seconds", None),
                                    scheduled_arrival,
                                    scheduled_departure,
                                    prediction_ts,
                                    predicted_arrival,
                                    arrival_delay,
                                    predicted_departure,
                                    departure_delay,
                                ]
                            )

                            rows_written += 1
                        transitions += 1

                    for (trip_id, vehicle_id), current_vp in current_by_key.items():
                        previous_by_key[(agency, trip_id, vehicle_id)] = (
                            current_file,
                            current_vp,
                        )

                text.flush()

    logger.info(
        "Finished vehicle timestep CSV. rows=%d observations=%d transitions=%d "
        "missing_files=%d skipped_files=%d missing_gtfs_trips=%d "
        "missing_stop_updates=%d output=%s",
        rows_written,
        observations,
        transitions,
        missing_files,
        skipped_files,
        missing_gtfs_trips,
        missing_stop_updates,
        output_zip_path,
    )

    return output_zip_path
