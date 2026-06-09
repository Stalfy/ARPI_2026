"""VehiclePosition protobuf parser and parquet-based actuals writer."""

import shutil
import threading
import traceback
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq
import tqdm
from google.transit import gtfs_realtime_pb2

from arpi.discovery.gtfs import GtfsFetchService
from arpi.discovery.rt_parser import GtfsRtParser
from arpi.discovery.shapes import calculate_position_on_shape
from arpi.log import ApplicationLogger
from arpi.models.agency_settings import VehiclePositionMappingStrategy
from arpi.models.gtfs_rt_strategy import StopIdStrategy
from arpi.models.gtfs_segment import ShapeSegment
from arpi.models.gtfs_trip import Trip
from arpi.models.transit import FetchRequest, TimePeriod, TransitAgency

from .constants import VP_SCHEMA
from .helpers import ProtoRow, attr, hf, parse_header

N_DEDUP_PARTITIONS = 256
MAX_ROWS_PER_BATCH = 200_000

TEMP_COMPRESSION = "lz4"
FINAL_COMPRESSION = "zstd"


# ==============================================================================
# Paths
# ==============================================================================


def vp_parquet_path(
    parquet_dir: Path,
    agency: TransitAgency,
    period: TimePeriod,
) -> Path:
    _ = agency
    day = period.start_time.date()
    return parquet_dir / day.strftime("%Y-%m") / f"{day.day:02d}.vehicle-positions.parquet"


# ==============================================================================
# Schema helpers
# ==============================================================================


def _schema_columns(schema: Any) -> list[str]:
    if isinstance(schema, dict):
        return list(schema.keys())

    columns: list[str] = []
    for item in schema:
        if isinstance(item, str):
            columns.append(item)
        else:
            columns.append(item[0])

    return columns


RAW_COLUMNS = _schema_columns(VP_SCHEMA)
PB_FEED_COLUMNS = [f"pb_feed_{column.replace('.', '_')}" for column in RAW_COLUMNS]


# ==============================================================================
# Raw protobuf flattening
# ==============================================================================


def parse_trip_descriptor(vp: gtfs_realtime_pb2.VehiclePosition) -> ProtoRow:
    prefix = "entity.vehicle.trip"

    if not hf(vp, "trip"):
        return {
            f"{prefix}.trip_id": None,
            f"{prefix}.route_id": None,
            f"{prefix}.direction_id": None,
            f"{prefix}.start_time": None,
            f"{prefix}.start_date": None,
            f"{prefix}.schedule_relationship": None,
            f"{prefix}.modified_trip.service_id": None,
            f"{prefix}.modified_trip.trip_modification_id": None,
        }

    trip = vp.trip

    if hf(trip, "modified_trip"):
        mt = trip.modified_trip
        mod_svc = mt.service_id or None
        mod_mod = mt.trip_modification_id or None
    else:
        mod_svc = None
        mod_mod = None

    return {
        f"{prefix}.trip_id": trip.trip_id or None,
        f"{prefix}.route_id": trip.route_id or None,
        f"{prefix}.direction_id": trip.direction_id,
        f"{prefix}.start_time": trip.start_time or None,
        f"{prefix}.start_date": trip.start_date or None,
        f"{prefix}.schedule_relationship": trip.schedule_relationship,
        f"{prefix}.modified_trip.service_id": mod_svc,
        f"{prefix}.modified_trip.trip_modification_id": mod_mod,
    }


def parse_vehicle_descriptor(vp: gtfs_realtime_pb2.VehiclePosition) -> ProtoRow:
    prefix = "entity.vehicle.vehicle"

    if not hf(vp, "vehicle"):
        return {
            f"{prefix}.id": None,
            f"{prefix}.label": None,
            f"{prefix}.license_plate": None,
            f"{prefix}.wheelchair_accessible": None,
        }

    v = vp.vehicle

    return {
        f"{prefix}.id": v.id or None,
        f"{prefix}.label": v.label or None,
        f"{prefix}.license_plate": v.license_plate or None,
        f"{prefix}.wheelchair_accessible": attr(v, "wheelchair_accessible"),
    }


def parse_position(vp: gtfs_realtime_pb2.VehiclePosition) -> ProtoRow:
    prefix = "entity.vehicle.position"

    if not hf(vp, "position"):
        return {
            f"{prefix}.latitude": None,
            f"{prefix}.longitude": None,
            f"{prefix}.bearing": None,
            f"{prefix}.odometer": None,
            f"{prefix}.speed": None,
        }

    p = vp.position

    return {
        f"{prefix}.latitude": p.latitude,
        f"{prefix}.longitude": p.longitude,
        f"{prefix}.bearing": p.bearing or None,
        f"{prefix}.odometer": attr(p, "odometer") or None,
        f"{prefix}.speed": p.speed or None,
    }


def build_raw_vehicle_row(
    file_name: str,
    header_data: ProtoRow,
    entity: gtfs_realtime_pb2.FeedEntity,
    vp: gtfs_realtime_pb2.VehiclePosition,
) -> ProtoRow:
    return {
        "file": file_name,
        **header_data,
        "entity.id": entity.id,
        "entity.is_deleted": entity.is_deleted,
        **parse_trip_descriptor(vp),
        **parse_vehicle_descriptor(vp),
        **parse_position(vp),
        "entity.vehicle.current_stop_sequence": (vp.current_stop_sequence if hf(vp, "current_stop_sequence") else None),
        "entity.vehicle.stop_id": vp.stop_id or None,
        "entity.vehicle.current_status": attr(vp, "current_status"),
        "entity.vehicle.timestamp": vp.timestamp if hf(vp, "timestamp") else None,
        "entity.vehicle.congestion_level": attr(vp, "congestion_level"),
        "entity.vehicle.occupancy_status": attr(vp, "occupancy_status"),
        "entity.vehicle.occupancy_percentage": attr(vp, "occupancy_percentage"),
    }


def prefix_raw_row(raw_row: ProtoRow) -> dict[str, Any]:
    return {f"pb_feed_{column.replace('.', '_')}": raw_row.get(column) for column in RAW_COLUMNS}


def parse_file(path: Path) -> list[ProtoRow]:
    """Compatibility helper: parse one .pb file into raw flattened rows."""
    feed = gtfs_realtime_pb2.FeedMessage()

    with open(path, "rb") as f:
        feed.ParseFromString(f.read())

    header_data = parse_header(feed.header)
    rows: list[ProtoRow] = []

    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue

        rows.append(
            build_raw_vehicle_row(
                file_name=path.name,
                header_data=header_data,
                entity=entity,
                vp=entity.vehicle,
            )
        )

    return rows


# ==============================================================================
# Strategy detection
# ==============================================================================


def _first_vehicle(feed: gtfs_realtime_pb2.FeedMessage) -> gtfs_realtime_pb2.VehiclePosition | None:
    for entity in feed.entity:
        if entity.HasField("vehicle"):
            return entity.vehicle
    return None


def _load_first_vehicle_feed(pb_files: list[Path]) -> gtfs_realtime_pb2.FeedMessage | None:
    for pb_file in pb_files:
        feed = gtfs_realtime_pb2.FeedMessage()

        with open(pb_file, "rb") as f:
            feed.ParseFromString(f.read())

        if _first_vehicle(feed) is not None:
            return feed

    return None


def _analyze_stop_id_strategy(
    feed: gtfs_realtime_pb2.FeedMessage,
    gtfs_stop_times: pl.DataFrame,
    logger: ApplicationLogger,
) -> StopIdStrategy | None:
    vp = _first_vehicle(feed)

    if vp is None:
        logger.wrn("No vehicle entity found. Cannot determine stop_id strategy.")
        return None

    if vp.stop_id:
        logger.dbg("Stop ID strategy: FROM_VEHICLE_POSITION")
        return StopIdStrategy.FROM_VEHICLE_POSITION

    trip_id = vp.trip.trip_id if hf(vp, "trip") else None
    stop_seq = vp.current_stop_sequence if hf(vp, "current_stop_sequence") else None

    if trip_id and stop_seq is not None and not gtfs_stop_times.is_empty():
        match = gtfs_stop_times.filter((pl.col("trip_id") == trip_id) & (pl.col("stop_sequence") == stop_seq))

        if match.height > 0 and "stop_id" in match.columns:
            logger.dbg("Stop ID strategy: FROM_STOPSEQUENCE")
            return StopIdStrategy.FROM_STOPSEQUENCE

    if hf(vp, "position") and trip_id and not gtfs_stop_times.is_empty():
        logger.dbg("Stop ID strategy: FROM_GEOLOCALIZATION")
        return StopIdStrategy.FROM_GEOLOCALIZATION

    logger.wrn("Unable to determine stop_id strategy from feed content and GTFS data.")
    return None


def _get_stop_id_strategy(
    first_feed: gtfs_realtime_pb2.FeedMessage,
    gtfs_stop_times: pl.DataFrame,
    configured_strategy: VehiclePositionMappingStrategy,
    logger: ApplicationLogger,
) -> StopIdStrategy | None:
    match configured_strategy:
        case VehiclePositionMappingStrategy.AUTOMATIC:
            return _analyze_stop_id_strategy(
                feed=first_feed,
                gtfs_stop_times=gtfs_stop_times,
                logger=logger,
            )
        case VehiclePositionMappingStrategy.FROM_VEHICLE_POSITION:
            return StopIdStrategy.FROM_VEHICLE_POSITION
        case VehiclePositionMappingStrategy.FROM_STOPSEQUENCE:
            return StopIdStrategy.FROM_STOPSEQUENCE
        case VehiclePositionMappingStrategy.FROM_GEOLOCALIZATION:
            return StopIdStrategy.FROM_GEOLOCALIZATION
        case _:
            raise ValueError(f"Unknown configured strategy: {configured_strategy}.")


# ==============================================================================
# GTFS lookups
# ==============================================================================


def _load_gtfs_lookups(
    stop_id_strategy: StopIdStrategy,
    gtfs_fetcher: GtfsFetchService,
    parser: GtfsRtParser,
    gtfs_stop_times: pl.DataFrame,
    fr: FetchRequest,
    logger: ApplicationLogger,
) -> tuple[
    dict[tuple[str, int], str] | None,
    dict[str, Trip] | None,
    dict[str, list[ShapeSegment]] | None,
    dict[str, list[tuple[str, float]]] | None,
]:
    if stop_id_strategy == StopIdStrategy.FROM_STOPSEQUENCE:
        stop_ids_lookup = parser.build_stop_id_lookup(gtfs_stop_times)
        logger.dbg(f"{fr.transit_agency}: stop_id lookup built " f"({len(stop_ids_lookup)} entry(ies)).")
        return stop_ids_lookup, None, None, None

    if stop_id_strategy == StopIdStrategy.FROM_VEHICLE_POSITION:
        return None, None, None, None

    logger.inf(f"{fr.transit_agency}: loading stops, trips and shape segments " f"for strategy {stop_id_strategy}.")

    gtfs_stops = gtfs_fetcher.load_stops(fr.transit_agency)
    trips = gtfs_fetcher.load_trips(fr.transit_agency)
    shape_segments = gtfs_fetcher.load_shape_segments(fr.transit_agency)

    trip_stops_lookup = parser.build_trip_stops_lookup(
        gtfs_stop_times,
        gtfs_stops,
    )

    projected = parser.build_projected_trip_stops_lookup(
        trip_stops_lookup,
        trips,
        shape_segments,
    )

    return None, trips, shape_segments, projected


# ==============================================================================
# Stop ID resolution
# ==============================================================================


def _resolve_stop_id_from_geolocalization(
    vp: gtfs_realtime_pb2.VehiclePosition,
    trip_id: str,
    trips: dict[str, Trip] | None,
    shape_segments: dict[str, list[ShapeSegment]] | None,
    projected_trip_stops_lookup: dict[str, list[tuple[str, float]]] | None,
) -> str | None:
    if not hf(vp, "position"):
        return None

    if not trips or not shape_segments or not projected_trip_stops_lookup:
        return None

    trip = trips.get(trip_id)
    if not trip or not trip.shape_id:
        return None

    segments = shape_segments.get(trip.shape_id)
    if not segments:
        return None

    projected_trip_stops = projected_trip_stops_lookup.get(trip_id)
    if not projected_trip_stops:
        return None

    position = vp.position

    vehicle_projection = calculate_position_on_shape(
        position.longitude,
        position.latitude,
        segments,
    )

    if not vehicle_projection:
        return None

    vehicle_distance_from_start, _ = vehicle_projection

    stop_projections: list[tuple[str, float]] = []

    for stop_id, stop_distance_from_start in projected_trip_stops:
        distance_diff = stop_distance_from_start - vehicle_distance_from_start
        stop_projections.append((stop_id, distance_diff))

    status = attr(vp, "current_status")

    if status == gtfs_realtime_pb2.VehiclePosition.STOPPED_AT:
        closest_stop = min(stop_projections, key=lambda x: abs(x[1]))
        return closest_stop[0]

    for stop_id, distance_diff in stop_projections:
        if abs(distance_diff) <= 10.0:
            return stop_id

    ahead_stops = [s for s in stop_projections if s[1] > 0]
    if ahead_stops:
        next_stop = min(ahead_stops, key=lambda x: x[1])
        return next_stop[0]

    return None


def resolve_stop_id(
    vp: gtfs_realtime_pb2.VehiclePosition,
    strategy: StopIdStrategy,
    stop_ids_lookup: dict[tuple[str, int], str] | None,
    trips: dict[str, Trip] | None,
    shape_segments: dict[str, list[ShapeSegment]] | None,
    projected_trip_stops_lookup: dict[str, list[tuple[str, float]]] | None,
) -> str | None:
    trip_id = vp.trip.trip_id if hf(vp, "trip") else None

    if strategy == StopIdStrategy.FROM_VEHICLE_POSITION:
        return vp.stop_id or None

    if strategy == StopIdStrategy.FROM_STOPSEQUENCE:
        if not trip_id or stop_ids_lookup is None:
            return None

        if not hf(vp, "current_stop_sequence"):
            return None

        return stop_ids_lookup.get((trip_id, int(vp.current_stop_sequence)))

    if strategy == StopIdStrategy.FROM_GEOLOCALIZATION:
        if not trip_id:
            return None

        return _resolve_stop_id_from_geolocalization(
            vp=vp,
            trip_id=trip_id,
            trips=trips,
            shape_segments=shape_segments,
            projected_trip_stops_lookup=projected_trip_stops_lookup,
        )

    return None


# ==============================================================================
# Feed parsing
# ==============================================================================


def parse_feed_to_actual_frame(
    feed: gtfs_realtime_pb2.FeedMessage,
    file_name: str,
    file_batch_id: int,
    stop_id_strategy: StopIdStrategy,
    stop_ids_lookup: dict[tuple[str, int], str] | None,
    trips: dict[str, Trip] | None,
    shape_segments: dict[str, list[ShapeSegment]] | None,
    projected_trip_stops_lookup: dict[str, list[tuple[str, float]]] | None,
) -> pl.DataFrame:
    header_timestamp = feed.header.timestamp if hf(feed.header, "timestamp") else None
    header_data = parse_header(feed.header)

    candidates: list[dict[str, Any]] = []
    last_timestamp_by_trip_vehicle_stop: dict[tuple[str, str | None, str], int] = {}
    latest_raw_by_trip_vehicle: dict[tuple[str, str | None], ProtoRow] = {}
    latest_raw_ts_by_trip_vehicle: dict[tuple[str, str | None], int] = {}

    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue

        vp = entity.vehicle

        trip_id = vp.trip.trip_id if hf(vp, "trip") else None
        if not trip_id:
            continue

        vehicle_id = vp.vehicle.id if hf(vp, "vehicle") else None

        timestamp = vp.timestamp if hf(vp, "timestamp") else header_timestamp
        if timestamp is None:
            continue

        ts = int(timestamp)

        stop_id = resolve_stop_id(
            vp=vp,
            strategy=stop_id_strategy,
            stop_ids_lookup=stop_ids_lookup,
            trips=trips,
            shape_segments=shape_segments,
            projected_trip_stops_lookup=projected_trip_stops_lookup,
        )

        raw_row = build_raw_vehicle_row(
            file_name=file_name,
            header_data=header_data,
            entity=entity,
            vp=vp,
        )

        trip_vehicle_key = (trip_id, vehicle_id)
        previous_latest = latest_raw_ts_by_trip_vehicle.get(trip_vehicle_key)

        if previous_latest is None or ts > previous_latest:
            latest_raw_ts_by_trip_vehicle[trip_vehicle_key] = ts
            latest_raw_by_trip_vehicle[trip_vehicle_key] = raw_row

        if not stop_id:
            continue

        stop_key = (trip_id, vehicle_id, stop_id)
        previous = last_timestamp_by_trip_vehicle_stop.get(stop_key)

        if previous is None or ts > previous:
            last_timestamp_by_trip_vehicle_stop[stop_key] = ts

        candidates.append(
            {
                "trip_id": trip_id,
                "vehicle_id": vehicle_id,
                "stop_id": stop_id,
                "actual_arrival_epoch": ts,
                "_trip_vehicle_key": trip_vehicle_key,
            }
        )

    if not candidates:
        return pl.DataFrame()

    rows: list[dict[str, Any]] = []
    row_in_file = 0

    for candidate in candidates:
        stop_key = (
            candidate["trip_id"],
            candidate["vehicle_id"],
            candidate["stop_id"],
        )

        if candidate["actual_arrival_epoch"] != last_timestamp_by_trip_vehicle_stop.get(stop_key):
            continue

        raw_row = latest_raw_by_trip_vehicle.get(candidate["_trip_vehicle_key"])
        if raw_row is None:
            continue

        row: dict[str, Any] = {
            "trip_id": candidate["trip_id"],
            "stop_id": candidate["stop_id"],
            "actual_arrival_epoch": candidate["actual_arrival_epoch"],
            "vehicle_id": candidate["vehicle_id"],
            "_batch_id": file_batch_id,
            "_row_in_batch": row_in_file,
            "_row_id": (file_batch_id << 32) + row_in_file,
        }

        row.update(prefix_raw_row(raw_row))

        rows.append(row)
        row_in_file += 1

    if not rows:
        return pl.DataFrame()

    return pl.DataFrame(rows).with_columns(
        pl.concat_str(
            [
                "trip_id",
                "stop_id",
            ],
            separator="|",
        )
        .hash(seed=0)
        .alias("key_hash")
    )


# ==============================================================================
# Temporary partitioned writes
# ==============================================================================


def write_partitioned_narrow_wide_batch(
    df: pl.DataFrame,
    narrow_dir: Path,
    wide_dir: Path,
    write_batch_id: int,
    n_partitions: int,
) -> None:
    df = df.with_columns((pl.col("key_hash") % n_partitions).cast(pl.UInt16).alias("_part"))

    narrow_cols = [
        "_row_id",
        "key_hash",
        "trip_id",
        "stop_id",
        "actual_arrival_epoch",
        "vehicle_id",
        "_batch_id",
        "_row_in_batch",
        "_part",
    ]

    wide_cols = [
        "_row_id",
        "_part",
        *PB_FEED_COLUMNS,
    ]

    narrow_df = df.select(narrow_cols)
    wide_df = df.select(wide_cols)

    for part_key, part_df in narrow_df.partition_by(
        "_part",
        as_dict=True,
        maintain_order=False,
    ).items():
        part = int(part_key[0] if isinstance(part_key, tuple) else part_key)
        part_dir = narrow_dir / f"part={part:03d}"
        part_dir.mkdir(parents=True, exist_ok=True)

        part_df.write_parquet(
            part_dir / f"batch_{write_batch_id:06d}.parquet",
            compression=TEMP_COMPRESSION,
        )

    for part_key, part_df in wide_df.partition_by(
        "_part",
        as_dict=True,
        maintain_order=False,
    ).items():
        part = int(part_key[0] if isinstance(part_key, tuple) else part_key)
        part_dir = wide_dir / f"part={part:03d}"
        part_dir.mkdir(parents=True, exist_ok=True)

        part_df.write_parquet(
            part_dir / f"batch_{write_batch_id:06d}.parquet",
            compression=TEMP_COMPRESSION,
        )


def flush_pending_frames(
    pending_frames: list[pl.DataFrame],
    narrow_dir: Path,
    wide_dir: Path,
    write_batch_id: int,
    n_partitions: int,
) -> None:
    if not pending_frames:
        return

    combined_df = pl.concat(
        pending_frames,
        how="vertical_relaxed",
    )

    write_partitioned_narrow_wide_batch(
        df=combined_df,
        narrow_dir=narrow_dir,
        wide_dir=wide_dir,
        write_batch_id=write_batch_id,
        n_partitions=n_partitions,
    )

    pending_frames.clear()


# ==============================================================================
# Deduplication
# ==============================================================================


def deduplicate_partitions(
    narrow_dir: Path,
    wide_dir: Path,
    dedup_dir: Path,
    transit_agency: TransitAgency,
    cancel_event: threading.Event | None,
    log,
) -> list[Path]:
    deduped_files: list[Path] = []

    part_dirs = sorted(narrow_dir.glob("part=*"))

    bar = tqdm.tqdm(
        part_dirs,
        desc="deduplicating",
        unit="partition",
        leave=False,
    )

    for narrow_part_dir in bar:
        if cancel_event and cancel_event.is_set():
            log("Cancellation signal received during deduplication. Stopping.")
            return deduped_files

        part_name = narrow_part_dir.name.split("=", maxsplit=1)[1]
        bar.set_postfix_str(part_name)

        narrow_files = sorted(narrow_part_dir.glob("*.parquet"))
        if not narrow_files:
            continue

        wide_part_dir = wide_dir / narrow_part_dir.name
        wide_files = sorted(wide_part_dir.glob("*.parquet")) if wide_part_dir.exists() else []

        if not wide_files:
            continue

        survivors = (
            pl.scan_parquet(narrow_files)
            .sort(["_batch_id", "_row_in_batch"])
            .unique(
                subset=[
                    "key_hash",
                    "trip_id",
                    "stop_id",
                ],
                keep="first",
                maintain_order=True,
            )
            .collect(streaming=True)
        )

        survivor_ids = survivors.select(["_row_id"])

        wide_survivors = (
            pl.scan_parquet(wide_files)
            .join(
                survivor_ids.lazy(),
                on="_row_id",
                how="inner",
            )
            .collect(streaming=True)
        )

        final_part = (
            survivors.join(
                wide_survivors,
                on="_row_id",
                how="left",
            )
            .with_columns(
                [
                    pl.from_epoch(pl.col("actual_arrival_epoch"), time_unit="s")
                    .dt.replace_time_zone("UTC")
                    .dt.convert_time_zone("America/Toronto")
                    .dt.replace_time_zone(None)
                    .alias("actual_arrival"),
                    pl.lit(transit_agency).alias("transit_agency"),
                ]
            )
            .drop(
                [
                    "_row_id",
                    "key_hash",
                    "_batch_id",
                    "_row_in_batch",
                    "_part",
                    "_part_right",
                    "actual_arrival_epoch",
                ],
                strict=False,
            )
        )

        ordered_cols = [
            "trip_id",
            "stop_id",
            "actual_arrival",
            "vehicle_id",
            *PB_FEED_COLUMNS,
            "transit_agency",
        ]

        final_part = final_part.select([col for col in ordered_cols if col in final_part.columns])

        part_output = dedup_dir / f"deduped_part_{part_name}.parquet"

        final_part.write_parquet(
            part_output,
            compression=TEMP_COMPRESSION,
        )

        deduped_files.append(part_output)

    return deduped_files


# ==============================================================================
# Public API
# ==============================================================================


def parse_directory(
    directory: Path,
    output_file: Path,
    fr: FetchRequest,
    gtfs_fetcher: GtfsFetchService,
    parser: GtfsRtParser,
    configured_strategy: VehiclePositionMappingStrategy = VehiclePositionMappingStrategy.AUTOMATIC,
    cancel_event: threading.Event | None = None,
    logger: ApplicationLogger | None = None,
    n_partitions: int = N_DEDUP_PARTITIONS,
    max_rows_per_batch: int = MAX_ROWS_PER_BATCH,
) -> None:
    if logger is None:
        logger = ApplicationLogger("vehicle_positions")

    log = logger.inf
    err = logger.err

    pb_files = sorted(directory.glob("*.pb"))
    if not pb_files:
        log(f"No .pb files found in {directory}")
        return

    total_mb = sum(path.stat().st_size for path in pb_files) / 1e6
    log(f"{len(pb_files)} VehiclePosition .pb files ({total_mb:.1f} MB total).")

    first_feed = _load_first_vehicle_feed(pb_files)
    if first_feed is None:
        log("No vehicle position entities found.")
        return

    gtfs_stop_times = gtfs_fetcher.load_stop_times_df(fr.transit_agency)

    stop_id_strategy = _get_stop_id_strategy(
        first_feed=first_feed,
        gtfs_stop_times=gtfs_stop_times,
        configured_strategy=configured_strategy,
        logger=logger,
    )

    if stop_id_strategy is None:
        logger.wrn(f"{fr.transit_agency}: unable to determine stop_id strategy.")
        return

    stop_ids_lookup, trips, shape_segments, projected = _load_gtfs_lookups(
        stop_id_strategy=stop_id_strategy,
        gtfs_fetcher=gtfs_fetcher,
        parser=parser,
        gtfs_stop_times=gtfs_stop_times,
        fr=fr,
        logger=logger,
    )

    log(f"{fr.transit_agency}, {fr.time_period.start_time.date()}: " f"using stop_id strategy {stop_id_strategy.value}.")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    tmp_root = output_file.parent / f".{output_file.stem}_tmp"
    narrow_dir = tmp_root / "narrow"
    wide_dir = tmp_root / "wide"
    dedup_dir = tmp_root / "deduped"

    narrow_dir.mkdir(parents=True, exist_ok=True)
    wide_dir.mkdir(parents=True, exist_ok=True)
    dedup_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    skipped = 0
    write_batch_id = 0
    pending_rows = 0
    pending_frames: list[pl.DataFrame] = []

    try:
        bar = tqdm.tqdm(
            enumerate(pb_files),
            total=len(pb_files),
            desc="vehicle_positions",
            unit="file",
            leave=False,
        )

        for file_batch_id, pb_file in bar:
            bar.set_postfix_str(pb_file.name)

            if cancel_event and cancel_event.is_set():
                log("Cancellation signal received. Stopping.")
                break

            try:
                feed = gtfs_realtime_pb2.FeedMessage()

                with open(pb_file, "rb") as f:
                    feed.ParseFromString(f.read())

                df = parse_feed_to_actual_frame(
                    feed=feed,
                    file_name=pb_file.name,
                    file_batch_id=file_batch_id,
                    stop_id_strategy=stop_id_strategy,
                    stop_ids_lookup=stop_ids_lookup,
                    trips=trips,
                    shape_segments=shape_segments,
                    projected_trip_stops_lookup=projected,
                )

                if df.is_empty():
                    continue

                pending_frames.append(df)
                pending_rows += df.height
                total_rows += df.height

                if pending_rows >= max_rows_per_batch:
                    write_batch_id += 1

                    flush_pending_frames(
                        pending_frames=pending_frames,
                        narrow_dir=narrow_dir,
                        wide_dir=wide_dir,
                        write_batch_id=write_batch_id,
                        n_partitions=n_partitions,
                    )

                    pending_rows = 0

            except Exception:
                skipped += 1
                err(f"SKIP {pb_file.name}:\n{traceback.format_exc()}")

        if pending_frames:
            write_batch_id += 1

            flush_pending_frames(
                pending_frames=pending_frames,
                narrow_dir=narrow_dir,
                wide_dir=wide_dir,
                write_batch_id=write_batch_id,
                n_partitions=n_partitions,
            )

        if not total_rows:
            log("No vehicle position data parsed.")
            return

        deduped_files = deduplicate_partitions(
            narrow_dir=narrow_dir,
            wide_dir=wide_dir,
            dedup_dir=dedup_dir,
            transit_agency=fr.transit_agency,
            cancel_event=cancel_event,
            log=log,
        )

        if not deduped_files:
            log("No vehicle position data after deduplication.")
            return

        (
            pl.scan_parquet(deduped_files).sink_parquet(
                output_file,
                compression=FINAL_COMPRESSION,
            )
        )

        n_unique = pq.ParquetFile(output_file).metadata.num_rows
        skip_msg = f", {skipped} file(s) skipped" if skipped else ""

        log(f"{total_rows:,} vehicle positions -> " f"{n_unique:,} unique actuals" f"{skip_msg} -> {output_file}")

    except Exception:
        err(f"\nError:\n{traceback.format_exc()}")
        raise

    finally:
        try:
            if tmp_root.exists():
                shutil.rmtree(tmp_root)
        except Exception:
            err(f"Failed to remove temporary directory {tmp_root}:\n" f"{traceback.format_exc()}")
