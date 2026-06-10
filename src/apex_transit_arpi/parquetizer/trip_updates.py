"""TripUpdate protobuf parser and parquet-based predictions writer.

Optimized implementation:
- parses each .pb file once;
- does not use MessageToDict;
- does not use DuckDB;
- does not use multithreading;
- batches multiple .pb files up to MAX_ROWS_PER_BATCH rows;
- keeps all pb_feed_* columns;
- deduplicates narrow rows only;
- joins wide pb_feed_* columns back only for survivors;
- uses 256 hash partitions to bound memory;
- keeps pred_time / pred_arrival as epoch integers until final output.
"""

import re
import shutil
import threading
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
import pyarrow.parquet as pq
import tqdm
from google.transit import gtfs_realtime_pb2

from apex_transit_arpi.discovery.gtfs import GtfsFetchService
from apex_transit_arpi.log import ApplicationLogger
from apex_transit_arpi.models.gtfs_stop_time import StopTime
from apex_transit_arpi.models.transit import DelayUsingAgencies, FetchRequest, TimePeriod, TransitAgency

from .constants import TU_SCHEMA
from .helpers import ProtoRow, attr, hf, parse_header, str_or_ts

N_DEDUP_PARTITIONS = 256
MAX_ROWS_PER_BATCH = 200_000
TIMEZONE_NAME = "America/Toronto"

TEMP_COMPRESSION = "lz4"
FINAL_COMPRESSION = "zstd"


# ==============================================================================
# Paths
# ==============================================================================


def tu_parquet_path(
    parquet_dir: Path,
    agency: TransitAgency,
    period: TimePeriod,
) -> Path:
    _ = agency
    day = period.start_time.date()
    ym = day.strftime("%Y-%m")
    return parquet_dir / ym / f"{day.day:02d}.trip-updates.parquet"


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


RAW_COLUMNS = _schema_columns(TU_SCHEMA)
PB_FEED_COLUMNS = [f"pb_feed_{column.replace('.', '_')}" for column in RAW_COLUMNS]


# ==============================================================================
# Time helpers
# ==============================================================================


def hms_to_seconds(hms: str) -> int:
    h, m, s = map(int, hms.split(":"))
    return h * 3600 + m * 60 + s


def compute_service_midnight(pred_time: int, trip_id: str) -> datetime:
    base_dt_utc = datetime.fromtimestamp(int(pred_time), tz=timezone.utc)
    base_dt_local = base_dt_utc.astimezone(ZoneInfo(TIMEZONE_NAME))

    is_service_day_carryover = False

    if base_dt_local.hour < 3:
        time_match = re.search(r"_(\d{2}):(\d{2})$", trip_id)
        if time_match and int(time_match.group(1)) >= 24:
            is_service_day_carryover = True

    service_midnight = datetime(
        base_dt_local.year,
        base_dt_local.month,
        base_dt_local.day,
    )

    if is_service_day_carryover:
        service_midnight -= timedelta(days=1)

    return service_midnight


def compute_pred_arrival_from_delay(
    stops_lookup: dict,
    delay: Any,
    trip_id: str,
    stop_id: str,
    service_midnight: datetime,
) -> int | None:
    if delay is None or not stops_lookup:
        return None

    scheduled: StopTime | None = stops_lookup.get((trip_id, stop_id))
    if not scheduled:
        return None

    scheduled_sec = hms_to_seconds(scheduled.arrival_time)
    scheduled_dt = service_midnight + timedelta(seconds=scheduled_sec)
    pred_arrival_dt = scheduled_dt + timedelta(seconds=int(delay))

    return int(pred_arrival_dt.timestamp())


# ==============================================================================
# Raw protobuf flattening
# ==============================================================================


def parse_trip_descriptor(trip: gtfs_realtime_pb2.TripDescriptor) -> ProtoRow:
    if hf(trip, "modified_trip"):
        mt = trip.modified_trip
        mod_svc = mt.service_id or None
        mod_mod = mt.trip_modification_id or None
    else:
        mod_svc = None
        mod_mod = None

    return {
        "entity.trip_update.trip.trip_id": trip.trip_id or None,
        "entity.trip_update.trip.route_id": trip.route_id or None,
        "entity.trip_update.trip.direction_id": trip.direction_id,
        "entity.trip_update.trip.start_time": trip.start_time or None,
        "entity.trip_update.trip.start_date": trip.start_date or None,
        "entity.trip_update.trip.schedule_relationship": trip.schedule_relationship,
        "entity.trip_update.trip.modified_trip.service_id": mod_svc,
        "entity.trip_update.trip.modified_trip.trip_modification_id": mod_mod,
    }


def parse_vehicle(tu: gtfs_realtime_pb2.TripUpdate) -> ProtoRow:
    if not hf(tu, "vehicle"):
        return {
            "entity.trip_update.vehicle.id": None,
            "entity.trip_update.vehicle.label": None,
            "entity.trip_update.vehicle.license_plate": None,
            "entity.trip_update.vehicle.wheelchair_accessible": None,
        }

    v = tu.vehicle

    return {
        "entity.trip_update.vehicle.id": v.id or None,
        "entity.trip_update.vehicle.label": v.label or None,
        "entity.trip_update.vehicle.license_plate": v.license_plate or None,
        "entity.trip_update.vehicle.wheelchair_accessible": attr(v, "wheelchair_accessible"),
    }


def parse_stop_time_event(
    stu: gtfs_realtime_pb2.TripUpdate.StopTimeUpdate,
    event: str,
) -> ProtoRow:
    prefix = f"entity.trip_update.stop_time_update.{event}"

    if not hf(stu, event):
        return {
            f"{prefix}.delay": None,
            f"{prefix}.time": None,
            f"{prefix}.scheduled_time": None,
            f"{prefix}.uncertainty": None,
        }

    e = getattr(stu, event)

    return {
        f"{prefix}.delay": e.delay if hf(e, "delay") else None,
        f"{prefix}.time": e.time if hf(e, "time") else None,
        f"{prefix}.scheduled_time": attr(e, "scheduled_time") or None,
        f"{prefix}.uncertainty": e.uncertainty if hf(e, "uncertainty") else None,
    }


def parse_stop_time_properties(
    stu: gtfs_realtime_pb2.TripUpdate.StopTimeUpdate,
) -> ProtoRow:
    prefix = "entity.trip_update.stop_time_update.stop_time_properties"

    if not hf(stu, "stop_time_properties"):
        return {
            f"{prefix}.assigned_stop_id": None,
            f"{prefix}.stop_headsign": None,
            f"{prefix}.drop_off_type": None,
            f"{prefix}.pickup_type": None,
        }

    stp = stu.stop_time_properties

    return {
        f"{prefix}.assigned_stop_id": attr(stp, "assigned_stop_id") or None,
        f"{prefix}.stop_headsign": str_or_ts(stp, "stop_headsign"),
        f"{prefix}.drop_off_type": attr(stp, "drop_off_type"),
        f"{prefix}.pickup_type": attr(stp, "pickup_type"),
    }


def parse_trip_properties(tu: gtfs_realtime_pb2.TripUpdate) -> ProtoRow:
    prefix = "entity.trip_update.trip_properties"

    if not hf(tu, "trip_properties"):
        return {
            f"{prefix}.trip_id": None,
            f"{prefix}.start_date": None,
            f"{prefix}.start_time": None,
            f"{prefix}.trip_headsign": None,
            f"{prefix}.trip_short_name": None,
            f"{prefix}.shape_id": None,
        }

    tp = tu.trip_properties

    return {
        f"{prefix}.trip_id": tp.trip_id or None,
        f"{prefix}.start_date": tp.start_date or None,
        f"{prefix}.start_time": tp.start_time or None,
        f"{prefix}.trip_headsign": str_or_ts(tp, "trip_headsign"),
        f"{prefix}.trip_short_name": str_or_ts(tp, "trip_short_name"),
        f"{prefix}.shape_id": attr(tp, "shape_id") or None,
    }


def build_raw_stop_update_row(
    file_name: str,
    header_data: ProtoRow,
    entity: gtfs_realtime_pb2.FeedEntity,
    tu: gtfs_realtime_pb2.TripUpdate,
    stu: gtfs_realtime_pb2.TripUpdate.StopTimeUpdate,
) -> ProtoRow:
    return {
        "file": file_name,
        **header_data,
        "entity.id": entity.id,
        "entity.is_deleted": entity.is_deleted,
        **parse_trip_descriptor(tu.trip),
        **parse_vehicle(tu),
        "entity.trip_update.stop_time_update.stop_sequence": (stu.stop_sequence if hf(stu, "stop_sequence") else None),
        "entity.trip_update.stop_time_update.stop_id": stu.stop_id or None,
        **parse_stop_time_event(stu, "arrival"),
        **parse_stop_time_event(stu, "departure"),
        "entity.trip_update.stop_time_update.departure_occupancy_status": attr(
            stu,
            "departure_occupancy_status",
        ),
        "entity.trip_update.stop_time_update.schedule_relationship": (stu.schedule_relationship if hf(stu, "schedule_relationship") else None),
        **parse_stop_time_properties(stu),
        "entity.trip_update.timestamp": tu.timestamp if hf(tu, "timestamp") else None,
        "entity.trip_update.delay": tu.delay if hf(tu, "delay") else None,
        **parse_trip_properties(tu),
    }


def prefix_raw_row(raw_row: ProtoRow) -> dict[str, Any]:
    return {f"pb_feed_{column.replace('.', '_')}": raw_row.get(column) for column in RAW_COLUMNS}


def parse_file(path: Path) -> list[ProtoRow]:
    """Compatibility helper: parse one .pb file into raw flattened rows (one per stop_time_update)."""
    feed = gtfs_realtime_pb2.FeedMessage()

    with open(path, "rb") as f:
        feed.ParseFromString(f.read())

    header_data = parse_header(feed.header)
    rows: list[ProtoRow] = []

    for entity in feed.entity:
        if not entity.HasField("trip_update"):
            continue
        tu = entity.trip_update
        for stu in tu.stop_time_update:
            rows.append(
                build_raw_stop_update_row(
                    file_name=path.name,
                    header_data=header_data,
                    entity=entity,
                    tu=tu,
                    stu=stu,
                )
            )

    return rows


# ==============================================================================
# Feed parsing
# ==============================================================================


def parse_feed_to_prediction_frame(
    feed: gtfs_realtime_pb2.FeedMessage,
    file_name: str,
    file_batch_id: int,
    stops_lookup: dict | None,
) -> pl.DataFrame:
    """Parse one FeedMessage into final-shape rows plus technical columns.

    This function intentionally builds normalized prediction columns and pb_feed_*
    columns in the same pass over stop_time_update. It avoids:
    - MessageToDict;
    - parsing the same protobuf twice;
    - raw_df -> unique -> join.
    """
    rows: list[dict[str, Any]] = []
    use_delay = bool(stops_lookup)

    header_timestamp = feed.header.timestamp if hf(feed.header, "timestamp") else None
    header_data = parse_header(feed.header)

    row_in_file = 0

    for entity in feed.entity:
        if not entity.HasField("trip_update"):
            continue

        tu = entity.trip_update
        trip = tu.trip

        trip_id = trip.trip_id or None
        route_id = trip.route_id or None

        if not trip_id:
            continue

        pred_time = tu.timestamp if hf(tu, "timestamp") else header_timestamp
        if pred_time is None:
            continue

        service_midnight = compute_service_midnight(
            pred_time=int(pred_time),
            trip_id=trip_id,
        )

        for stu in tu.stop_time_update:
            stop_id = stu.stop_id or None
            if not stop_id:
                continue

            if not hf(stu, "arrival"):
                continue

            arrival = stu.arrival

            if use_delay:
                delay = arrival.delay if hf(arrival, "delay") else None
                pred_arrival = compute_pred_arrival_from_delay(
                    stops_lookup=stops_lookup,
                    delay=delay,
                    trip_id=trip_id,
                    stop_id=stop_id,
                    service_midnight=service_midnight,
                )
            else:
                arrival_time = arrival.time if hf(arrival, "time") else None
                pred_arrival = int(arrival_time) if arrival_time is not None else None

            if pred_arrival is None:
                continue

            raw_row = build_raw_stop_update_row(
                file_name=file_name,
                header_data=header_data,
                entity=entity,
                tu=tu,
                stu=stu,
            )

            row: dict[str, Any] = {
                "trip_id": trip_id,
                "stop_id": stop_id,
                "pred_time_epoch": int(pred_time),
                "pred_arrival_epoch": int(pred_arrival),
                "route_id": route_id,
                "file": file_name,
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
        [
            pl.col("route_id").cast(pl.Utf8),
            pl.concat_str(
                [
                    "trip_id",
                    "stop_id",
                    "pred_time_epoch",
                    "pred_arrival_epoch",
                ],
                separator="|",
            )
            .hash(seed=0)
            .alias("key_hash"),
        ]
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
        "pred_time_epoch",
        "pred_arrival_epoch",
        "route_id",
        "file",
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
                    "pred_time_epoch",
                    "pred_arrival_epoch",
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
                    pl.from_epoch(pl.col("pred_time_epoch"), time_unit="s")
                    .dt.replace_time_zone("UTC")
                    .dt.convert_time_zone(TIMEZONE_NAME)
                    .dt.replace_time_zone(None)
                    .alias("pred_time"),
                    pl.from_epoch(pl.col("pred_arrival_epoch"), time_unit="s")
                    .dt.replace_time_zone("UTC")
                    .dt.convert_time_zone(TIMEZONE_NAME)
                    .dt.replace_time_zone(None)
                    .alias("pred_arrival"),
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
                    "pred_time_epoch",
                    "pred_arrival_epoch",
                ],
                strict=False,
            )
        )

        ordered_cols = [
            "trip_id",
            "stop_id",
            "pred_time",
            "pred_arrival",
            "route_id",
            "file",
            *PB_FEED_COLUMNS,
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
    cancel_event: threading.Event | None = None,
    logger: ApplicationLogger | None = None,
    n_partitions: int = N_DEDUP_PARTITIONS,
    max_rows_per_batch: int = MAX_ROWS_PER_BATCH,
) -> None:
    """Parse all TripUpdate .pb files and write a deduplicated predictions parquet."""
    log = logger.inf if logger else print
    err = logger.err if logger else print

    agency = fr.transit_agency

    stops_mappings = None
    if agency in DelayUsingAgencies:
        log(f"{agency}: loading stop_times mappings for delay-based TripUpdates.")
        stops_mappings = gtfs_fetcher.load_stop_times_mappings(
            transit_agency=agency,
        )

        if not stops_mappings:
            raise ValueError(f"Failed to load stop_times.txt for {agency}.")

        log(f"{agency}: loaded {len(stops_mappings)} stop_times mapping entries.")

    pb_files = sorted(directory.glob("*.pb"))
    if not pb_files:
        log(f"No .pb files in {directory}")
        return

    total_mb = sum(path.stat().st_size for path in pb_files) / 1e6
    log(f"{len(pb_files)} TripUpdate .pb files ({total_mb:.1f} MB total).")

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

    pending_frames: list[pl.DataFrame] = []
    pending_rows = 0

    try:
        bar = tqdm.tqdm(
            enumerate(pb_files),
            total=len(pb_files),
            desc="trip_updates",
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

                df = parse_feed_to_prediction_frame(
                    feed=feed,
                    file_name=pb_file.name,
                    file_batch_id=file_batch_id,
                    stops_lookup=stops_mappings,
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

            pending_rows = 0

        if not total_rows:
            log("No predictions parsed.")
            return

        deduped_files = deduplicate_partitions(
            narrow_dir=narrow_dir,
            wide_dir=wide_dir,
            dedup_dir=dedup_dir,
            cancel_event=cancel_event,
            log=log,
        )

        if not deduped_files:
            log("No predictions parsed after deduplication.")
            return

        (
            pl.scan_parquet(deduped_files).sink_parquet(
                output_file,
                compression=FINAL_COMPRESSION,
            )
        )

        n_unique = pq.ParquetFile(output_file).metadata.num_rows
        skip_msg = f", {skipped} file(s) skipped" if skipped else ""

        log(f"{total_rows:,} predictions -> " f"{n_unique:,} unique" f"{skip_msg} -> {output_file}")

    except Exception:
        err(f"\nError:\n{traceback.format_exc()}")
        raise

    finally:
        try:
            if tmp_root.exists():
                shutil.rmtree(tmp_root)
        except Exception:
            err(f"Failed to remove temporary directory {tmp_root}:\n" f"{traceback.format_exc()}")
