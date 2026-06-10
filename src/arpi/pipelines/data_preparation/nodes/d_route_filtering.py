# nodes.py

from __future__ import annotations

import csv
import logging
import shutil
import zipfile
from io import BytesIO, TextIOWrapper
from pathlib import Path, PurePosixPath
from typing import Any

import polars as pl
from google.transit import gtfs_realtime_pb2
from tqdm import tqdm

logger = logging.getLogger(__name__)


GTFS_ROOT = "GTFS"
GTFS_RT_ROOT = "GTFS-RT"
MATCHES_CSV_NAME = "feed_file_sync.csv"


def _as_set(values: str | list[str]) -> set[str]:
    if isinstance(values, str):
        return {values}
    return {str(value) for value in values}


def _read_gtfs_csv(zf: zipfile.ZipFile, name: str) -> pl.DataFrame:
    with zf.open(name) as f:
        return pl.read_csv(f, infer_schema_length=0)


def _df_to_csv_bytes(df: pl.DataFrame) -> bytes:
    buffer = BytesIO()
    df.write_csv(buffer)
    return buffer.getvalue()


def _write_df_to_zip(zout: zipfile.ZipFile, path: str, df: pl.DataFrame) -> None:
    if not df.is_empty():
        zout.writestr(path, _df_to_csv_bytes(df))


def _copy_zip_member(zin: zipfile.ZipFile, zout: zipfile.ZipFile, source_path: str, target_path: str) -> None:
    info = zin.getinfo(source_path)
    out_info = zipfile.ZipInfo(filename=target_path, date_time=info.date_time)
    out_info.compress_type = info.compress_type
    out_info.external_attr = info.external_attr
    out_info.comment = info.comment
    out_info.extra = info.extra
    out_info.create_system = info.create_system

    with zin.open(info, "r") as src, zout.open(out_info, "w") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def _gtfs_path(agency: str, filename: str) -> str:
    return f"{GTFS_ROOT}/{agency}/{filename}"


def _extract_static_gtfs_for_routes(zin: zipfile.ZipFile, zout: zipfile.ZipFile, agency: str, route_ids: set[str]) -> dict[str, set[str]]:
    """
    Reduces GTFS/AGENCY/*.txt by route_id and writes the reduced static GTFS
    into the output ZIP.

    Returns IDs used to reduce GTFS-RT.
    """
    names = set(zin.namelist())
    trips_path = _gtfs_path(agency, "trips.txt")

    if trips_path not in names:
        raise FileNotFoundError(f"Missing required GTFS file: {trips_path}")

    trips = _read_gtfs_csv(zin, trips_path).with_columns(
        pl.col("route_id").cast(pl.String),
        pl.col("trip_id").cast(pl.String),
    )

    route_trips = trips.filter(pl.col("route_id").is_in(route_ids))
    if route_trips.is_empty():
        raise ValueError(f"No trips found for agency={agency!r}, route_ids={sorted(route_ids)!r}")

    trip_ids = set(route_trips["trip_id"].cast(pl.String).to_list())
    service_ids: set[str] = set()
    if "service_id" in route_trips.columns:
        service_ids = set(route_trips["service_id"].cast(pl.String).to_list())

    shape_ids: set[str] = set()
    if "shape_id" in route_trips.columns:
        shape_ids = set(route_trips["shape_id"].cast(pl.String).to_list())

    _write_df_to_zip(zout, trips_path, route_trips)
    routes_path = _gtfs_path(agency, "routes.txt")
    if routes_path in names:
        routes = _read_gtfs_csv(zin, routes_path).with_columns(pl.col("route_id").cast(pl.String))
        _write_df_to_zip(zout, routes_path, routes.filter(pl.col("route_id").is_in(route_ids)))

    stop_ids: set[str] = set()

    stop_times_path = _gtfs_path(agency, "stop_times.txt")
    if stop_times_path in names:
        stop_times = _read_gtfs_csv(zin, stop_times_path).with_columns(pl.col("trip_id").cast(pl.String))
        route_stop_times = stop_times.filter(pl.col("trip_id").is_in(trip_ids))
        _write_df_to_zip(zout, stop_times_path, route_stop_times)
        if "stop_id" in route_stop_times.columns:
            stop_ids = set(route_stop_times["stop_id"].cast(pl.String).to_list())

    stops_path = _gtfs_path(agency, "stops.txt")
    if stops_path in names and stop_ids:
        stops = _read_gtfs_csv(zin, stops_path).with_columns(pl.col("stop_id").cast(pl.String))
        _write_df_to_zip(zout, stops_path, stops.filter(pl.col("stop_id").is_in(stop_ids)))

    calendar_path = _gtfs_path(agency, "calendar.txt")
    if calendar_path in names and service_ids:
        calendar = _read_gtfs_csv(zin, calendar_path).with_columns(pl.col("service_id").cast(pl.String))
        _write_df_to_zip(zout, calendar_path, calendar.filter(pl.col("service_id").is_in(service_ids)))

    calendar_dates_path = _gtfs_path(agency, "calendar_dates.txt")
    if calendar_dates_path in names and service_ids:
        calendar_dates = _read_gtfs_csv(zin, calendar_dates_path).with_columns(pl.col("service_id").cast(pl.String))
        _write_df_to_zip(zout, calendar_dates_path, calendar_dates.filter(pl.col("service_id").is_in(service_ids)))

    shapes_path = _gtfs_path(agency, "shapes.txt")
    if shapes_path in names and shape_ids:
        shapes = _read_gtfs_csv(zin, shapes_path).with_columns(pl.col("shape_id").cast(pl.String))
        _write_df_to_zip(zout, shapes_path, shapes.filter(pl.col("shape_id").is_in(shape_ids)))

    for optional in ["agency.txt", "feed_info.txt", "fare_attributes.txt", "fare_rules.txt", "__Licence.txt"]:
        source_path = _gtfs_path(agency, optional)
        if source_path in names:
            _copy_zip_member(zin, zout, source_path, source_path)

    return {"trip_ids": trip_ids, "route_ids": route_ids, "stop_ids": stop_ids}


def _load_matched_file_pairs(matches_zip_path: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    with zipfile.ZipFile(matches_zip_path, "r") as z:
        with z.open(MATCHES_CSV_NAME, "r") as raw:
            text = TextIOWrapper(raw, encoding="utf-8", newline="")
            reader = csv.DictReader(text)
            for row in reader:
                vp_file = row.get("vehicle_positions_file", "")
                tu_file = row.get("trip_updates_file", "")

                if not vp_file or not tu_file:
                    continue

                pairs.append((vp_file, tu_file))

    return pairs


def _agency_from_gtfs_rt_path(path: str) -> str | None:
    parts = PurePosixPath(path).parts
    if len(parts) < 3:
        return None
    if parts[0] != GTFS_RT_ROOT:
        return None
    return parts[1]


def _should_keep_entity(entity: Any, ids: dict[str, set[str]]) -> bool:
    if entity.HasField("trip_update"):
        tu = entity.trip_update

        if tu.trip.route_id and tu.trip.route_id not in ids["route_ids"]:
            return False

        if tu.trip.trip_id and tu.trip.trip_id not in ids["trip_ids"]:
            return False

        reduced_stu = [stu for stu in tu.stop_time_update if stu.stop_id and stu.stop_id in ids["stop_ids"]]
        del tu.stop_time_update[:]
        tu.stop_time_update.extend(reduced_stu)

        return len(reduced_stu) > 0

    if entity.HasField("vehicle"):
        vp = entity.vehicle
        if vp.trip.route_id and vp.trip.route_id not in ids["route_ids"]:
            return False

        if vp.trip.trip_id and vp.trip.trip_id not in ids["trip_ids"]:
            return False

        if vp.stop_id and vp.stop_id not in ids["stop_ids"]:
            return False

        return True

    if entity.HasField("alert"):
        alert = entity.alert
        for selector in alert.informed_entity:
            if selector.trip.trip_id and selector.trip.trip_id in ids["trip_ids"]:
                return True

            if selector.route_id and selector.route_id in ids["route_ids"]:
                return True

            if selector.stop_id and selector.stop_id in ids["stop_ids"]:
                return True

    return False


def _reduce_feed_bytes(raw: bytes, ids: dict[str, set[str]]) -> tuple[bytes, int, int]:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(raw)

    reduced = gtfs_realtime_pb2.FeedMessage()
    reduced.header.CopyFrom(feed.header)

    total = len(feed.entity)
    kept = 0

    for entity in feed.entity:
        if _should_keep_entity(entity, ids):
            reduced.entity.add().CopyFrom(entity)
            kept += 1

    return reduced.SerializeToString(), total, kept


def _write_reduced_gtfs_rt_file(
    zin: zipfile.ZipFile,
    zout: zipfile.ZipFile,
    path: str,
    ids: dict[str, set[str]],
) -> tuple[int, int, bool]:
    raw = zin.read(path)
    reduced_bytes, total, kept = _reduce_feed_bytes(raw, ids)
    if kept == 0:
        return total, kept, False

    source_info = zin.getinfo(path)
    out_info = zipfile.ZipInfo(filename=path, date_time=source_info.date_time)
    out_info.compress_type = zipfile.ZIP_DEFLATED
    out_info.external_attr = source_info.external_attr

    zout.writestr(out_info, reduced_bytes)
    return total, kept, True


def filter_by_route_id(
    input_zip_path: str, matches_zip_path: str, output_zip_path: str, agency: str, route_ids: str | list[str], force: bool = False
) -> str:
    """
    Kedro node.

    Inputs:
      - input_zip_path:
          ZIP containing GTFS/AGENCY/*.txt and GTFS-RT/AGENCY/.../*.pb

      - matches_zip_path:
          ZIP containing feed_file_sync.csv with valid matched VP/TU pairs

      - agency:
          GTFS agency folder to reduce, e.g. "RTL"

      - route_ids:
          route_id or list of route_id values to keep

    Output:
      - output_zip_path:
          ZIP containing:
            - reduced static GTFS for the selected route_id(s)
            - reduced GTFS-RT protobufs only for matched file pairs
            - no unmatched GTFS-RT files
            - no GTFS-RT entities outside the reduced static GTFS
    """
    output_path = Path(output_zip_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        logger.info("Output already exists, skipping route GTFS/GTFS-RT reduction: %s", output_path)
        return output_zip_path

    route_id_set = _as_set(route_ids)
    matched_pairs = _load_matched_file_pairs(matches_zip_path)

    logger.info("Loaded %d matched VP/TU pair(s) from %s", len(matched_pairs), matches_zip_path)

    processed_files: set[str] = set()
    processed = 0
    written = 0
    skipped = 0
    total_entities = 0
    kept_entities = 0

    with (
        zipfile.ZipFile(input_zip_path, "r") as zin,
        zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zout,
    ):
        ids = _extract_static_gtfs_for_routes(zin=zin, zout=zout, agency=agency, route_ids=route_id_set)

        logger.info(
            "Reduced static GTFS. agency=%s route_ids=%s trips=%d stops=%d",
            agency,
            sorted(route_id_set),
            len(ids["trip_ids"]),
            len(ids["stop_ids"]),
        )

        for vp_file, tu_file in tqdm(
            matched_pairs,
            desc="Reducing matched GTFS-RT pairs",
            unit="pair",
        ):
            for rt_file in (vp_file, tu_file):
                if rt_file in processed_files:
                    continue

                processed_files.add(rt_file)

                if _agency_from_gtfs_rt_path(rt_file) != agency:
                    skipped += 1
                    continue

                if rt_file not in zin.namelist():
                    logger.warning("Matched GTFS-RT file not found in input ZIP: %s", rt_file)
                    skipped += 1
                    continue

                try:
                    total, kept, did_write = _write_reduced_gtfs_rt_file(zin=zin, zout=zout, path=rt_file, ids=ids)

                    processed += 1
                    total_entities += total
                    kept_entities += kept

                    if did_write:
                        written += 1
                    else:
                        skipped += 1

                except Exception:
                    logger.exception("Failed to reduce GTFS-RT file: %s", rt_file)
                    skipped += 1

    ratio = 100 * kept_entities / max(total_entities, 1)
    logger.info(
        "Finished route GTFS/GTFS-RT reduction. processed=%d written=%d skipped=%d " "kept_entities=%d total_entities=%d ratio=%.1f%% output=%s",
        processed,
        written,
        skipped,
        kept_entities,
        total_entities,
        ratio,
        output_zip_path,
    )

    return output_zip_path
