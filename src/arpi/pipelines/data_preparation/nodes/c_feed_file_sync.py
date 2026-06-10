# nodes.py

from __future__ import annotations

import csv
import logging
import re
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta, timezone
from io import TextIOWrapper
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

from google.transit import gtfs_realtime_pb2
from tqdm import tqdm

logger = logging.getLogger(__name__)

GTFS_RT_ROOT = "GTFS-RT"
TRIP_UPDATES = "trip_updates"
VEHICLE_POSITIONS = "vehicle_positions"

FILENAME_RE = re.compile(r"^(?P<date>\d{8})_(?P<time>\d{6})_" r"(?P<kind>vehicle_positions|trip_updates)_" r"(?P<agency>.+)\.pb$")


def _parse_gtfs_rt_file_path(path: str) -> tuple[str, str, str] | None:
    parts = PurePosixPath(path).parts

    if len(parts) < 7:
        return None

    root, agency_from_path, feed_type = parts[:3]
    filename = parts[-1]

    if root != GTFS_RT_ROOT:
        return None

    if feed_type not in {VEHICLE_POSITIONS, TRIP_UPDATES}:
        return None

    match = FILENAME_RE.match(filename)
    if match is None:
        return None

    expected_kind = {
        VEHICLE_POSITIONS: "vehicle_positions",
        TRIP_UPDATES: "trip_updates",
    }[feed_type]

    if match.group("kind") != expected_kind:
        return None

    return agency_from_path, feed_type, path


def _read_feed_header_timestamp(zin: ZipFile, zip_path: str) -> datetime | None:
    message = gtfs_realtime_pb2.FeedMessage()

    try:
        with zin.open(zip_path, "r") as f:
            message.ParseFromString(f.read())
    except Exception:
        logger.exception("Failed to parse protobuf file: %s", zip_path)
        return None

    if not message.HasField("header"):
        logger.warning("Missing GTFS-RT header: %s", zip_path)
        return None

    if not message.header.HasField("timestamp"):
        logger.warning("Missing GTFS-RT header.timestamp: %s", zip_path)
        return None

    return datetime.fromtimestamp(message.header.timestamp, tz=timezone.utc)


def feed_file_sync(input_zip_path: str, output_zip_path: str, lag_seconds: int = 10, force: bool = False) -> str:
    output_path = Path(output_zip_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        logger.info("Output already exists, skipping feed_file_sync: %s", output_path)
        return output_zip_path

    lag = timedelta(seconds=lag_seconds)
    trip_updates: list[tuple[str, datetime, str]] = []

    # Pass 1: collect trip_updates using protobuf header.timestamp.
    with ZipFile(input_zip_path, "r") as zin:
        infos = zin.infolist()
        for info in tqdm(infos, desc="Pass 1: trip_updates", unit="file"):
            if info.is_dir():
                continue

            parsed = _parse_gtfs_rt_file_path(info.filename)
            if parsed is None:
                continue

            agency, feed_type, filename = parsed

            if feed_type != TRIP_UPDATES:
                continue

            timestamp = _read_feed_header_timestamp(zin, filename)
            if timestamp is None:
                continue

            trip_updates.append((agency, timestamp, filename))

    trip_updates.sort(key=lambda item: (item[0], item[1], item[2]))
    trip_update_keys = [(agency, timestamp) for agency, timestamp, _ in trip_updates]
    rows_written = 0
    unmatched = 0

    # Pass 2: scan vehicle_positions, read protobuf header.timestamp, match.
    with (
        ZipFile(input_zip_path, "r") as zin,
        ZipFile(output_zip_path, "w", compression=ZIP_DEFLATED, compresslevel=9, allowZip64=True) as zout,
    ):
        infos = zin.infolist()
        with zout.open("feed_file_sync.csv", "w") as raw:
            text = TextIOWrapper(raw, encoding="utf-8", newline="")
            writer = csv.writer(text)
            writer.writerow(
                ["vehicle_positions_file", "vehicle_positions_feed_timestamp", "trip_updates_file", "trip_updates_feed_timestamp", "lag"]
            )

            for info in tqdm(infos, desc="Pass 2: vehicle_positions", unit="file"):
                if info.is_dir():
                    continue

                parsed = _parse_gtfs_rt_file_path(info.filename)
                if parsed is None:
                    continue

                agency, feed_type, vp_file = parsed
                if feed_type != VEHICLE_POSITIONS:
                    continue

                vp_timestamp = _read_feed_header_timestamp(zin, vp_file)
                if vp_timestamp is None:
                    writer.writerow([vp_file, vp_timestamp.isoformat(), "", "", ""])
                    unmatched += 1
                    rows_written += 1
                    continue

                lower_key = (agency, vp_timestamp)
                upper_key = (agency, vp_timestamp + lag)
                start_idx = bisect_left(trip_update_keys, lower_key)
                end_idx = bisect_right(trip_update_keys, upper_key)
                if start_idx >= end_idx:
                    writer.writerow([vp_file, vp_timestamp.isoformat(), "", "", ""])
                    unmatched += 1
                    rows_written += 1
                    continue

                tu_agency, tu_timestamp, tu_file = trip_updates[start_idx]
                if tu_agency != agency:
                    writer.writerow([vp_file, vp_timestamp.isoformat(), "", "", ""])
                    unmatched += 1
                    rows_written += 1
                    continue

                actual_lag_seconds = int((tu_timestamp - vp_timestamp).total_seconds())
                writer.writerow([vp_file, vp_timestamp.isoformat(), tu_file, tu_timestamp.isoformat(), actual_lag_seconds])
                rows_written += 1

            text.flush()

    logger.info("Finished feed_file_sync. rows=%d unmatched=%d output=%s", rows_written, unmatched, output_zip_path)
    return output_zip_path
