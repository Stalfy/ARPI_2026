# uv run scripts/enrich_travel_times.py --gtfs ref/GTFS_ROUTE_4.zip --gtfs-rt ref/GTFS_REALTIME_ROUTE_4_SLIM.zip --segments ref/travel_times.csv --agency RTL --output ref/enriched_travel_times.csv --max-lag-seconds 5 --keep-unmatched

import argparse
import csv
import re
from time import sleep
import zipfile
from bisect import bisect_right
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
from google.transit import gtfs_realtime_pb2
from tqdm import tqdm

NEW_FIELDS = [
    "gtfsrt_tu_file",
    "gtfsrt_tu_file_timestamp",
    "gtfsrt_tu_file_lag_seconds",
    "gtfsrt_tu_predicted_stop_sequence",
    "gtfsrt_tu_predicted_stop_name",
    "gtfsrt_tu_predicted_stop_latitude",
    "gtfsrt_tu_predicted_stop_longitude",
    "gtfsrt_tu_predicted_stop_arrival",
    "gtfsrt_tu_predicted_stop_departure",
    "gtfsrt_tu_scheduled_stop_departure",
    "gtfsrt_tu_scheduled_stop_arrival",
]


TIMESTAMP_RE = re.compile(r"(\d{10,})")
FILENAME_TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})_")


def parse_int(value: str | int | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def infer_service_date_from_file_and_start_time(
    file_dt: datetime,
    start_time: str | None,
) -> datetime:
    """
    Infer GTFS service date from the TripUpdate file timestamp and
    TripDescriptor.start_time.

    If the trip start time-of-day is later than the file time-of-day,
    the trip started on the previous service date.

    Example:
        file_dt    = 0001-01-02 00:00:00
        start_time = 23:59:00

        service date = 0001-01-01
    """
    service_date = file_dt.date()

    if not start_time:
        return service_date.isoformat()

    h, m, s = map(int, start_time.split(":"))

    start_time_of_day_seconds = (h % 24) * 3600 + m * 60 + s
    file_time_of_day_seconds = (
        file_dt.hour * 3600
        + file_dt.minute * 60
        + file_dt.second
    )

    if start_time_of_day_seconds > file_time_of_day_seconds:
        service_date -= timedelta(days=1)

    return service_date


def extract_file_timestamp(filename: str) -> int | None:
    """
    Extracts yyyymmdd_hhmmss from filenames such as:

        20240131_153045_vehicle_positions.pb
        20240131_153048_trip_updates.pb

    Returns Unix timestamp seconds.
    """
    name = Path(filename).name
    match = FILENAME_TIMESTAMP_RE.search(name)

    if not match:
        return None

    dt = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    dt = dt.replace(tzinfo=timezone.utc)

    return int(dt.timestamp())


def extract_file_datetime(filename: str) -> datetime:
    """
    Extract datetime from GTFS-RT filename.

    Examples:
        20250808_143501_trip_updates.pb
        20250808_143501_vehicle_positions.pb

    Returns:
        timezone-aware UTC datetime
    """

    name = Path(filename).name

    match = FILENAME_TIMESTAMP_RE.search(name)
    if match is None:
        raise ValueError(
            f"Could not extract datetime from filename: {filename}"
        )

    return datetime.strptime(
        match.group(1),
        "%Y%m%d_%H%M%S",
    ).replace(tzinfo=timezone.utc)

def gtfs_time_to_seconds(value: str | None) -> int | None:
    if not value:
        return None

    h, m, s = value.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def gtfs_time_to_posix(
    gtfs_service_date: str | None,
    gtfs_time: str | None,
    agency_tz: ZoneInfo,
) -> int | None:
    if gtfs_service_date is None or not gtfs_time:
        return None

    total_seconds = gtfs_time_to_seconds(gtfs_time)
    if total_seconds is None:
        return None

    service_day = date.fromisoformat(gtfs_service_date)
    local_midnight = datetime(
        service_day.year,
        service_day.month,
        service_day.day,
        tzinfo=agency_tz,
    )

    local_dt = local_midnight + timedelta(seconds=total_seconds)
    return int(local_dt.timestamp())


def load_agency_timezone(gtfs_zip: Path) -> str:
    with zipfile.ZipFile(gtfs_zip) as zf:
        if "agency.txt" not in zf.namelist():
            return "America/Montreal"

        with zf.open("agency.txt") as f:
            agency = pl.read_csv(f, infer_schema_length=0)

    if agency.is_empty() or "agency_timezone" not in agency.columns:
        return "America/Montreal"

    return str(agency["agency_timezone"][0])


def load_stops_lookup(gtfs_zip: Path) -> dict[str, dict]:
    with zipfile.ZipFile(gtfs_zip) as zf:
        with zf.open("stops.txt") as f:
            stops = pl.read_csv(f, infer_schema_length=0)

    stops = stops.with_columns(pl.col("stop_id").cast(pl.String))

    return {row["stop_id"]: row for row in stops.iter_rows(named=True)}


def load_stop_times_lookup(
    gtfs_zip: Path,
) -> tuple[dict[tuple[str, int], dict], dict[tuple[str, str], list[dict]]]:
    with zipfile.ZipFile(gtfs_zip) as zf:
        with zf.open("stop_times.txt") as f:
            stop_times = pl.read_csv(f, infer_schema_length=0)

    stop_times = stop_times.select(
        [
            pl.col("trip_id").cast(pl.String),
            pl.col("stop_id").cast(pl.String),
            pl.col("stop_sequence").cast(pl.Int64),
            pl.col("arrival_time"),
            pl.col("departure_time"),
        ]
    )

    by_trip_sequence: dict[tuple[str, int], dict] = {}
    by_trip_stop_id: dict[tuple[str, str], list[dict]] = {}

    for row in stop_times.iter_rows(named=True):
        trip_id = row["trip_id"]
        stop_id = row["stop_id"]
        stop_sequence = int(row["stop_sequence"])

        item = {
            "trip_id": trip_id,
            "stop_id": stop_id,
            "stop_sequence": stop_sequence,
            "arrival_time": row["arrival_time"],
            "departure_time": row["departure_time"],
        }

        by_trip_sequence[(trip_id, stop_sequence)] = item
        by_trip_stop_id.setdefault((trip_id, stop_id), []).append(item)

    return by_trip_sequence, by_trip_stop_id


def resolve_scheduled_stop_time(
    trip_id: str,
    stop_sequence: int | None,
    stop_id: str | None,
    by_trip_sequence: dict[tuple[str, int], dict],
    by_trip_stop_id: dict[tuple[str, str], list[dict]],
) -> dict:
    if stop_sequence is not None:
        scheduled = by_trip_sequence.get((trip_id, stop_sequence))
        if scheduled:
            return scheduled

    if stop_id:
        candidates = by_trip_stop_id.get((trip_id, stop_id), [])
        if len(candidates) == 1:
            return candidates[0]

    return {}


def posix_to_iso(ts: int | None) -> str | None:
    if ts is None:
        return None

    return datetime.fromtimestamp(
        ts,
        tz=timezone.utc,
    ).isoformat(timespec="microseconds")


def stu_arrival_to_posix(
    stu,
    scheduled_arrival_posix: int | None,
    scheduled_departure_posix: int | None,
) -> int | None:
    if stu.HasField("arrival"):
        if stu.arrival.HasField("time"):
            return int(stu.arrival.time)
        if stu.arrival.HasField("delay") and scheduled_arrival_posix is not None:
            return scheduled_arrival_posix + int(stu.arrival.delay)

    if stu.HasField("departure") and stu.departure.HasField("delay"):
        if stu.arrival.HasField("time"):
            return int(stu.departure.time)
        if stu.arrival.HasField("delay") and scheduled_departure_posix is not None:
            return scheduled_departure_posix + int(stu.departure.delay)

    return None

def stu_departure_to_posix(
    stu,
    scheduled_arrival_posix: int | None,
    scheduled_departure_posix: int | None,
) -> int | None:
    if stu.HasField("departure") and stu.departure.HasField("delay"):
        if stu.arrival.HasField("time"):
            return int(stu.departure.time)
        if stu.arrival.HasField("delay") and scheduled_departure_posix is not None:
            return scheduled_departure_posix + int(stu.departure.delay)

    if stu.HasField("arrival"):
        if stu.arrival.HasField("time"):
            return int(stu.arrival.time)
        if stu.arrival.HasField("delay") and scheduled_arrival_posix is not None:
            return scheduled_arrival_posix + int(stu.arrival.delay)

    return None


def stop_time_event_to_posix(event, scheduled_posix: int | None) -> int | None:
    if event is None:
        return None

    if event.HasField("time"):
        return int(event.time)

    if event.HasField("delay") and scheduled_posix is not None:
        return scheduled_posix + int(event.delay)

    return None


def collect_needed_vehicle_file_timestamps(segments_csv: Path) -> set[int]:
    needed: set[int] = set()

    with segments_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            vp_file = row.get("gtfs_rt_file_current")
            if not vp_file:
                continue

            ts = extract_file_timestamp(vp_file)
            if ts is not None:
                needed.add(ts)

    return needed


def iter_trip_update_files(rt_zip: Path, gtfs_agency: str | None):
    with zipfile.ZipFile(rt_zip) as zf:
        entries = [
            info
            for info in zf.infolist()
            if (
                not info.is_dir()
                and "/trip_updates/" in info.filename
                and info.filename.lower().endswith(".pb")
                and (gtfs_agency is None or info.filename.startswith(f"{gtfs_agency}/"))
            )
        ]

        entries.sort(key=lambda i: i.filename)

        for info in tqdm(entries, desc="Reading trip_updates/*.pb", unit="file"):
            file_ts = extract_file_timestamp(info.filename)
            if file_ts is None:
                continue

            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(zf.read(info.filename))

            yield info.filename, file_ts, feed


def build_vehicle_to_trip_update_file_map(
    gtfs_rt_zip: Path,
    gtfs_agency: str | None,
    needed_vehicle_file_timestamps: set[int],
    max_lag_seconds: int,
) -> dict[int, tuple[str, int, int]]:
    trip_update_files: list[tuple[int, str]] = []

    with zipfile.ZipFile(gtfs_rt_zip) as zf:
        entries = [
            info
            for info in zf.infolist()
            if (
                not info.is_dir()
                and "/trip_updates/" in info.filename
                and info.filename.lower().endswith(".pb")
                and (gtfs_agency is None or info.filename.startswith(f"{gtfs_agency}/"))
            )
        ]

        for info in entries:
            ts = extract_file_timestamp(info.filename)
            if ts is not None:
                trip_update_files.append((ts, info.filename))

    trip_update_files.sort(key=lambda x: x[0])
    trip_update_timestamps = [ts for ts, _ in trip_update_files]

    mapping: dict[int, tuple[str, int, int]] = {}

    for vp_ts in needed_vehicle_file_timestamps:
        idx = bisect_right(trip_update_timestamps, vp_ts)

        if idx >= len(trip_update_files):
            continue

        tu_ts, tu_file = trip_update_files[idx]
        lag = tu_ts - vp_ts

        if 0 < lag <= max_lag_seconds:
            mapping[vp_ts] = (tu_file, tu_ts, lag)

    return mapping


def build_trip_update_index(
    gtfs_rt_zip: Path,
    gtfs_agency: str | None,
    needed_trip_update_files: set[str],
    stops_lookup: dict[str, dict],
    by_trip_sequence: dict[tuple[str, int], dict],
    by_trip_stop_id: dict[tuple[str, str], list[dict]],
    agency_tz: ZoneInfo,
) -> dict[tuple[str, str], list[dict]]:
    index: dict[tuple[str, str], list[dict]] = {}

    for source_file, file_ts, feed in iter_trip_update_files(gtfs_rt_zip, gtfs_agency):
        if source_file not in needed_trip_update_files:
            continue

        file_dt = extract_file_datetime(source_file)


        for entity in feed.entity:
            if not entity.HasField("trip_update"):
                continue

            tu = entity.trip_update
            trip_id = tu.trip.trip_id or None
            effective_service_date = infer_service_date_from_file_and_start_time(file_dt=file_dt, start_time=tu.trip.start_time or None)

            if not trip_id:
                continue

            for stu in tu.stop_time_update:
                stop_sequence = int(stu.stop_sequence) if stu.HasField("stop_sequence") else None
                stop_id = stu.stop_id or None

                scheduled = resolve_scheduled_stop_time(
                    trip_id=trip_id,
                    stop_sequence=stop_sequence,
                    stop_id=stop_id,
                    by_trip_sequence=by_trip_sequence,
                    by_trip_stop_id=by_trip_stop_id,
                )

                resolved_stop_id = stop_id or scheduled.get("stop_id")
                stop = stops_lookup.get(resolved_stop_id, {}) if resolved_stop_id else {}

                scheduled_arrival = scheduled.get("arrival_time")
                scheduled_departure = scheduled.get("departure_time")

                scheduled_arrival_posix = gtfs_time_to_posix(effective_service_date.strftime("%Y%m%d"), scheduled_arrival, agency_tz)
                scheduled_departure_posix = gtfs_time_to_posix(effective_service_date.strftime("%Y%m%d"), scheduled_departure, agency_tz)
                predicted_arrival = (
                    stu_arrival_to_posix(stu, scheduled_arrival_posix, scheduled_departure_posix) if stu.HasField("arrival") else None
                )

                predicted_departure = stu_departure_to_posix(stu, scheduled_arrival_posix, scheduled_departure_posix) if stu.HasField("departure") else None
                index.setdefault((source_file, trip_id), []).append(
                    {
                        "gtfsrt_tu_file": source_file,
                        "gtfsrt_tu_file_timestamp": file_ts,
                        "gtfsrt_tu_file_lag_seconds": None,
                        "gtfsrt_tu_predicted_stop_sequence": stop_sequence,
                        "gtfsrt_tu_predicted_stop_name": stop.get("stop_name"),
                        "gtfsrt_tu_predicted_stop_latitude": stop.get("stop_lat"),
                        "gtfsrt_tu_predicted_stop_longitude": stop.get("stop_lon"),
                        "gtfsrt_tu_predicted_stop_arrival": posix_to_iso(predicted_arrival),
                        "gtfsrt_tu_predicted_stop_departure": posix_to_iso(predicted_departure),
                        "gtfsrt_tu_scheduled_stop_arrival": posix_to_iso(scheduled_arrival_posix),
                        "gtfsrt_tu_scheduled_stop_departure": posix_to_iso(scheduled_departure_posix),
                    }
                )

    return index


def enrich_segments(
    segments_csv: Path,
    output_csv: Path,
    vehicle_to_trip_update_file: dict[int, tuple[str, int, int]],
    trip_update_index: dict[tuple[str, str], list[dict]],
    keep_unmatched: bool,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    unmatched = 0

    with segments_csv.open("r", newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        input_fields = reader.fieldnames or []

        output_fields = input_fields + [field for field in NEW_FIELDS if field not in input_fields]

        with output_csv.open("w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=output_fields)
            writer.writeheader()

            for row in reader:
                trip_id = row.get("gtfs_trip_id")
                vp_file = row.get("gtfs_rt_file_current")
                vp_ts = extract_file_timestamp(vp_file) if vp_file else None

                if trip_id is None or vp_ts is None:
                    unmatched += 1
                    if keep_unmatched:
                        out = dict(row)
                        for field in NEW_FIELDS:
                            out.setdefault(field, None)
                        writer.writerow(out)
                        written += 1
                    continue

                mapped = vehicle_to_trip_update_file.get(vp_ts)

                if mapped is None:
                    unmatched += 1
                    if keep_unmatched:
                        out = dict(row)
                        for field in NEW_FIELDS:
                            out.setdefault(field, None)
                        writer.writerow(out)
                        written += 1
                    continue

                tu_file, tu_ts, lag = mapped
                trip_updates = trip_update_index.get((tu_file, trip_id), [])

                if not trip_updates:
                    unmatched += 1
                    if keep_unmatched:
                        out = dict(row)
                        for field in NEW_FIELDS:
                            out.setdefault(field, None)
                        writer.writerow(out)
                        written += 1
                    continue

                for trip_update in trip_updates:
                    out = dict(row)
                    out.update(trip_update)
                    out["gtfsrt_tu_file_lag_seconds"] = lag
                    writer.writerow(out)
                    written += 1

    print(f"Wrote {written:,} row(s) -> {output_csv}")
    print(f"Unmatched segment row(s): {unmatched:,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtfs", type=Path, required=True)
    parser.add_argument("--gtfs-rt", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agency", type=str, default=None)
    parser.add_argument("--max-lag-seconds", type=int, default=3)
    parser.add_argument("--keep-unmatched", action="store_true")

    args = parser.parse_args()

    agency_timezone = load_agency_timezone(args.gtfs)
    agency_tz = ZoneInfo(agency_timezone)

    stops_lookup = load_stops_lookup(args.gtfs)
    by_trip_sequence, by_trip_stop_id = load_stop_times_lookup(args.gtfs)
    needed_vehicle_file_timestamps = collect_needed_vehicle_file_timestamps(args.segments)

    print(f"Agency timezone        : {agency_timezone}")
    print(f"Loaded stops           : {len(stops_lookup):,}")
    print(f"Loaded stop_times      : {len(by_trip_sequence):,}")
    print(f"Needed VP file stamps  : {len(needed_vehicle_file_timestamps):,}")
    print(f"Max TU lag seconds     : {args.max_lag_seconds:,}")

    vehicle_to_trip_update_file = build_vehicle_to_trip_update_file_map(
        gtfs_rt_zip=args.gtfs_rt,
        gtfs_agency=args.agency,
        needed_vehicle_file_timestamps=needed_vehicle_file_timestamps,
        max_lag_seconds=args.max_lag_seconds,
    )

    needed_trip_update_files = {tu_file for tu_file, _tu_ts, _lag in vehicle_to_trip_update_file.values()}

    print(f"Matched TU files       : {len(needed_trip_update_files):,}")

    trip_update_index = build_trip_update_index(
        gtfs_rt_zip=args.gtfs_rt,
        gtfs_agency=args.agency,
        needed_trip_update_files=needed_trip_update_files,
        stops_lookup=stops_lookup,
        by_trip_sequence=by_trip_sequence,
        by_trip_stop_id=by_trip_stop_id,
        agency_tz=agency_tz,
    )

    print(f"TripUpdate join keys   : {len(trip_update_index):,}")

    enrich_segments(
        segments_csv=args.segments,
        output_csv=args.output,
        vehicle_to_trip_update_file=vehicle_to_trip_update_file,
        trip_update_index=trip_update_index,
        keep_unmatched=args.keep_unmatched,
    )


if __name__ == "__main__":
    main()
