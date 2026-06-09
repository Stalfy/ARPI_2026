# uv run python trip_segments_stream.py \
#   --gtfs ref/GTFS_ROUTE_4.zip \
#   --gtfs-rt ref/GTFS_REALTIME_ROUTE_4_SLIM.zip \
#   --gtfs_agency RTL \
#   --output travel_times.csv

import argparse
import csv
import zipfile
from bisect import bisect_left
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
from google.transit import gtfs_realtime_pb2
from tqdm import tqdm

OUT_FIELDS = [
    "gtfs_agency",
    "gtfs_service_date",
    "gtfs_route_id",
    "gtfs_service_id",
    "gtfs_trip_id",
    "gtfs_shape_id",
    "gtfs_trip_headsign",
    "gtfs_direction_id",
    "gtfs_block_id",
    "gtfsrt_vp_vehicle_id",
    "gtfsrt_vp_vehicle_label",
    "travel_time_seconds",
    "gtfsrt_vp_position_timestamp_previous",
    "gtfsrt_vp_position_latitude_previous",
    "gtfsrt_vp_position_longitude_previous",
    "gtfsrt_vp_position_timestamp_current",
    "gtfsrt_vp_position_latitude_current",
    "gtfsrt_vp_position_longitude_current",
    "gtfs_next_stop_id",
    "gtfs_next_stop_sequence_number",
    "gtfs_next_stop_name",
    "gtfs_next_stop_arrival_time",
    "gtfs_next_stop_departure_time",
    "gtfs_next_stop_latitude",
    "gtfs_next_stop_longitude",
    "gtfsrt_vp_entity_id",
    "gtfsrt_vp_trip_route_id",
    "gtfs_rt_file_previous",
    "gtfs_rt_file_current",
]


def gtfs_time_to_seconds(value: str | None) -> int | None:
    if not value:
        return None
    h, m, s = value.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def load_agency_timezone(gtfs_zip: Path) -> str:
    with zipfile.ZipFile(gtfs_zip) as zf:
        if "agency.txt" not in zf.namelist():
            return "America/Montreal"

        with zf.open("agency.txt") as f:
            gtfs_agency = pl.read_csv(f, infer_schema_length=0)

    if gtfs_agency.is_empty() or "agency_timezone" not in gtfs_agency.columns:
        return "America/Montreal"

    return str(gtfs_agency["agency_timezone"][0])


def load_trip_lookup(gtfs_zip: Path) -> dict[str, dict]:
    with zipfile.ZipFile(gtfs_zip) as zf:
        with zf.open("trips.txt") as f:
            trips = pl.read_csv(f, infer_schema_length=0)

    keep = [
        c
        for c in [
            "route_id",
            "trip_id",
            "shape_id",
            "service_id",
            "trip_headsign",
            "direction_id",
            "block_id",
        ]
        if c in trips.columns
    ]

    trips = trips.select(keep).with_columns(
        [
            pl.col("trip_id").cast(pl.String),
            pl.col("route_id").cast(pl.String) if "route_id" in keep else pl.lit(None).alias("route_id"),
            pl.col("shape_id").cast(pl.String) if "shape_id" in keep else pl.lit(None).alias("shape_id"),
        ]
    )

    return {row["trip_id"]: row for row in trips.iter_rows(named=True)}


def load_stop_time_lookup(gtfs_zip: Path) -> dict[str, list[dict]]:
    with zipfile.ZipFile(gtfs_zip) as zf:
        with zf.open("stop_times.txt") as f:
            stop_times = pl.read_csv(f, infer_schema_length=0)

        with zf.open("stops.txt") as f:
            stops = pl.read_csv(f, infer_schema_length=0)

    stops_by_id = {row["stop_id"]: row for row in stops.with_columns(pl.col("stop_id").cast(pl.String)).iter_rows(named=True)}

    lookup: dict[str, list[dict]] = {}

    stop_times = stop_times.select(
        [
            pl.col("trip_id").cast(pl.String),
            pl.col("stop_id").cast(pl.String),
            pl.col("arrival_time"),
            pl.col("departure_time"),
            pl.col("stop_sequence").cast(pl.Int64),
        ]
    )

    for row in stop_times.iter_rows(named=True):
        stop_id = row["stop_id"]
        stop = stops_by_id.get(stop_id, {})
        arrival_sec = gtfs_time_to_seconds(row["arrival_time"])

        if arrival_sec is None:
            continue

        item = {
            "arrival_sec": arrival_sec,
            "gtfs_next_stop_id": stop_id,
            "gtfs_next_stop_name": stop.get("stop_name"),
            "gtfs_next_stop_arrival_time": row["arrival_time"],
            "gtfs_next_stop_departure_time": row["departure_time"],
            "gtfs_next_stop_sequence_number": row["stop_sequence"],
            "gtfs_next_stop_latitude": stop.get("stop_lat"),
            "gtfs_next_stop_longitude": stop.get("stop_lon"),
        }

        lookup.setdefault(row["trip_id"], []).append(item)

    for trip_id in lookup:
        lookup[trip_id].sort(key=lambda x: x["arrival_sec"])

    return lookup


def local_seconds_since_service_midnight(ts: datetime, gtfs_service_date: str | None, agency_tz: ZoneInfo) -> int | None:
    if gtfs_service_date is None:
        return None

    local = ts.astimezone(agency_tz)
    service_day = date.fromisoformat(gtfs_service_date)
    day_offset = (local.date() - service_day).days
    return day_offset * 86400 + local.hour * 3600 + local.minute * 60 + local.second


def utc_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None

    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds")


def gtfs_time_to_iso(gtfs_service_date: str | None, gtfs_time: str | None, agency_tz: ZoneInfo) -> str | None:
    if gtfs_service_date is None or not gtfs_time:
        return None

    h, m, s = map(int, gtfs_time.split(":"))
    day_offset = h // 24
    h = h % 24

    base = datetime.fromisoformat(gtfs_service_date).replace(tzinfo=agency_tz)
    local_dt = (base + timedelta(days=day_offset)).replace(hour=h, minute=m, second=s, microsecond=0)
    return utc_iso(local_dt)


def find_upcoming_stop(stop_lookup: dict[str, list[dict]], trip_id: str, local_sec: int | None) -> dict:
    if local_sec is None:
        return {}

    stops = stop_lookup.get(trip_id)
    if not stops:
        return {}

    arrival_secs = [s["arrival_sec"] for s in stops]
    idx = bisect_left(arrival_secs, local_sec)

    if idx >= len(stops):
        return {}

    return stops[idx]


def iter_vehicle_position_entries(rt_zip: Path, gtfs_agency: str | None):
    with zipfile.ZipFile(rt_zip) as zf:
        entries = [
            i
            for i in zf.infolist()
            if (
                not i.is_dir()
                and "/vehicle_positions/" in i.filename
                and i.filename.lower().endswith(".pb")
                and (gtfs_agency is None or i.filename.startswith(f"{gtfs_agency}/"))
            )
        ]

        entries.sort(key=lambda i: i.filename)

        for info in tqdm(entries, desc="Reading vehicle_positions/*.pb", unit="file"):
            parts = Path(info.filename).parts

            agency_name = parts[0] if len(parts) >= 6 else None
            yyyy = parts[2] if len(parts) >= 6 else None
            mm = parts[3] if len(parts) >= 6 else None
            dd = parts[4] if len(parts) >= 6 else None
            gtfs_service_date = f"{yyyy}-{mm}-{dd}" if yyyy and mm and dd else None

            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(zf.read(info.filename))

            feed_ts = feed.header.timestamp if feed.header.HasField("timestamp") else None

            for entity in feed.entity:
                if not entity.HasField("vehicle"):
                    continue

                vp = entity.vehicle
                route_id = vp.trip.route_id
                trip_id = vp.trip.trip_id or None
                if not trip_id:
                    continue

                ts = vp.timestamp if vp.HasField("timestamp") else feed_ts
                if ts is None or not vp.HasField("position"):
                    continue

                yield {
                    "source_file": info.filename,
                    "gtfs_agency": agency_name,
                    "gtfs_service_date": gtfs_service_date,
                    "gtfsrt_vp_entity_id": entity.id,
                    "gtfs_trip_id": trip_id,
                    "gtfsrt_vp_trip_route_id": route_id or None,
                    "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc),
                    "gtfsrt_vp_vehicle_id": vp.vehicle.id if vp.HasField("vehicle") else None,
                    "gtfsrt_vp_vehicle_label": vp.vehicle.label if vp.HasField("vehicle") else None,
                    "latitude": vp.position.latitude,
                    "longitude": vp.position.longitude,
                }


def build_segment(
    prev: dict, cur: dict, trip_lookup: dict[str, dict], stop_lookup: dict[str, list[dict]], agency_tz: ZoneInfo, max_segment_seconds: int | None
) -> dict | None:
    travel_time = int((cur["timestamp"] - prev["timestamp"]).total_seconds())

    if travel_time <= 0:
        return None

    if max_segment_seconds is not None and travel_time > max_segment_seconds:
        return None

    trip_id = cur["gtfs_trip_id"]
    trip_def = trip_lookup.get(trip_id, {})
    local_sec = local_seconds_since_service_midnight(cur["timestamp"], cur["gtfs_service_date"], agency_tz)
    upcoming_stop = find_upcoming_stop(stop_lookup, trip_id, local_sec)
    gtfs_route_id = trip_def.get("route_id") or cur.get("gtfsrt_vp_trip_route_id")

    return {
        "gtfs_agency": cur["gtfs_agency"],
        "gtfs_service_date": cur["gtfs_service_date"],
        "gtfs_route_id": gtfs_route_id,
        "gtfs_trip_id": trip_id,
        "gtfs_shape_id": trip_def.get("shape_id"),
        "gtfs_service_id": trip_def.get("service_id"),
        "gtfs_trip_headsign": trip_def.get("trip_headsign"),
        "gtfs_direction_id": trip_def.get("direction_id"),
        "gtfs_block_id": trip_def.get("block_id"),
        "gtfsrt_vp_vehicle_id": cur["gtfsrt_vp_vehicle_id"],
        "gtfsrt_vp_vehicle_label": cur["gtfsrt_vp_vehicle_label"],
        "travel_time_seconds": travel_time,
        "gtfsrt_vp_position_timestamp_previous": utc_iso(prev["timestamp"]),
        "gtfsrt_vp_position_latitude_previous": prev["latitude"],
        "gtfsrt_vp_position_longitude_previous": prev["longitude"],
        "gtfsrt_vp_position_timestamp_current": utc_iso(cur["timestamp"]),
        "gtfsrt_vp_position_latitude_current": cur["latitude"],
        "gtfsrt_vp_position_longitude_current": cur["longitude"],
        "gtfs_next_stop_sequence_number": upcoming_stop.get("gtfs_next_stop_sequence_number"),
        "gtfs_next_stop_id": upcoming_stop.get("gtfs_next_stop_id"),
        "gtfs_next_stop_name": upcoming_stop.get("gtfs_next_stop_name"),
        "gtfs_next_stop_arrival_time": gtfs_time_to_iso(cur["gtfs_service_date"], upcoming_stop.get("gtfs_next_stop_arrival_time"), agency_tz),
        "gtfs_next_stop_departure_time": gtfs_time_to_iso(cur["gtfs_service_date"], upcoming_stop.get("gtfs_next_stop_departure_time"), agency_tz),
        "gtfs_next_stop_latitude": upcoming_stop.get("gtfs_next_stop_latitude"),
        "gtfs_next_stop_longitude": upcoming_stop.get("gtfs_next_stop_longitude"),
        "gtfsrt_vp_entity_id": cur["gtfsrt_vp_entity_id"],
        "gtfsrt_vp_trip_route_id": cur.get("gtfsrt_vp_trip_route_id"),
        "gtfs_rt_file_previous": prev["source_file"],
        "gtfs_rt_file_current": cur["source_file"],
    }


def stream_segments(
    gtfs_rt_zip: Path,
    output: Path,
    trip_lookup: dict[str, dict],
    stop_lookup: dict[str, list[dict]],
    agency_timezone: str,
    gtfs_agency: str | None,
    max_segment_seconds: int | None,
) -> None:
    agency_tz = ZoneInfo(agency_timezone)
    output.parent.mkdir(parents=True, exist_ok=True)

    last_seen: dict[tuple, dict] = {}
    written = 0

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()

        for cur in iter_vehicle_position_entries(gtfs_rt_zip, gtfs_agency):
            trip_id = cur["gtfs_trip_id"]
            trip_def = trip_lookup.get(trip_id, {})

            gtfs_route_id = trip_def.get("route_id") or cur.get("gtfsrt_vp_trip_route_id")
            gtfs_shape_id = trip_def.get("shape_id")

            key = (
                gtfs_route_id,
                trip_id,
                gtfs_shape_id,
                cur["gtfs_service_date"],
                cur.get("gtfsrt_vp_vehicle_id"),
            )

            prev = last_seen.get(key)
            if prev is not None:
                segment = build_segment(
                    prev=prev,
                    cur=cur,
                    trip_lookup=trip_lookup,
                    stop_lookup=stop_lookup,
                    agency_tz=agency_tz,
                    max_segment_seconds=max_segment_seconds,
                )

                if segment is not None:
                    writer.writerow(segment)
                    written += 1

            last_seen[key] = cur

    print(f"Wrote {written:,} segment row(s) -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtfs", type=Path, required=True)
    parser.add_argument("--gtfs-rt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agency", type=str, default=None)
    parser.add_argument("--max-segment-seconds", type=int, default=1300)
    args = parser.parse_args()

    agency_timezone = load_agency_timezone(args.gtfs)
    trip_lookup = load_trip_lookup(args.gtfs)
    stop_lookup = load_stop_time_lookup(args.gtfs)

    print(f"Agency timezone : {agency_timezone}")
    print(f"Loaded trips    : {len(trip_lookup):,}")
    print(f"Loaded stop seq : {len(stop_lookup):,}")

    stream_segments(
        gtfs_rt_zip=args.gtfs_rt,
        output=args.output,
        trip_lookup=trip_lookup,
        stop_lookup=stop_lookup,
        agency_timezone=agency_timezone,
        gtfs_agency=args.agency,
        max_segment_seconds=args.max_segment_seconds,
    )


if __name__ == "__main__":
    main()
