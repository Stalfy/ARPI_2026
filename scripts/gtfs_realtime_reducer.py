import argparse
import zipfile
from pathlib import Path

import polars as pl
from google.transit import gtfs_realtime_pb2
from tqdm import tqdm


def load_gtfs_ids(gtfs_zip: Path) -> dict[str, set[str]]:
    with zipfile.ZipFile(gtfs_zip) as zf:
        with zf.open("trips.txt") as f:
            trips = pl.read_csv(f, infer_schema_length=0).with_columns(
                pl.col("trip_id").cast(pl.String),
                pl.col("route_id").cast(pl.String),
            )

        stops = pl.DataFrame({"stop_id": []})
        if "stops.txt" in zf.namelist():
            with zf.open("stops.txt") as f:
                stops = pl.read_csv(f, infer_schema_length=0).with_columns(
                    pl.col("stop_id").cast(pl.String),
                )

    return {
        "trip_ids": set(trips["trip_id"].to_list()),
        "route_ids": set(trips["route_id"].to_list()),
        "stop_ids": set(stops["stop_id"].to_list()),
    }


def should_keep(entity, ids: dict[str, set[str]]) -> bool:
    if entity.HasField("trip_update"):
        tu = entity.trip_update

        if tu.trip.trip_id and tu.trip.trip_id in ids["trip_ids"]:
            return True

        if tu.trip.route_id and tu.trip.route_id in ids["route_ids"]:
            return True

        for stu in tu.stop_time_update:
            if stu.stop_id and stu.stop_id in ids["stop_ids"]:
                return True

    if entity.HasField("vehicle"):
        vp = entity.vehicle

        if vp.trip.trip_id and vp.trip.trip_id in ids["trip_ids"]:
            return True

        if vp.trip.route_id and vp.trip.route_id in ids["route_ids"]:
            return True

        if vp.stop_id and vp.stop_id in ids["stop_ids"]:
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


def reduce_feed_bytes(raw: bytes, ids: dict[str, set[str]]) -> tuple[bytes, int, int]:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(raw)

    reduced = gtfs_realtime_pb2.FeedMessage()
    reduced.header.CopyFrom(feed.header)

    total = len(feed.entity)
    kept = 0

    for entity in feed.entity:
        if should_keep(entity, ids):
            reduced.entity.add().CopyFrom(entity)
            kept += 1

    return reduced.SerializeToString(), total, kept


def reduce_file(input_file: Path, output_file: Path, ids: dict[str, set[str]]) -> tuple[int, int]:
    reduced_bytes, total, kept = reduce_feed_bytes(input_file.read_bytes(), ids)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(reduced_bytes)

    return total, kept


def reduce_directory(input_dir: Path, output_dir: Path, ids: dict[str, set[str]]) -> None:
    files = [f for f in input_dir.rglob("*") if f.is_file()]

    processed = 0
    kept_entities = 0
    total_entities = 0
    skipped = 0

    for input_file in tqdm(files, desc="Reducing protobuf files", unit="file"):
        rel = input_file.relative_to(input_dir)
        output_file = output_dir / rel

        try:
            total, kept = reduce_file(input_file, output_file, ids)

            processed += 1
            kept_entities += kept
            total_entities += total

        except Exception as exc:
            skipped += 1
            tqdm.write(f"SKIP {rel}: {exc}")

    print_summary(processed, skipped, kept_entities, total_entities)


def reduce_zip(input_zip: Path, output_zip: Path, ids: dict[str, set[str]]) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    kept_entities = 0
    total_entities = 0
    skipped = 0

    with zipfile.ZipFile(input_zip, "r") as zin:
        entries = [info for info in zin.infolist() if not info.is_dir()]

        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for info in tqdm(entries, desc="Reducing zip entries", unit="file"):
                raw = zin.read(info.filename)

                try:
                    reduced_bytes, total, kept = reduce_feed_bytes(raw, ids)

                    out_info = zipfile.ZipInfo(
                        filename=info.filename,
                        date_time=info.date_time,
                    )
                    out_info.compress_type = zipfile.ZIP_DEFLATED
                    out_info.external_attr = info.external_attr

                    zout.writestr(out_info, reduced_bytes)

                    processed += 1
                    kept_entities += kept
                    total_entities += total

                except Exception as exc:
                    skipped += 1
                    tqdm.write(f"SKIP {info.filename}: {exc}")

    print_summary(processed, skipped, kept_entities, total_entities)


def print_summary(
    processed: int,
    skipped: int,
    kept_entities: int,
    total_entities: int,
) -> None:
    ratio = 100 * kept_entities / max(total_entities, 1)

    print()
    print(f"Processed files : {processed:,}")
    print(f"Skipped files   : {skipped:,}")
    print(f"Kept entities   : {kept_entities:,}/{total_entities:,} ({ratio:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduce GTFS-RT protobuf feeds using trip/route/stop IDs from a GTFS.zip."
    )
    parser.add_argument(
        "--gtfs",
        type=Path,
        required=True,
        help="GTFS.zip used as the filter source.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input GTFS-RT protobuf file, directory, or zip archive.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output protobuf file, directory, or zip archive.",
    )
    args = parser.parse_args()

    ids = load_gtfs_ids(args.gtfs)

    print(f"Loaded {len(ids['trip_ids']):,} trip_id(s)")
    print(f"Loaded {len(ids['route_ids']):,} route_id(s)")
    print(f"Loaded {len(ids['stop_ids']):,} stop_id(s)")
    print()

    if args.input.is_file() and args.input.suffix.lower() == ".zip":
        reduce_zip(args.input, args.output, ids)

    elif args.input.is_file():
        total, kept = reduce_file(args.input, args.output, ids)
        print_summary(
            processed=1,
            skipped=0,
            kept_entities=kept,
            total_entities=total,
        )

    elif args.input.is_dir():
        reduce_directory(args.input, args.output, ids)

    else:
        raise FileNotFoundError(args.input)


if __name__ == "__main__":
    main()