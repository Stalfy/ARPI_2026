import argparse
import zipfile
from pathlib import Path

import polars as pl


def read_gtfs_csv(zf: zipfile.ZipFile, name: str) -> pl.DataFrame:
    with zf.open(name) as f:
        return pl.read_csv(f, infer_schema_length=0)


def write_csv(df: pl.DataFrame, out_dir: Path, name: str) -> None:
    if not df.is_empty():
        df.write_csv(out_dir / name)


def extract_route(gtfs_zip: Path, route_id: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(gtfs_zip) as zf:
        names = set(zf.namelist())

        trips = read_gtfs_csv(zf, "trips.txt").with_columns(
            pl.col("route_id").cast(pl.String),
            pl.col("trip_id").cast(pl.String),
        )

        route_trips = trips.filter(pl.col("route_id") == str(route_id))

        if route_trips.is_empty():
            raise ValueError(f"No trips found for route_id={route_id!r}")

        trip_ids = route_trips["trip_id"].unique()
        service_ids = route_trips["service_id"].unique() if "service_id" in route_trips.columns else None
        shape_ids = route_trips["shape_id"].unique() if "shape_id" in route_trips.columns else None

        write_csv(route_trips, out_dir, "trips.txt")

        if "routes.txt" in names:
            routes = read_gtfs_csv(zf, "routes.txt").with_columns(pl.col("route_id").cast(pl.String))
            write_csv(routes.filter(pl.col("route_id") == str(route_id)), out_dir, "routes.txt")

        stop_ids = None
        if "stop_times.txt" in names:
            stop_times = read_gtfs_csv(zf, "stop_times.txt").with_columns(pl.col("trip_id").cast(pl.String))
            route_stop_times = stop_times.filter(pl.col("trip_id").is_in(trip_ids))
            write_csv(route_stop_times, out_dir, "stop_times.txt")

            if "stop_id" in route_stop_times.columns:
                stop_ids = route_stop_times["stop_id"].cast(pl.String).unique()

        if "stops.txt" in names and stop_ids is not None:
            stops = read_gtfs_csv(zf, "stops.txt").with_columns(pl.col("stop_id").cast(pl.String))
            write_csv(stops.filter(pl.col("stop_id").is_in(stop_ids)), out_dir, "stops.txt")

        if "calendar.txt" in names and service_ids is not None:
            calendar = read_gtfs_csv(zf, "calendar.txt").with_columns(pl.col("service_id").cast(pl.String))
            write_csv(calendar.filter(pl.col("service_id").is_in(service_ids)), out_dir, "calendar.txt")

        if "calendar_dates.txt" in names and service_ids is not None:
            calendar_dates = read_gtfs_csv(zf, "calendar_dates.txt").with_columns(pl.col("service_id").cast(pl.String))
            write_csv(calendar_dates.filter(pl.col("service_id").is_in(service_ids)), out_dir, "calendar_dates.txt")

        if "shapes.txt" in names and shape_ids is not None:
            shapes = read_gtfs_csv(zf, "shapes.txt").with_columns(pl.col("shape_id").cast(pl.String))
            write_csv(shapes.filter(pl.col("shape_id").is_in(shape_ids)), out_dir, "shapes.txt")

        for optional in ["agency.txt", "feed_info.txt", "fare_attributes.txt", "fare_rules.txt", "__Licence.txt"]:
            if optional in names:
                with zf.open(optional) as src, open(out_dir / optional, "wb") as dst:
                    dst.write(src.read())

    print(f"Extracted route_id={route_id} to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gtfs_zip", type=Path)
    parser.add_argument("route_id")
    parser.add_argument("--out-dir", type=Path, default=Path("gtfs_route_extract"))
    args = parser.parse_args()

    extract_route(args.gtfs_zip, args.route_id, args.out_dir)


if __name__ == "__main__":
    main()
