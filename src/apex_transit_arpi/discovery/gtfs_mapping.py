# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.

"""Build a GTFS mapping dict from a local GTFS zip file."""

import hashlib
import json
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Iterator

import polars as pl
import tqdm

from apex_transit_arpi.models.gtfs import GTFSFile

_TRIP_TIME_RE = re.compile(r"_(\d{1,2}:\d{2})$")

_SCHEMA_OVERRIDES: dict[GTFSFile, dict] = {
    GTFSFile.AGENCY: {"agency_id": pl.String},
    GTFSFile.ROUTES: {"route_id": pl.String, "agency_id": pl.String, "route_short_name": pl.String},
    GTFSFile.TRIPS: {
        "route_id": pl.String,
        "service_id": pl.String,
        "trip_id": pl.String,
        "block_id": pl.String,
        "shape_id": pl.String,
        "trip_short_name": pl.String,
    },
    GTFSFile.STOPS: {"stop_id": pl.String, "stop_code": pl.String},
    GTFSFile.STOP_TIMES: {"trip_id": pl.String, "stop_id": pl.String},
}


def build_mapping(zip_path: Path) -> dict:
    """Build a GTFS mapping dict from a zip file.

    Args:
        zip_path: Path to a GTFS zip file (e.g. GTFS.zip).

    Returns:
        Dict with keys ``gtfs_hash``, ``agency``, and ``routes``.
    """
    gtfs_hash = hashlib.sha256(zip_path.read_bytes()).hexdigest()

    def _bar(desc: str) -> tqdm.tqdm:
        return tqdm.tqdm(desc=f"  {desc}", unit="row", leave=True, dynamic_ncols=True)

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())

        def load(filename: GTFSFile) -> pl.DataFrame:
            entry = _find_entry(names, filename)
            if entry is None:
                return pl.DataFrame()
            return pl.read_csv(
                BytesIO(zf.read(entry)),
                schema_overrides=_SCHEMA_OVERRIDES[filename],
                encoding="utf8",
            )

        # 1. agency
        with _bar("agency.txt") as bar:
            agency = _map_agency(load)
            bar.set_postfix_str(", ".join(agency.values()) or "(no agency)")
            bar.update(len(agency))

        # 2. routes
        with _bar("routes.txt") as bar:
            routes_df = _ensure_cols(load(GTFSFile.ROUTES), {"route_id": "", "route_long_name": ""})
            routes: dict = {}
            for r in routes_df.select(["route_id", "route_long_name"]).unique().iter_rows(named=True):
                route_id = r["route_id"] or ""
                routes[route_id] = {
                    "route_name": f"{route_id} - {r['route_long_name'] or ''}",
                    "trips": {},
                }
                bar.update()
            bar.set_postfix_str(f"{len(routes)} routes")

        # 3. trips → add to routes
        trip_to_route = {}
        with _bar("trips.txt") as bar:
            trips_df = _ensure_cols(load(GTFSFile.TRIPS), {"trip_id": "", "route_id": "", "trip_headsign": ""})
            for route_id, trip_id, headsign in _iterate_trips(trips_df):
                try:
                    routes[route_id]["trips"][trip_id] = {"trip_name": headsign, "stops": {}}
                    trip_to_route[trip_id] = route_id
                    bar.update()
                except KeyError:
                    pass
            bar.set_postfix_str(f"{sum(len(r['trips']) for r in routes.values())} trips")

        # 4. stop_times → store stop_sequence as value (sorted and replaced in step 5)
        with _bar("stop_times.txt") as bar:
            stop_times_df = _ensure_cols(load(GTFSFile.STOP_TIMES), {"trip_id": "", "stop_id": "", "stop_sequence": 0})
            for trip_id, stop_id, sequence_no in _iterate_stop_times(stop_times_df):
                route_id = trip_to_route.get(trip_id, "")
                try:
                    routes[route_id]["trips"][trip_id]["stops"][stop_id] = sequence_no
                    bar.update()
                except KeyError:
                    pass
            bar.set_postfix_str(f"{stop_times_df.height:,} rows")

        # 5. stops → sort by sequence, replace with "stop_id - stop_name"
        with _bar("stops.txt") as bar:
            stops_df = _ensure_cols(load(GTFSFile.STOPS), {"stop_id": "", "stop_name": ""})
            stop_names: dict[str, str] = dict(zip(stops_df["stop_id"].cast(pl.String), stops_df["stop_name"].cast(pl.String)))
            for route_data in routes.values():
                for trip_data in route_data["trips"].values():
                    trip_data["stops"] = {
                        sid: f"{sid} - {stop_names.get(sid, sid)}" for sid, _ in sorted(trip_data["stops"].items(), key=lambda x: x[1])
                    }
            bar.update(len(stop_names))
            bar.set_postfix_str(f"{len(stop_names)} stops")

    return {"gtfs_hash": gtfs_hash, "agency": agency, "routes": routes}


def _find_entry(names: set[str], filename: str) -> str | None:
    if filename in names:
        return filename
    for name in names:
        if name.endswith("/" + filename):
            return name
    return None


def _map_agency(load) -> dict:
    df = load(GTFSFile.AGENCY)
    if df.is_empty() or "agency_id" not in df.columns or "agency_name" not in df.columns:
        return {}
    return dict(zip(df["agency_id"], df["agency_name"]))


def _ensure_cols(df: pl.DataFrame, cols_with_defaults: dict) -> pl.DataFrame:
    for col, default in cols_with_defaults.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(default).alias(col))
    return df


def _iterate_trips(trips_df: pl.DataFrame) -> Iterator[tuple[str, str, str]]:
    df = trips_df.select(["trip_id", "route_id", "trip_headsign"]).with_columns(
        [
            pl.col("trip_id").cast(pl.String),
            pl.col("route_id").cast(pl.String),
            pl.col("trip_headsign").fill_null("").cast(pl.String),
        ]
    )

    for trip_id, route_id, headsign in df.iter_rows():
        m = _TRIP_TIME_RE.search(trip_id)
        prefix = m.group(1) if m else trip_id.split("-")[0]
        yield route_id, trip_id, f"{prefix} - {headsign}"


def _iterate_stop_times(data: pl.DataFrame) -> Iterator[tuple[str, str, int]]:
    if data.is_empty():
        return

    df = data.select(["trip_id", "stop_id", "stop_sequence"]).with_columns(
        [pl.col("trip_id").cast(pl.String), pl.col("stop_id").cast(pl.String), pl.col("stop_sequence").cast(pl.Int64)]
    )
    for trip_id, stop_id, stop_sequence in df.iter_rows():
        yield trip_id, stop_id, stop_sequence


def run(zip_path: Path, output_file: Path) -> None:
    """Build and write GTFS mapping as a json.zip from zip_path to output_file.

    If output_file already exists and its stored gtfs_hash matches the current
    zip, the file is reused as-is (cache hit).

    Args:
        zip_path: Path to the GTFS zip file to read.
        output_file: Destination path for the .json.zip output.
    """
    if output_file.exists():
        current_hash = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        try:
            with zipfile.ZipFile(output_file) as zf:
                cached = json.loads(zf.read("mapping.json"))
            if cached.get("gtfs_hash") == current_hash:
                print(f"  -> {output_file} (cached)")
                return
        except Exception:
            pass  # corrupt or unreadable cache — fall through and rebuild

    mapping = build_mapping(zip_path)
    json_bytes = json.dumps(mapping, ensure_ascii=False).encode("utf-8")
    with zipfile.ZipFile(output_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mapping.json", json_bytes)
    print(f"  -> {output_file} ({len(json_bytes):,} bytes uncompressed)")


if __name__ == "__main__":
    import sys

    zip_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("GTFS.zip")
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else zip_path.with_suffix(".mappings.json.zip")

    print(f"Input:  {zip_path}")
    print(f"Output: {output_file}")
    run(zip_path, output_file)
