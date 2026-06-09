# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.

import io
import os

import polars as pl

from arpi.discovery.files import FileClient
from arpi.discovery.shapes import shapes_to_segments
from arpi.log import ApplicationLogger
from arpi.models.gtfs import GTFSFile
from arpi.models.gtfs_agency import Agency
from arpi.models.gtfs_route import Route
from arpi.models.gtfs_segment import ShapeSegment
from arpi.models.gtfs_shape import Shape
from arpi.models.gtfs_stop import Stop
from arpi.models.gtfs_stop_time import StopTime
from arpi.models.gtfs_trip import Trip
from arpi.models.transit import TransitAgency


class GtfsFetchService:
    """Service to fetch and parse GTFS data into Polars DataFrames.

    Attributes:
        file_client (FileClient): Client to interact with file storage.
    """

    def __init__(self, file_client: FileClient, batch_size: int | None = None) -> None:
        """Initialize the fetch service with a `FileClient` instance."""
        self.file_client = file_client
        self._logger = ApplicationLogger(__class__.__name__)
        self.cpu_cores = os.cpu_count() if batch_size is None else batch_size
        self._cache_stop_times_mappings: dict[str, dict[tuple[str, str], StopTime]] = {}

    def load_stop_times_mappings(self, transit_agency: TransitAgency) -> dict[tuple[str, str], StopTime]:
        """Load stop times file from GCS and return a (trip_id, stop_id) lookup.

        Args:
            transit_agency (TransitAgency): Agency identifier used to build GCS path.

        Returns:
            dict[tuple[str, str], StopTime]: A lookup dictionary with (trip_id, stop_id) keys.
        """
        key = str(transit_agency)
        if key in self._cache_stop_times_mappings:
            return self._cache_stop_times_mappings[key]
        try:
            gcs_path = f"GTFS/{transit_agency}/{GTFSFile.STOP_TIMES}"
            content = self.file_client.retrieve_file_content(gcs_path)

            if not content:
                self._logger.err(f"Could not download {GTFSFile.STOP_TIMES} from GCS path: {gcs_path}")
                return {}

            df = pl.read_csv(
                io.BytesIO(content),
                schema_overrides={
                    "trip_id": pl.Utf8,
                    "stop_id": pl.Utf8,
                    "stop_sequence": pl.Int32,
                    "arrival_time": pl.Utf8,
                    "departure_time": pl.Utf8,
                    "stop_headsign": pl.Utf8,
                    "pickup_type": pl.Int32,
                    "drop_off_type": pl.Int32,
                    "shape_dist_traveled": pl.Float64,
                    "timepoint": pl.Int32,
                },
                ignore_errors=True,
                infer_schema_length=10000,
            )

            df = df.with_columns(
                [
                    pl.col("trip_id").str.strip_chars(),
                    pl.col("stop_id").str.strip_chars(),
                    pl.col("arrival_time").str.strip_chars(),
                    pl.col("departure_time").str.strip_chars(),
                ]
            )

            mapping = {
                (row["trip_id"], row["stop_id"]): StopTime(
                    trip_id=row["trip_id"],
                    stop_id=row["stop_id"],
                    stop_sequence=row["stop_sequence"],
                    arrival_time=row["arrival_time"],
                    departure_time=row["departure_time"],
                    stop_headsign=row.get("stop_headsign"),
                    pickup_type=row.get("pickup_type"),
                    drop_off_type=row.get("drop_off_type"),
                    shape_dist_traveled=row.get("shape_dist_traveled"),
                    timepoint=row.get("timepoint"),
                )
                for row in df.iter_rows(named=True)
            }
            self._logger.dbg(f"{transit_agency}: {len(mapping)} stop_times mapping entries loaded.")
            self._cache_stop_times_mappings[key] = mapping
            return mapping

        except Exception as e:
            self._logger.err(f"Error loading {GTFSFile.STOP_TIMES} from GCS: {e}")
            return {}

    def load_stop_times_df(self, transit_agency: TransitAgency) -> pl.DataFrame:
        """Load stop_times as a Polars DataFrame.

        Args:
            transit_agency (TransitAgency): Agency identifier used to build GCS path.

        Returns:
            pl.DataFrame: DataFrame containing stop_times data.
        """
        try:
            gcs_path = f"GTFS/{transit_agency}/{GTFSFile.STOP_TIMES}"
            content = self.file_client.retrieve_file_content(gcs_path)

            if not content:
                self._logger.err(f"Could not download {GTFSFile.STOP_TIMES} from GCS path: {gcs_path}")
                return pl.DataFrame()

            df = pl.read_csv(
                io.BytesIO(content),
                schema_overrides={
                    "trip_id": pl.Utf8,
                    "stop_id": pl.Utf8,
                    "stop_sequence": pl.Int32,
                    "arrival_time": pl.Utf8,
                    "departure_time": pl.Utf8,
                },
                ignore_errors=True,
                infer_schema_length=10000,
            )

            df = df.with_columns(
                [
                    pl.col("trip_id").str.strip_chars(),
                    pl.col("stop_id").str.strip_chars(),
                ]
            )

            self._logger.dbg(f"{transit_agency}: stop_times DataFrame loaded ({df.shape[0]} row(s)).")
            return df

        except Exception as e:
            self._logger.err(f"Error loading {GTFSFile.STOP_TIMES} from GCS: {e}")
            return pl.DataFrame()

    def load_agency(self, transit_agency: TransitAgency) -> dict[str, Agency]:
        """Load agency file from GCS and return an agency_id lookup.

        Args:
            transit_agency (TransitAgency): Agency identifier used to build GCS path.

        Returns:
            dict[str, Agency]: A lookup dictionary with agency_id keys.
        """
        try:
            gcs_path = f"GTFS/{transit_agency}/{GTFSFile.AGENCY}"
            content = self.file_client.retrieve_file_content(gcs_path)

            if not content:
                self._logger.err(f"Could not download {GTFSFile.AGENCY} from GCS path: {gcs_path}")
                return {}

            df = pl.read_csv(
                io.BytesIO(content),
                schema_overrides={
                    "agency_id": pl.Utf8,
                    "agency_name": pl.Utf8,
                    "agency_url": pl.Utf8,
                    "agency_timezone": pl.Utf8,
                    "agency_phone": pl.Utf8,
                    "agency_lang": pl.Utf8,
                    "agency_fare_url": pl.Utf8,
                },
                ignore_errors=True,
                infer_schema_length=10000,
            )

            df = df.with_columns(
                [
                    pl.col("agency_id").str.strip_chars(),
                    pl.col("agency_name").str.strip_chars(),
                ]
            )

            mapping = {
                row["agency_id"]: Agency(
                    agency_id=row["agency_id"],
                    agency_name=row["agency_name"],
                    agency_url=row["agency_url"],
                    agency_timezone=row["agency_timezone"],
                    agency_phone=row.get("agency_phone"),
                    agency_lang=row.get("agency_lang"),
                    agency_fare_url=row.get("agency_fare_url"),
                )
                for row in df.iter_rows(named=True)
            }
            self._logger.dbg(f"{transit_agency}: {len(mapping)} agency(ies) loaded.")
            return mapping

        except Exception as e:
            self._logger.err(f"Error loading {GTFSFile.AGENCY} from GCS: {e}")
            return {}

    def load_routes(self, transit_agency: TransitAgency) -> dict[str, Route]:
        """Load routes file from GCS and return a route_id lookup.

        Args:
            transit_agency (TransitAgency): Agency identifier used to build GCS path.

        Returns:
            dict[str, Route]: A lookup dictionary with route_id keys.
        """
        try:
            gcs_path = f"GTFS/{transit_agency}/{GTFSFile.ROUTES}"
            content = self.file_client.retrieve_file_content(gcs_path)

            if not content:
                self._logger.err(f"Could not download {GTFSFile.ROUTES} from GCS path: {gcs_path}")
                return {}

            df = pl.read_csv(
                io.BytesIO(content),
                schema_overrides={
                    "route_id": pl.Utf8,
                    "agency_id": pl.Utf8,
                    "route_short_name": pl.Utf8,
                    "route_long_name": pl.Utf8,
                    "route_desc": pl.Utf8,
                    "route_type": pl.Int32,
                    "route_url": pl.Utf8,
                    "route_color": pl.Utf8,
                    "route_text_color": pl.Utf8,
                },
                ignore_errors=True,
                infer_schema_length=10000,
            )

            df = df.with_columns(
                [
                    pl.col("route_id").str.strip_chars(),
                    pl.col("route_short_name").str.strip_chars(),
                    pl.col("route_long_name").str.strip_chars(),
                ]
            )

            mapping = {
                row["route_id"]: Route(
                    route_id=row["route_id"],
                    route_short_name=row["route_short_name"],
                    route_long_name=row["route_long_name"],
                    route_type=row["route_type"],
                    agency_id=row.get("agency_id"),
                    route_desc=row.get("route_desc"),
                    route_url=row.get("route_url"),
                    route_color=row.get("route_color"),
                    route_text_color=row.get("route_text_color"),
                )
                for row in df.iter_rows(named=True)
            }
            self._logger.dbg(f"{transit_agency}: {len(mapping)} route(s) loaded.")
            return mapping

        except Exception as e:
            self._logger.err(f"Error loading {GTFSFile.ROUTES} from GCS: {e}")
            return {}

    def load_trips(self, transit_agency: TransitAgency) -> dict[str, Trip]:
        """Load trips file from GCS and return a trip_id lookup.

        Args:
            transit_agency (TransitAgency): Agency identifier used to build GCS path.

        Returns:
            dict[str, Trip]: A lookup dictionary with trip_id keys.
        """
        try:
            gcs_path = f"GTFS/{transit_agency}/{GTFSFile.TRIPS}"
            content = self.file_client.retrieve_file_content(gcs_path)

            if not content:
                self._logger.err(f"Could not download {GTFSFile.TRIPS} from GCS path: {gcs_path}")
                return {}

            df = pl.read_csv(
                io.BytesIO(content),
                schema_overrides={
                    "route_id": pl.Utf8,
                    "service_id": pl.Utf8,
                    "trip_id": pl.Utf8,
                    "trip_headsign": pl.Utf8,
                    "direction_id": pl.Int32,
                    "block_id": pl.Utf8,
                    "shape_id": pl.Utf8,
                    "wheelchair_accessible": pl.Int32,
                },
                ignore_errors=True,
                infer_schema_length=10000,
            )

            df = df.with_columns(
                [
                    pl.col("route_id").str.strip_chars(),
                    pl.col("service_id").str.strip_chars(),
                    pl.col("trip_id").str.strip_chars(),
                ]
            )

            mapping = {
                row["trip_id"]: Trip(
                    route_id=row["route_id"],
                    service_id=row["service_id"],
                    trip_id=row["trip_id"],
                    trip_headsign=row.get("trip_headsign"),
                    direction_id=row.get("direction_id"),
                    block_id=row.get("block_id"),
                    shape_id=row.get("shape_id"),
                    wheelchair_accessible=row.get("wheelchair_accessible"),
                )
                for row in df.iter_rows(named=True)
            }
            self._logger.dbg(f"{transit_agency}: {len(mapping)} trip(s) loaded.")
            return mapping

        except Exception as e:
            self._logger.err(f"Error loading {GTFSFile.TRIPS} from GCS: {e}")
            return {}

    def load_stops(self, transit_agency: TransitAgency) -> dict[str, Stop]:
        """Load stops file from GCS and return a stop_id lookup.

        Args:
            transit_agency (TransitAgency): Agency identifier used to build GCS path.

        Returns:
            dict[str, Stop]: A lookup dictionary with stop_id keys.
        """
        try:
            gcs_path = f"GTFS/{transit_agency}/{GTFSFile.STOPS}"
            content = self.file_client.retrieve_file_content(gcs_path)

            if not content:
                self._logger.err(f"Could not download {GTFSFile.STOPS} from GCS path: {gcs_path}")
                return {}

            df = pl.read_csv(
                io.BytesIO(content),
                schema_overrides={
                    "stop_id": pl.Utf8,
                    "stop_code": pl.Utf8,
                    "stop_name": pl.Utf8,
                    "stop_lat": pl.Float64,
                    "stop_lon": pl.Float64,
                    "location_type": pl.Int32,
                    "parent_station": pl.Utf8,
                    "wheelchair_boarding": pl.Int32,
                },
                ignore_errors=True,
                infer_schema_length=10000,
            )

            df = df.with_columns(
                [
                    pl.col("stop_id").str.strip_chars(),
                    pl.col("stop_name").str.strip_chars(),
                ]
            )

            mapping = {
                row["stop_id"]: Stop(
                    stop_id=row["stop_id"],
                    stop_name=row["stop_name"],
                    stop_lat=row["stop_lat"],
                    stop_lon=row["stop_lon"],
                    stop_code=row.get("stop_code"),
                    location_type=row.get("location_type"),
                    parent_station=row.get("parent_station"),
                    wheelchair_boarding=row.get("wheelchair_boarding"),
                )
                for row in df.iter_rows(named=True)
            }
            self._logger.dbg(f"{transit_agency}: {len(mapping)} stop(s) loaded.")
            return mapping

        except Exception as e:
            self._logger.err(f"Error loading {GTFSFile.STOPS} from GCS: {e}")
            return {}

    def load_shapes(self, transit_agency: TransitAgency) -> dict[str, list[Shape]]:
        """Load shapes file from GCS and return a shape_id lookup.

        Args:
            transit_agency (TransitAgency): Agency identifier used to build GCS path.

        Returns:
            dict[str, list[Shape]]: A lookup dictionary with shape_id keys mapping to ordered lists of Shape points.
        """
        try:
            gcs_path = f"GTFS/{transit_agency}/{GTFSFile.SHAPES}"
            content = self.file_client.retrieve_file_content(gcs_path)

            if not content:
                self._logger.err(f"Could not download {GTFSFile.SHAPES} from GCS path: {gcs_path}")
                return {}

            df = pl.read_csv(
                io.BytesIO(content),
                schema_overrides={
                    "shape_id": pl.Utf8,
                    "shape_pt_lat": pl.Float64,
                    "shape_pt_lon": pl.Float64,
                    "shape_pt_sequence": pl.Int32,
                    "shape_dist_traveled": pl.Float64,
                },
                ignore_errors=True,
                infer_schema_length=10000,
            )

            df = df.with_columns(
                [
                    pl.col("shape_id").str.strip_chars(),
                ]
            )

            # Sort by shape_id and shape_pt_sequence to ensure correct ordering
            df = df.sort(["shape_id", "shape_pt_sequence"])

            # Group shapes by shape_id
            shapes_dict: dict[str, list[Shape]] = {}
            for row in df.iter_rows(named=True):
                shape_id = row["shape_id"]
                shape_point = Shape(
                    shape_id=shape_id,
                    shape_pt_lat=row["shape_pt_lat"],
                    shape_pt_lon=row["shape_pt_lon"],
                    shape_pt_sequence=row["shape_pt_sequence"],
                    shape_dist_traveled=row.get("shape_dist_traveled"),
                )

                if shape_id not in shapes_dict:
                    shapes_dict[shape_id] = []
                shapes_dict[shape_id].append(shape_point)

            if not shapes_dict:
                self._logger.wrn(f"{transit_agency}: No shapes found in {GTFSFile.SHAPES}.")
            else:
                self._logger.dbg(f"{transit_agency}: {len(shapes_dict)} shape(s) loaded ({df.shape[0]} point(s) total).")
            return shapes_dict

        except Exception as e:
            self._logger.err(f"Error loading {GTFSFile.SHAPES} from GCS: {e}")
            return {}

    def load_shape_segments(self, transit_agency: TransitAgency) -> dict[str, list[ShapeSegment]]:
        """Load shapes file from GCS and return segments with calculated distances.

        Args:
            transit_agency (TransitAgency): Agency identifier used to build GCS path.

        Returns:
            dict[str, list[ShapeSegment]]: A lookup dictionary with shape_id keys mapping to ordered lists of segments.
        """
        shapes_dict = self.load_shapes(transit_agency)

        if not shapes_dict:
            self._logger.wrn(f"{transit_agency}: No shapes available; shape segments will be empty.")

        segments_dict: dict[str, list[ShapeSegment]] = {}
        for shape_id, shape_points in shapes_dict.items():
            segments_dict[shape_id] = shapes_to_segments(shape_points)

        self._logger.dbg(f"{transit_agency}: {len(segments_dict)} shape segment group(s) built.")
        return segments_dict
