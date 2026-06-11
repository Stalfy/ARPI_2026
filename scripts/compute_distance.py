import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point


class RouteDistanceCalculator:
    def __init__(
        self,
        shape_id,
        trip_id=None,
        source_crs="EPSG:4326",
        metric_crs="EPSG:32188",
        stops_file="ref/GTFS/RTL/stops.txt",
        trips_file="ref/GTFS/RTL/trips.txt",
        stop_times_file="ref/GTFS/RTL/stop_times.txt",
        exclude_stations=True,
    ):
        shapes_df = pd.read_csv("ref/GTFS/RTL/shapes.txt")
        trips_df = pd.read_csv(trips_file)
        stop_times_df = pd.read_csv(stop_times_file)
        stops_df = pd.read_csv(stops_file)

        self.shape_id = shape_id

        route = shapes_df[shapes_df["shape_id"] == shape_id].sort_values("shape_pt_sequence").reset_index(drop=True)

        if len(route) < 2:
            raise ValueError(f"Shape {shape_id} has fewer than 2 points")

        self.transformer = Transformer.from_crs(source_crs, metric_crs, always_xy=True)

        self.shape_dist_traveled = route["shape_dist_traveled"].to_numpy(float)

        self.xy = np.array([self.transformer.transform(lon, lat) for lon, lat in zip(route["shape_pt_lon"], route["shape_pt_lat"])])

        self.line = LineString(self.xy)

        seg_lengths = np.sqrt(np.sum(np.diff(self.xy, axis=0) ** 2, axis=1))
        self.geom_cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])

        # Choose a trip for this shape if one is not supplied.
        if trip_id is None:
            shape_trips = trips_df[trips_df["shape_id"] == shape_id].copy()
            if shape_trips.empty:
                raise ValueError(f"No trips found for shape_id={shape_id}")
            trip_id = shape_trips.iloc[0]["trip_id"]

        self.trip_id = trip_id

        # Stop order for that trip.
        route_stop_times = stop_times_df[stop_times_df["trip_id"] == trip_id].sort_values("stop_sequence").copy()

        if route_stop_times.empty:
            raise ValueError(f"No stop_times found for trip_id={trip_id}")

        # Optional cleanup for station rows.
        if exclude_stations and "location_type" in stops_df.columns:
            stops_df = stops_df[stops_df["location_type"].isna() | (stops_df["location_type"] == 0)].copy()

        # Attach stop metadata once.
        route_stop_times = route_stop_times.merge(
            stops_df[["stop_id", "stop_name", "stop_lat", "stop_lon", "parent_station"]], on="stop_id", how="left"
        )

        self.route_stops_df = route_stop_times

        self._point_cache = {}

        # Internal preprocessing only: project each stop once.
        self.stop_positions_df = self._build_stop_positions(self.route_stops_df)

        # Lookups used at runtime.
        self.stop_sequence_lookup = self.route_stops_df.set_index("stop_sequence")["stop_id"].to_dict()

        self.stop_distance_lookup = self.stop_positions_df.set_index("stop_id")["distance_along_route_m"].to_dict()

    def _project_point_uncached(self, lon, lat):
        x, y = self.transformer.transform(lon, lat)
        p = Point(x, y)

        s = self.line.project(p)
        offset = p.distance(self.line)

        idx = np.searchsorted(self.geom_cum, s, side="right") - 1
        idx = np.clip(idx, 0, len(self.shape_dist_traveled) - 2)

        seg_start = self.geom_cum[idx]
        seg_end = self.geom_cum[idx + 1]

        if seg_end == seg_start:
            distance_along_route_m = self.shape_dist_traveled[idx]
        else:
            t = (s - seg_start) / (seg_end - seg_start)
            distance_along_route_m = self.shape_dist_traveled[idx] + t * (self.shape_dist_traveled[idx + 1] - self.shape_dist_traveled[idx])

        return distance_along_route_m, offset

    def project_point(self, lon, lat):
        key = (float(lon), float(lat))
        if key not in self._point_cache:
            self._point_cache[key] = self._project_point_uncached(lon, lat)
        return self._point_cache[key]

    def distance_between_points_m(self, lat1, lon1, lat2, lon2, verbose=False):
        d1, off1 = self.project_point(lon1, lat1)
        d2, off2 = self.project_point(lon2, lat2)
        dist = abs(d2 - d1)

        if verbose:
            return {
                "distance_along_route_m": dist,
                "point1_distance_along_route_m": d1,
                "point2_distance_along_route_m": d2,
                "point1_offset_from_route_m": off1,
                "point2_offset_from_route_m": off2,
            }

        return dist

    def _build_stop_positions(self, stops_df):
        rows = []

        for _, stop in stops_df.iterrows():
            if pd.isna(stop.get("stop_lat")) or pd.isna(stop.get("stop_lon")):
                continue

            distance_along_route_m, offset_m = self.project_point(stop["stop_lon"], stop["stop_lat"])

            rows.append(
                {
                    "stop_sequence": stop.get("stop_sequence"),
                    "stop_id": stop.get("stop_id"),
                    "stop_name": stop.get("stop_name"),
                    "parent_station": stop.get("parent_station"),
                    "stop_lat": stop.get("stop_lat"),
                    "stop_lon": stop.get("stop_lon"),
                    "distance_along_route_m": distance_along_route_m,
                    "offset_from_route_m": offset_m,
                }
            )

        return pd.DataFrame(rows).sort_values("stop_sequence").reset_index(drop=True)

    def add_distance_to_stops(self, df, current_lat_col, current_lon_col, stop_sequence_col):
        """
        Adds:
          - current_distance_along_route_m
          - current_offset_from_route_m
          - stop_id
          - stop_distance_along_route_m
          - distance_to_stop_m
        """
        out = df.copy()

        out[stop_sequence_col] = pd.to_numeric(out[stop_sequence_col], errors="coerce")

        current_pairs = list(zip(out[current_lon_col].astype(float), out[current_lat_col].astype(float)))

        current_unique = {}
        for lon, lat in set(current_pairs):
            current_unique[(lon, lat)] = self.project_point(lon, lat)

        out["current_distance_along_route_m"] = [current_unique[p][0] for p in current_pairs]
        out["current_offset_from_route_m"] = [current_unique[p][1] for p in current_pairs]

        # Lookup path only
        out["stop_id"] = out[stop_sequence_col].map(self.stop_sequence_lookup)
        out["stop_distance_along_route_m"] = out["stop_id"].map(self.stop_distance_lookup)
        out["distance_to_stop_m"] = out["stop_distance_along_route_m"] - out["current_distance_along_route_m"]

        return out
