import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point
from pyproj import Transformer


class RouteDistanceCalculator:
    def __init__(
        self,
        shape_id,
        source_crs="EPSG:4326",
        metric_crs="EPSG:32188",
    ):
        shapes_df = pd.read_csv("ref/GTFS/RTL/shapes.txt")
        self.shape_id = shape_id

        route = (
            shapes_df[shapes_df["shape_id"] == shape_id]
            .sort_values("shape_pt_sequence")
            .reset_index(drop=True)
        )

        if len(route) < 2:
            raise ValueError(f"Shape {shape_id} has fewer than 2 points")

        self.transformer = Transformer.from_crs(
            source_crs,
            metric_crs,
            always_xy=True
        )

        self.cum_dist = route["shape_dist_traveled"].to_numpy(float)

        self.xy = np.array([
            self.transformer.transform(lon, lat)
            for lon, lat in zip(
                route["shape_pt_lon"],
                route["shape_pt_lat"]
            )
        ])

        self.line = LineString(self.xy)

        seg_lengths = np.sqrt(
            np.sum(
                np.diff(self.xy, axis=0) ** 2,
                axis=1
            )
        )

        self.geom_cum = np.concatenate([
            [0.0],
            np.cumsum(seg_lengths)
        ])

    def project_point(self, lon, lat):
        """
        Returns:
            distance_along_route
            offset_from_route
        """

        x, y = self.transformer.transform(lon, lat)
        p = Point(x, y)

        # distance along geometry
        s = self.line.project(p)

        # perpendicular distance to route
        offset = p.distance(self.line)

        # find segment
        idx = np.searchsorted(
            self.geom_cum,
            s,
            side="right"
        ) - 1

        idx = np.clip(idx, 0, len(self.cum_dist) - 2)

        seg_start = self.geom_cum[idx]
        seg_end = self.geom_cum[idx + 1]

        if seg_end == seg_start:
            route_dist = self.cum_dist[idx]
        else:
            t = (s - seg_start) / (seg_end - seg_start)

            route_dist = (
                self.cum_dist[idx]
                + t * (
                    self.cum_dist[idx + 1]
                    - self.cum_dist[idx]
                )
            )

        return route_dist, offset

    def route_distance(self, lat1, lon1, lat2, lon2, verbose=False):
        d1, off1 = self.project_point(lon1, lat1)
        d2, off2 = self.project_point(lon2, lat2)

        if verbose:
            return {
                "distance_along_route": abs(d2 - d1),
                "point1_chainage": d1,
                "point2_chainage": d2,
                "point1_offset": off1,
                "point2_offset": off2
            }
        else:
            return abs(d2 - d1)