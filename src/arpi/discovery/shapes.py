"""Utilities for working with GTFS shapes using local Cartesian coordinates."""

from math import cos, hypot, radians

from arpi.models.gtfs_segment import ShapeSegment
from arpi.models.gtfs_shape import Shape
from arpi.models.gtfs_stop import Stop


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance in meters between two XY points."""
    return hypot(x2 - x1, y2 - y1)


def lonlat_to_local_meters(lon: float, lat: float, lon0: float, lat0: float) -> tuple[float, float]:
    """Convert lon/lat to local XY meters around a reference point."""
    x = (lon - lon0) * 111320 * cos(radians(lat0))
    y = (lat - lat0) * 111320
    return x, y


def shapes_to_segments(shapes: list[Shape]) -> list[ShapeSegment]:
    """Convert ordered shape points into ordered XY segments."""
    if len(shapes) < 2:
        return []

    shapes = sorted(shapes, key=lambda s: s.shape_pt_sequence)
    lon0 = shapes[0].shape_pt_lon
    lat0 = shapes[0].shape_pt_lat

    shape_id = shapes[0].shape_id
    if any(s.shape_id != shape_id for s in shapes):
        raise ValueError("All shape points must have the same shape_id")

    segments: list[ShapeSegment] = []

    for i in range(len(shapes) - 1):
        origin = shapes[i]
        end = shapes[i + 1]

        origin_x, origin_y = lonlat_to_local_meters(origin.shape_pt_lon, origin.shape_pt_lat, lon0, lat0)
        end_x, end_y = lonlat_to_local_meters(end.shape_pt_lon, end.shape_pt_lat, lon0, lat0)

        length_meters = calculate_distance(origin_x, origin_y, end_x, end_y)

        segments.append(
            ShapeSegment(
                shape_id=shape_id,
                segment_number=i,
                lon0=lon0,
                lat0=lat0,
                origin_x=origin_x,
                origin_y=origin_y,
                end_x=end_x,
                end_y=end_y,
                length_meters=length_meters,
            )
        )

    return segments


def project_point_to_segment(px: float, py: float, segment: ShapeSegment) -> tuple[float, float, float]:
    """Project a point onto a segment in XY space."""
    ax = segment.origin_x
    ay = segment.origin_y
    bx = segment.end_x
    by = segment.end_y

    dx = bx - ax
    dy = by - ay

    if dx == 0 and dy == 0:
        return 0.0, ax, ay

    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t_clamped = max(0.0, min(1.0, t))

    closest_x = ax + t_clamped * dx
    closest_y = ay + t_clamped * dy

    return t_clamped, closest_x, closest_y


def distance_point_to_segment_xy(px: float, py: float, segment: ShapeSegment) -> float:
    """Calculate the shortest Euclidean distance from an XY point to a segment."""
    _, closest_x, closest_y = project_point_to_segment(px, py, segment)
    return calculate_distance(px, py, closest_x, closest_y)


def find_closest_segment_xy(px: float, py: float, segments: list[ShapeSegment]) -> tuple[ShapeSegment, float] | None:
    """Find the closest segment to a given XY point."""
    if not segments:
        return None

    closest_segment: ShapeSegment | None = None
    min_distance = float("inf")

    for segment in segments:
        distance = distance_point_to_segment_xy(px, py, segment)
        if distance < min_distance:
            min_distance = distance
            closest_segment = segment

    return (closest_segment, min_distance) if closest_segment else None


def distance_along_segment_xy(px: float, py: float, segment: ShapeSegment) -> float:
    """Calculate distance along a segment from origin to projected point."""
    t_clamped, _, _ = project_point_to_segment(px, py, segment)
    return t_clamped * segment.length_meters


def distance_from_shape_start(
    distance_on_segment: float,
    segment_number: int,
    segments: list[ShapeSegment],
) -> float:
    """Calculate cumulative distance from shape start to a point on a segment."""
    if not segments or segment_number < 0 or segment_number >= len(segments):
        return 0.0

    cumulative_distance = sum(seg.length_meters for seg in segments[:segment_number])
    return cumulative_distance + distance_on_segment


def calculate_position_on_shape_xy(px: float, py: float, segments: list[ShapeSegment]) -> tuple[float, float] | None:
    """Calculate a point's position along a shape in XY space."""
    if not segments:
        return None

    result = find_closest_segment_xy(px, py, segments)
    if not result:
        return None

    closest_segment, perpendicular_distance = result
    dist_on_segment = distance_along_segment_xy(px, py, closest_segment)
    total_distance = distance_from_shape_start(dist_on_segment, closest_segment.segment_number, segments)

    return total_distance, perpendicular_distance


def calculate_position_on_shape(
    lon: float,
    lat: float,
    segments: list[ShapeSegment],
) -> tuple[float, float] | None:
    """Convert lon/lat to XY, then calculate position along shape."""
    if not segments:
        return None

    lon0 = segments[0].lon0
    lat0 = segments[0].lat0
    px, py = lonlat_to_local_meters(lon, lat, lon0, lat0)
    return calculate_position_on_shape_xy(px, py, segments)


def compare_vehicle_to_stops(
    vehicle_lon: float,
    vehicle_lat: float,
    stops: list[Stop],
    segments: list[ShapeSegment],
) -> dict[str, float] | None:
    """Compare a vehicle's position to stops along a shape."""
    if not segments:
        return None

    vehicle_result = calculate_position_on_shape(vehicle_lon, vehicle_lat, segments)
    if not vehicle_result:
        return None

    vehicle_distance, _ = vehicle_result
    comparisons: dict[str, float] = {}

    for stop in stops:
        stop_result = calculate_position_on_shape(stop.stop_lon, stop.stop_lat, segments)
        if stop_result:
            stop_distance, _ = stop_result
            comparisons[stop.stop_id] = stop_distance - vehicle_distance

    return comparisons
