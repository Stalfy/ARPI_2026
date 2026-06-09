# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.

"""Dataclass model for a GTFS shape segment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ShapeSegment:
    """Represents a segment between two consecutive points in a GTFS shape."""

    shape_id: str
    segment_number: int
    lon0: float
    lat0: float
    origin_x: float
    origin_y: float
    end_x: float
    end_y: float
    length_meters: float
