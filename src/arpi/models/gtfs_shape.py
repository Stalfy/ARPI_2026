# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.

"""Dataclass model for a GTFS shapes.txt record."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Shape:
    """Represents a single row from a GTFS shapes.txt file."""

    shape_id: str
    shape_pt_lat: float
    shape_pt_lon: float
    shape_pt_sequence: int
    shape_dist_traveled: Optional[float] = None
