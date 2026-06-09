# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.

"""Dataclass model for a GTFS trips.txt record."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Trip:
    """Represents a single row from a GTFS trips.txt file."""

    route_id: str
    service_id: str
    trip_id: str
    trip_headsign: Optional[str] = None
    direction_id: Optional[int] = None
    block_id: Optional[str] = None
    shape_id: Optional[str] = None
    wheelchair_accessible: Optional[int] = None
