# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.

"""Dataclass model for a GTFS stops.txt record."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Stop:
    """Represents a single row from a GTFS stops.txt file."""

    stop_id: str
    stop_name: str
    stop_lat: float
    stop_lon: float
    stop_code: Optional[str] = None
    location_type: Optional[int] = None
    parent_station: Optional[str] = None
    wheelchair_boarding: Optional[int] = None
