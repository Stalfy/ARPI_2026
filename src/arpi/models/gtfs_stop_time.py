# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.

"""Dataclass model for a GTFS stop_times.txt record."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StopTime:
    """Represents a single row from a GTFS stop_times.txt file."""

    trip_id: str
    stop_id: str
    stop_sequence: int
    arrival_time: str
    departure_time: str
    stop_headsign: Optional[str] = None
    pickup_type: Optional[int] = None
    drop_off_type: Optional[int] = None
    shape_dist_traveled: Optional[float] = None
    timepoint: Optional[int] = None
