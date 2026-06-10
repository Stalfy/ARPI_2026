# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.

"""Dataclass model for a GTFS routes.txt record."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Route:
    """Represents a single row from a GTFS routes.txt file."""

    route_id: str
    route_short_name: str
    route_long_name: str
    route_type: int
    agency_id: Optional[str] = None
    route_desc: Optional[str] = None
    route_url: Optional[str] = None
    route_color: Optional[str] = None
    route_text_color: Optional[str] = None
