# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.

"""Dataclass model for a GTFS agency.txt record."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Agency:
    """Represents a single row from a GTFS agency.txt file."""

    agency_id: str
    agency_name: str
    agency_url: str
    agency_timezone: str
    agency_phone: Optional[str] = None
    agency_lang: Optional[str] = None
    agency_fare_url: Optional[str] = None
