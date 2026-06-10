# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.
"""Transit-related models and simple helper types.

Contains enums and dataclasses used to represent transit agency
configurations, time periods and GTFS-RT artifacts.
"""

import datetime
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional


class TransitAgency(StrEnum):
    """Known transit agencies supported by the system."""

    BRAMPTON = "BRAMPTON"
    BURLINGTON = "BURLINGTON"
    CITLA = "CITLA"
    DRT = "DRT"
    GRT = "GRT"
    HAMILTON = "HAMILTON"
    KINGSTON = "KINGSTON"
    LTC = "LTC"
    MISSISSAUGA = "MISSISSAUGA"
    STM = "STM"
    STO = "STO"
    STTR = "STTR"
    RTL = "RTL"
    TRANSLINK = "TRANSLINK"
    YRT = "YRT"


DelayUsingAgencies: set[TransitAgency] = {TransitAgency.RTL, TransitAgency.CITLA, TransitAgency.LTC}


class FeedType(StrEnum):
    """Types of GTFS real-time feeds."""

    TRIP_UPDATES = "trip_updates"
    VEHICLE_POSITIONS = "vehicle_positions"


class APIType(StrEnum):
    """API backends used to fetch GTFS-RT feeds."""

    BRAMPTON = "brampton"
    BURLINGTON = "burlington"
    DRT = "drt"
    GRT = "grt"
    HAMILTON = "hamilton"
    KINGSTON = "kingston"
    LTC = "ltc"
    MISSISSAUGA = "mississauga"
    STM = "stm"
    STO = "sto"
    STTR = "sttr"
    CHRONO = "chrono"
    TRANSLINK = "translink"
    YRT = "yrt"


@dataclass
class TransitAgencyConfig:
    """Configuration for an agency used by acquisition services."""

    agency_name: TransitAgency
    api_type: APIType
    enabled: bool
    api_config: dict


@dataclass
class TransitAgencyData:
    """Runtime view of an agency including display name and last fetch info."""

    agency_name: TransitAgency
    display_name: str
    api_type: APIType
    enabled: bool
    api_config: dict
    feed_url: Optional[str] = None
    last_fetch: Optional[datetime.datetime] = None
    last_error: Optional[str] = None


@dataclass
class TransitAgencyRecord:
    """Database record representation for a transit agency."""

    agency: TransitAgency
    feedUrl: str
    lastFetch: Optional[str]
    active: bool
    lastError: Optional[str] = None


@dataclass
class ToggleFetchResponse:
    """Response payload for enabling/disabling feed collection."""

    active: bool
    lastFetch: Optional[str]


@dataclass
class TimePeriod:
    """A time range used to request or query feeds."""

    start_time: datetime.datetime
    end_time: datetime.datetime

    def __post_init__(self) -> None:
        """Validate that `end_time` is after or equal to `start_time`."""
        if self.end_time < self.start_time:
            raise ValueError(f"end_time ({self.end_time}) must be at or after start_time ({self.start_time})")


@dataclass
class FetchRequest:
    """Request object used when fetching GTFS-RT data for an agency."""

    transit_agency: TransitAgency
    feed_type: FeedType
    time_period: TimePeriod


@dataclass
class GTFSRTFile:
    """Container for a GTFS-RT protobuf payload prepared for storage."""

    transit_agency: TransitAgency
    feed_type: FeedType
    timestamp: datetime.datetime
    content: bytes
