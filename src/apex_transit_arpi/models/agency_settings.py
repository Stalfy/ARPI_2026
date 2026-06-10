# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.

"""Models for per-agency configuration settings."""

from enum import StrEnum

from pydantic import BaseModel


class VehiclePositionMappingStrategy(StrEnum):
    """Strategy used to map a vehicle position to a stop in the GTFS schedule."""

    AUTOMATIC = "automatic"
    FROM_VEHICLE_POSITION = "from_vehicle_position"
    FROM_STOPSEQUENCE = "from_stop_sequence"
    FROM_GEOLOCALIZATION = "from_geolocalization"


class AgencySettings(BaseModel):
    """Persisted configuration settings for a transit agency.

    Attributes:
        mappingStrategy: The vehicle-position mapping strategy to apply when
            processing GTFS-RT data for this agency.
    """

    mappingStrategy: VehiclePositionMappingStrategy


class UpdateMappingStrategyRequest(BaseModel):
    """Request body for updating the mapping strategy of a transit agency.

    Attributes:
        agency_name: Transit agency identifier (case-insensitive).
        mapping_strategy: The new mapping strategy to apply.
    """

    agency_name: str
    mapping_strategy: VehiclePositionMappingStrategy
