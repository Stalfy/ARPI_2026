# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.

"""Enums for GTFS-RT parsing strategies."""

from enum import StrEnum


class StopIdStrategy(StrEnum):
    """Strategy for determining the stop_id from GTFS-RT vehicle positions."""

    FROM_VEHICLE_POSITION = "from_vehicle_position"
    FROM_STOPSEQUENCE = "from_stop_sequence"
    FROM_GEOLOCALIZATION = "from_geolocalization"
