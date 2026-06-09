# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.

"""Types for GTFS mapping and file identifiers."""

from enum import StrEnum
from typing import Dict, TypedDict


class GTFSFile(StrEnum):
    """Constants for common GTFS filenames."""

    AGENCY = "agency.txt"
    ROUTES = "routes.txt"
    TRIPS = "trips.txt"
    STOPS = "stops.txt"
    STOP_TIMES = "stop_times.txt"
    SHAPES = "shapes.txt"


class TripDict(TypedDict):
    """TypedDict for a trip entry in a GTFS mapping."""

    trip_name: str
    stops: Dict[str, str]


class RouteDict(TypedDict):
    """TypedDict for a route entry containing trips mapping."""

    route_name: str
    trips: Dict[str, TripDict]


class GTFSDict(TypedDict):
    """GTFS mapping structure.

    Keys:
        gtfs_hash (str): Snapshot hash used to build the mapping.
        agency (Dict[str, str]): Mapping of agency_id -> agency_name.
        routes (Dict[str, RouteDict]): Mapping of route_id -> RouteDict.

    Example:
        {
            "gtfs_hash": "...",
            "agency": {"RTL": "Réseau de transport de Longueuil"},
            "routes": {
                "1": {
                    "route_name": "Desaulniers / Victoria / Windsor",
                    "trips": {
                        "1_1_R_DI_1601_mtro_15:46": {
                            "trip_name": "Terminus Longueuil 15:46",
                            "stops": {"4416": "Terminus Longueuil"}
                        }
                    }
                }
            }
        }
    """

    gtfs_hash: str
    agency: Dict[str, str]
    routes: Dict[str, RouteDict]
