# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2026, CIMA+
# All rights reserved.

from dataclasses import dataclass
from datetime import date

from apex_transit_arpi.models.transit import TransitAgency


@dataclass(frozen=True)
class GenerationRecord:
    """Generation record comparing expected vs actual predictions for a trip/stop/day."""

    transit_agency: TransitAgency
    analysis_date: date
    route_id: str
    trip_id: str
    stop_id: str
    expected_predictions: int
    actual_predictions: int
