# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.

"""Models used to build the generation dashboard overview and summary metrics."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from arpi.models.transit import TransitAgency


class GenerationSummary(BaseModel):
    """Global summary of prediction generation (available vs expected)."""

    available: int
    expected: int
    rate: float


class DailyGeneration(BaseModel):
    """Generation rate for a single calendar day."""

    date: str
    rate: float


class RouteGeneration(BaseModel):
    """Generation rate aggregated by route."""

    route_id: str
    rate: float


class StopGeneration(BaseModel):
    """Generation rate aggregated by stop."""

    stop_id: str
    rate: float


# --------------------------------------------------------------------------- #
# Main Model for Generation Overview
# --------------------------------------------------------------------------- #


class GenerationMetrics(BaseModel):
    """All generation-related metrics bundled together."""

    summary: GenerationSummary
    generation_by_day: List[DailyGeneration] = Field(default_factory=list)
    generation_by_route: Optional[List[RouteGeneration]] = Field(default_factory=list)
    generation_by_stop: Optional[List[StopGeneration]] = Field(default_factory=list)


class GenerationOverview(BaseModel):
    """Complete generation overview payload returned by the API."""

    transit_agency: TransitAgency
    generated_at: datetime
    metrics: GenerationMetrics
