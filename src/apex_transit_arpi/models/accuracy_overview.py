# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.

"""Models used to build the accuracy dashboard overview and summary metrics."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from apex_transit_arpi.models.transit import TransitAgency


class TimeBucketAccuracy(BaseModel):
    """Accuracy metric for a single time bucket."""

    time_bucket: str
    accuracy: float


class RingDistribution(BaseModel):
    """Distribution of early / on-time / late predictions."""

    time_bucket: str
    early: float
    on_time: float
    late: float


class ScatterPoint(BaseModel):
    """Single point for the scatter plot (time to arrival vs error)."""

    time_to_arrival_min: float
    error_min: float


class MinuteAccuracy(BaseModel):
    """Accuracy for a particular minute of day."""

    minute: int
    accuracy: float


class RouteAccuracy(BaseModel):
    """Accuracy aggregated by route."""

    route_id: str
    accuracy: float


class WeekdayAccuracy(BaseModel):
    """Accuracy aggregated by weekday."""

    weekday: str
    accuracy: float


class StopAccuracy(BaseModel):
    """Accuracy metric for a specific trip stop."""

    stop_id: str
    accuracy: float


# --------------------------------------------------------------------------- #
# Main Model for Dashboard Overview
# --------------------------------------------------------------------------- #


class AccuracyMetrics(BaseModel):
    """Top-level metrics for the dashboard overview."""

    accuracy_global: float
    accuracy_by_bucket: List[TimeBucketAccuracy]
    ring_distribution: RingDistribution
    ring_distribution_by_bucket: List[RingDistribution]


class AccuracyOverview(BaseModel):
    """Complete accuracy overview payload returned by the API."""

    transit_agency: TransitAgency
    generated_at: datetime
    total_processed_predictions: int
    metrics: AccuracyMetrics
    scatter_points: List[ScatterPoint]
    accuracy_by_minute: List[MinuteAccuracy]
    accuracy_by_route: Optional[List[RouteAccuracy]] = Field(default_factory=list)
    accuracy_by_weekday: List[WeekdayAccuracy]
    accuracy_by_stop: Optional[List[StopAccuracy]] = Field(default_factory=list)
