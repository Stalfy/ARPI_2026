# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2026, CIMA+
# All rights reserved.

"""Models used to build the comparison overview metrics."""

from datetime import datetime

from pydantic import BaseModel, Field


class StackedHorizontalBar(BaseModel):
    """Single bar of the stacked horizontal bar graph."""

    agency_name: str
    on_time: float
    late: float
    early: float
    number_of_days: int  # Number of days we're used to compute the metrics for this bar


# --------------------------------------------------------------------------- #
# Main Model for Comparison Overview
# --------------------------------------------------------------------------- #


class AgenciesComparisonMetrics(BaseModel):
    """Top-level metrics for the comparison overview."""

    stacked_horizontal_bars: list[StackedHorizontalBar] = Field(default_factory=list)


class AgenciesComparisonOverview(BaseModel):
    """Complete comparison overview payload returned by the API."""

    generated_at: datetime
    metrics: AgenciesComparisonMetrics
