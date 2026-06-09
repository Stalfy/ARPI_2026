# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.
"""Models related to analysis results and job tracking.

Contains lightweight dataclasses for Actual/Prediction/Benchmark rows,
and Pydantic models used to serialize analysis jobs and daily statistics.
"""

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel
from typing_extensions import TypedDict

from arpi.models.transit import TransitAgency


@dataclass(frozen=True)
class Benchmark:
    """Benchmark record combining prediction, actual and computed error."""

    transit_agency: TransitAgency
    trip_id: str
    stop_id: str
    pred_time: datetime
    error_sec: int
    time_to_arrival_sec: int
    time_bucket: str
    is_accurate: bool
    route_id: str


class ErrorDetail(TypedDict):
    """Structure for reporting an error that occurred during analysis."""

    date: str
    error: str
    traceback: str


class AnalysisJobStatus(StrEnum):
    """Possible statuses for an analysis job."""

    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL_SUCCESS = "partial_success"


class AnalysisResult(TypedDict):
    """TypedDict representing a summary of an analysis run."""

    status: AnalysisJobStatus
    days_processed: int
    days_skipped: int
    days_failed: int
    start_date: str
    end_date: str
    transit_agency: TransitAgency
    execution_time: str
    errors: list[ErrorDetail]
    daily_stats_ids: Optional[list[int]]

    @staticmethod
    def default_from_dates(agency: TransitAgency, start_date: date, end_date: date):
        return {
            "status": AnalysisJobStatus.SUCCESS,
            "days_processed": 0,
            "days_skipped": 0,
            "days_failed": 0,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "transit_agency": agency,
            "execution_time": "",
            "errors": [],
            "daily_stats_ids": [],
        }


class TaskResponse(BaseModel):
    """Response model for a started analysis task."""

    task_token: str


class AnalysisJobState(StrEnum):
    """Enumeration of possible analysis job states."""

    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELED = "canceled"


class DailyStatistics(BaseModel):
    """Daily statistics persisted for each analysis run."""

    agency_name: TransitAgency
    analysis_date: date
    total_predictions: int
    predictions_horizon_over_15_min: int
    merged_predictions: int
    merged_predictions_over_15_min: int
    benchmark_predictions: int
    benchmark_execution_time_sec: float
    id: Optional[int] = None
    created_at: Optional[datetime] = None


class AnalysisJob(BaseModel):
    """Model representing a persisted analysis job and its metadata."""

    token: str
    agency_name: TransitAgency
    created_at: datetime
    stats: Optional[AnalysisResult]
    state: AnalysisJobState
    daily_stats: Optional[list[DailyStatistics]] = None
