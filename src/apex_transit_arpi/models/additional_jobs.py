# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2026, CIMA+
# All rights reserved.

import datetime as dt
import threading
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel

from apex_transit_arpi.models.analysis import AnalysisJob, AnalysisResult
from apex_transit_arpi.models.transit import TransitAgency


class AdditionalJobsName(StrEnum):
    PREDICTION_GENERATION = "prediction_generation"


class TaskResponseAdditionalJobs(BaseModel):
    """Response model for a started analysis task with multiple additional jobs.

    Returns a dict mapping each additional job name to its task token.
    """

    additional_jobs: dict[AdditionalJobsName, str]  # (additional_job_name: additional_job_token)


class AdditionalJobService(Protocol):
    """Interface that every additional-job service must satisfy.

    Any class exposing generate_days and find_missing_dates with these
    signatures is automatically considered a valid AdditionalJobService.
    """

    def generate_days(
        self,
        agency: TransitAgency,
        days: list[dt.date],
        job: AnalysisJob | None,
        cancel_event: threading.Event | None,
    ) -> AnalysisResult:
        """Compute and persist data for the given list of days."""
        ...

    def find_missing_dates(self, agency: TransitAgency) -> list[dt.date]:
        """Return dates that have no processed data for the given agency."""
        ...
