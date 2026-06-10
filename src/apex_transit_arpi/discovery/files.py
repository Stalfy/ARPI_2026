# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.

"""Local filesystem client used by the legacy analyzer.

Provides utilities to upload and retrieve GTFS and GTFS-RT files, compute
hashes and list available files/dates.
"""

import os
import threading
from datetime import date, datetime
from typing import Optional

from apex_transit_arpi.log import ApplicationLogger
from apex_transit_arpi.models.transit import FeedType, FetchRequest, TransitAgency


def _extract_timestamp_from_path(object_path: str) -> Optional[datetime]:
    """Extract timestamp from a file path."""
    try:
        file_name = object_path.split(os.path.sep)[-1]
        parts = file_name.split("_")
        timestamp_str = f"{parts[0]}_{parts[1]}"  # "YYYYMMDD_HHMMSS"
        return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    except (ValueError, IndexError):
        return None


class FileClient:
    """Reads and writes GTFS and GTFS-RT files on the local filesystem."""

    def __init__(self, local_path: Optional[str] = None) -> None:
        """Initialize the file client with a base directory."""
        self._logger = ApplicationLogger(__class__.__name__)
        self.local_path = local_path or (os.path.sep + os.path.join("opt", "arpi"))

    def retrieve_file_by_time_period(self, fetch_request: FetchRequest, cancel_event: threading.Event | None = None) -> list[dict]:
        """Retrieve GTFS-RT files for an agency and feed type within a time period.

        Args:
            fetch_request (FetchRequest): Contains transit agency, feed type, and time period.

        Returns:
            list[dict]: A list of dicts with keys `file_path` (str) and `timestamp` (datetime).
        """
        files = []
        try:
            time_period = fetch_request.time_period
            prefix = os.path.join(
                "GTFS-RT",
                f"{fetch_request.transit_agency}",
                f"{fetch_request.feed_type}",
                f"{time_period.start_time.year}",
                f"{time_period.start_time.month:02d}",
            )

            offset = len(self.local_path) + len(os.path.sep)
            for root, _, files_iter in os.walk(os.path.join(f"{self.local_path}", prefix)):
                for file in files_iter:
                    if cancel_event and cancel_event.is_set():
                        self._logger.inf("Cancellation signal received. Stopping file retrieval.")
                        return files
                    try:
                        file_timestamp = _extract_timestamp_from_path(file)
                        if file_timestamp and time_period.start_time <= file_timestamp <= time_period.end_time:
                            files.append({"file_path": os.path.join(root[offset:], file), "timestamp": file_timestamp})

                    except Exception as e:
                        self._logger.wrn(f"Could not parse timestamp from file '{file}': {e}")
                        continue

            files.sort(key=lambda x: x["timestamp"])
            summary = {
                "Agency": fetch_request.transit_agency,
                "Feed": fetch_request.feed_type.value,
                "Files": len(files),
                "Start": str(time_period.start_time),
                "End": str(time_period.end_time),
            }
            self._logger.inf(f"{summary}")

            return files
        except Exception as e:
            self._logger.err(f"Error retrieving files: {e}")
            return []

    def retrieve_file_content(self, file_path: str) -> bytes | None:
        """Retrieve content of a GTFS-RT Protobuf file from GCS.

        Args:
            file_path (str): Full path to the object in the bucket.

        Returns:
            bytes | None: File content as bytes, or None on error.
        """
        try:
            # content = self.bucket.blob(file_path).download_as_bytes()
            # return content
            file_bytes = None
            with open(os.path.join(f"{self.local_path}", file_path), "rb") as f:
                file_bytes = f.read()

            return file_bytes
        except Exception as e:
            self._logger.err(f"Error retrieving content for '{file_path}': {e}")
            return None

    def get_available_dates(self, transit_agency: TransitAgency) -> list[date]:
        """Get dates for which both trip_updates and vehicle_positions exist for an agency.

        Args:
            transit_agency (TransitAgency): The transit agency to check.

        Returns:
            list[date]: Sorted list of available dates (most recent first).
        """
        self._logger.inf(f"Retrieving available dates for {transit_agency}...")
        try:
            trip_updates_dates = self._get_available_dates_by_feed_type(transit_agency, FeedType.TRIP_UPDATES)
            vehicle_positions_dates = self._get_available_dates_by_feed_type(transit_agency, FeedType.VEHICLE_POSITIONS)

            common_dates = trip_updates_dates & vehicle_positions_dates

            self._logger.inf(f"Retrieved {len(common_dates)} available dates for {transit_agency}.")
            return sorted(common_dates, reverse=True)
        except Exception as e:
            self._logger.err(f"Error retrieving available dates for {transit_agency}: {e}")  # noqa: E501
            return []

    def _get_available_dates_by_feed_type(self, transit_agency: TransitAgency, feed_type: FeedType) -> set[date]:
        dates = set()
        prefix = os.path.join(self.local_path, "GTFS-RT", transit_agency, feed_type.value)

        if not os.path.isdir(prefix):
            return dates

        for year in os.listdir(prefix):
            year_path = os.path.join(prefix, year)
            if not os.path.isdir(year_path):
                continue

            for month in os.listdir(year_path):
                month_path = os.path.join(year_path, month)
                if not os.path.isdir(month_path):
                    continue

                for day in os.listdir(month_path):
                    day_path = os.path.join(month_path, day, ".parquet")
                    if not os.path.exists(day_path) or os.path.isdir(day_path):
                        continue

                    try:
                        dates.add(date(int(year), int(month), int(day)))
                    except ValueError:
                        continue

        return dates

    def count_files_for_date(
        self, transit_agency: TransitAgency, feed_type: FeedType, target_date: date, cancel_event: threading.Event | None = None
    ) -> int:
        """Count GTFS-RT files for a transit agency, feed type, and date.

        Args:
            transit_agency: The transit agency to check.
            feed_type: The feed type (trip_updates or vehicle_positions).
            target_date: The date to count files for.

        Returns:
            int: Number of files for that day, or 0 on error.
        """
        try:
            prefix = os.path.join(
                f"{self.local_path}",
                "GTFS-RT",
                transit_agency,
                feed_type.value,
                f"{target_date.year}",
                f"{target_date.month:02d}",
                f"{target_date.day:02d}",
            )
            count = 0
            for _, _, files in os.walk(prefix):
                if cancel_event and cancel_event.is_set():
                    self._logger.inf("Cancellation signal received. Stopping file counting.")
                    return count
                count = len(files)
                break

            return count
        except Exception as e:
            self._logger.err(f"Error counting files for {os.path.join(transit_agency, feed_type.value)} on {target_date}: {e}")
            return 0
