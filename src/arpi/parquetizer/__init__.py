"""High-level orchestration for parsing GTFS-RT protobuf files into parquet."""

import threading
from pathlib import Path

from arpi.discovery.gtfs import GtfsFetchService
from arpi.discovery.rt_parser import GtfsRtParser
from arpi.log import ApplicationLogger
from arpi.models.agency_settings import VehiclePositionMappingStrategy
from arpi.models.transit import FetchRequest

from . import trip_updates, vehicle_positions


def run(
    data_type: str,
    folder: Path,
    output_file: Path,
    *,
    fr: FetchRequest | None = None,
    gtfs_fetcher: GtfsFetchService | None = None,
    parser: GtfsRtParser | None = None,
    configured_strategy: VehiclePositionMappingStrategy = VehiclePositionMappingStrategy.AUTOMATIC,
    cancel_event: threading.Event | None = None,
    logger: ApplicationLogger | None = None,
) -> None:
    """Parse all .pb files from folder in timestamp order and export to parquet.

    Both ``"trip_update"`` and ``"vehicle_position"`` require ``fr``,
    ``gtfs_fetcher``, and ``parser`` to be provided; they delegate to their
    respective ``parse_directory`` function which handles GTFS context, stop-id
    resolution, and deduplication before writing a processed parquet.

    Args:
        data_type: Either ``"trip_update"`` or ``"vehicle_position"``.
        folder: Directory containing the .pb files to process.
        output_file: Destination path for the final parquet file.
        fr: FetchRequest (agency + period). Required for both data types.
        gtfs_fetcher: Static GTFS loader. Required for both data types.
        parser: GTFS-RT parser. Required for both data types.
        configured_strategy: Stop-id strategy; forwarded to VP ``parse_directory``.
        cancel_event: Optional cancellation signal; forwarded to ``parse_directory``.
        logger: Logger instance; forwarded to ``parse_directory``.

    Raises:
        ValueError: If ``fr``, ``gtfs_fetcher``, or ``parser`` are not provided.
        KeyError: If ``data_type`` is not ``"trip_update"`` or ``"vehicle_position"``.
    """
    if fr is None or gtfs_fetcher is None or parser is None:
        raise ValueError("fr, gtfs_fetcher, and parser are required.")

    if data_type == "vehicle_position":
        vehicle_positions.parse_directory(
            folder,
            output_file,
            fr,
            gtfs_fetcher,
            parser,
            configured_strategy=configured_strategy,
            cancel_event=cancel_event,
            logger=logger,
        )
        return

    if data_type == "trip_update":
        trip_updates.parse_directory(
            folder,
            output_file,
            fr,
            gtfs_fetcher,
            cancel_event=cancel_event,
            logger=logger,
        )
        return

    raise KeyError(f"Unknown data_type: {data_type!r}. Expected 'trip_update' or 'vehicle_position'.")
