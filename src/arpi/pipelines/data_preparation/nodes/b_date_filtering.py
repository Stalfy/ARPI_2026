from __future__ import annotations

import logging
import pathlib
import shutil
from datetime import date
from pathlib import PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from tqdm import tqdm

logger = logging.getLogger(__name__)
RT_TYPES = {"trip_updates", "vehicle_positions"}


def _parse_gtfs_rt_date(path: str) -> date | None:
    """
    Expected:
      GTFS-RT/AGENCY/trip_updates/yyyy/mm/dd/*.pb
      GTFS-RT/AGENCY/vehicle_positions/yyyy/mm/dd/*.pb
    """
    parts = PurePosixPath(path).parts
    if len(parts) < 7:
        return None
    if parts[0] != "GTFS-RT":
        return None
    if parts[2] not in RT_TYPES:
        return None
    try:
        return date(int(parts[3]), int(parts[4]), int(parts[5]))
    except ValueError:
        return None


def _filter(input_zip_path: str, output_zip_path: str, start_date: date, end_date: date) -> None:
    """
    Copies:
      - all GTFS/* files as-is
      - GTFS-RT files only when their path date is within [start_date, end_date]

    Does not extract the archive and does not load files into memory.
    """
    with ZipFile(input_zip_path, "r") as zin, ZipFile(output_zip_path, "w", compression=ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zout:
        for info in tqdm(zin.infolist(), desc="Filtering dates", unit="file"):
            if info.is_dir():
                continue

            path = info.filename
            keep = False

            if path.startswith("GTFS/"):
                keep = True
            elif path.startswith("GTFS-RT/"):
                file_date = _parse_gtfs_rt_date(path)
                keep = file_date is not None and start_date <= file_date <= end_date

            if not keep:
                continue

            # Preserve metadata where practical
            out_info = ZipInfo(filename=info.filename, date_time=info.date_time)
            out_info.external_attr = info.external_attr
            out_info.comment = info.comment

            with zin.open(info, "r") as src, zout.open(out_info, "w") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)


def filter_realtime_dates(input_zip_path: str, output_zip_path: str, start_date: str, end_date: str, force: bool = False) -> str:
    if pathlib.Path(output_zip_path).exists() and not force:
        logger.info("Output already exists, skipping: %s", output_zip_path)
        return output_zip_path

    _filter(
        input_zip_path=input_zip_path,
        output_zip_path=output_zip_path,
        start_date=date.fromisoformat(start_date),
        end_date=date.fromisoformat(end_date),
    )

    return output_zip_path
