from __future__ import annotations

import logging
import os
import pathlib
import shutil
from pathlib import PurePosixPath
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _get_agency(path: str) -> str | None:
    """
    Expected:
      GTFS/AGENCY/...
      GTFS-RT/AGENCY/trip_updates/yyyy/mm/dd/*.pb
      GTFS-RT/AGENCY/vehicle_positions/yyyy/mm/dd/*.pb
    """
    parts = PurePosixPath(path).parts

    if len(parts) < 2:
        return None

    if parts[0] not in {"GTFS", "GTFS-RT"}:
        return None

    return parts[1]


def filter_gtfs_agencies(input_zip_path: str, output_zip_path: str, agencies: list[str], force: bool = False) -> str:
    if pathlib.Path(output_zip_path).exists() and not force:
        logger.info("Output already exists, skipping: %s", output_zip_path)
        return output_zip_path

    agency_set = set(agencies)
    logger.info("Filtering GTFS and GTFS-RT agencies. agencies=%s", sorted(agency_set))

    kept = 0
    skipped = 0
    with ZipFile(input_zip_path, "r") as zin, ZipFile(output_zip_path, "w", compression=ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zout:
        for info in tqdm(zin.infolist(), desc="Filtering agencies", unit="file"):
            if info.is_dir():
                continue

            agency = _get_agency(info.filename)
            keep = agency in agency_set
            if not keep:
                skipped += 1
                continue

            out_info = ZipInfo(filename=info.filename, date_time=info.date_time)
            out_info.external_attr = info.external_attr
            out_info.comment = info.comment

            with zin.open(info, "r") as src, zout.open(out_info, "w") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)

            kept += 1

    logger.info("Finished agency filtering. kept=%d skipped=%d output=%s", kept, skipped, output_zip_path)
    return output_zip_path
