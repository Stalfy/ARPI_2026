# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.
"""Time bucket definitions used for benchmarking ETA accuracy."""

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Bucket:
    """Representation of a single time bucket."""

    name: str
    min: float
    max: float
    early: float
    late: float


class TimeBuckets(Enum):
    """Enumeration for time buckets used in ETA accuracy benchmarking (seconds)."""

    BUCKET_0_180 = Bucket("0-3", 0, 180, -30, 90)
    BUCKET_180_360 = Bucket("3-6", 180, 360, -60, 150)
    BUCKET_360_600 = Bucket("6-10", 360, 600, -60, 210)
    BUCKET_600_900 = Bucket("10-15", 600, 900, -90, 270)

    @classmethod
    def values(cls) -> list[Bucket]:
        """Return the list of Bucket values in enumeration order."""
        return [b.value for b in cls]

    @classmethod
    def bins(cls) -> list[float]:
        """Return the numeric bin boundaries used for grouping horizons."""
        return [
            cls.BUCKET_0_180.value.min,
            cls.BUCKET_180_360.value.min,
            cls.BUCKET_360_600.value.min,
            cls.BUCKET_600_900.value.min,
            cls.BUCKET_600_900.value.max,
        ]

    @classmethod
    def labels(cls) -> list[str]:
        """Return the human-readable labels for each time bucket."""
        return [
            cls.BUCKET_0_180.value.name,
            cls.BUCKET_180_360.value.name,
            cls.BUCKET_360_600.value.name,
            cls.BUCKET_600_900.value.name,
        ]
