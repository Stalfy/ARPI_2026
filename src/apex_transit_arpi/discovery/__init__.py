"""GTFS-RT .pb file discovery by operator, feed, and date."""

from __future__ import annotations

from collections.abc import Generator, Iterable
from datetime import date, timedelta
from pathlib import Path

DEFAULT_GTFS_RT_ROOT = Path("GTFS-RT")
DEFAULT_FEED_NAMES = ("trip_updates", "vehicle_positions")


def _iter_discovery_roots(
    base_path: Path,
    gtfs_rt_root: str | Path,
    feed_names: Iterable[str],
) -> Generator[Path, None, None]:
    """Yield each feed root directory found under every operator subdirectory of gtfs_rt_root.

    Resolves gtfs_rt_root relative to base_path when it is not absolute.
    Operator directories are sorted alphabetically. Missing or non-directory
    paths are skipped silently.

    Args:
        base_path: Absolute base directory used to resolve a relative gtfs_rt_root.
        gtfs_rt_root: Root path containing operator subdirectories.
        feed_names: Feed directory names to look for under each operator directory.

    Yields:
        Paths to feed root directories that exist, in alphabetical operator order.
    """
    root_path = Path(gtfs_rt_root)
    if not root_path.is_absolute():
        root_path = base_path / root_path

    if not root_path.exists() or not root_path.is_dir():
        return

    for operator_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
        for feed_name in feed_names:
            feed_root = operator_dir / feed_name
            if feed_root.exists() and feed_root.is_dir():
                yield feed_root


def _iter_date_dirs_desc(root: Path, today: date) -> Generator[Path, None, None]:
    """Yield YYYY/MM/DD day directories under root in descending date order.

    Only directories whose names are purely numeric strings are considered at
    each level. Directories that fall after today are skipped so partially
    written future dates are never surfaced.

    Args:
        root: The feed root directory containing year subdirectories.
        today: Reference date; directories after this date are skipped.

    Yields:
        Paths to day directories, newest first.
    """
    if not root.exists() or not root.is_dir():
        return

    years = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit() and len(path.name) == 4),
        key=lambda path: int(path.name),
        reverse=True,
    )

    for year_dir in years:
        year = int(year_dir.name)
        if year > today.year:
            continue

        months = sorted(
            (path for path in year_dir.iterdir() if path.is_dir() and path.name.isdigit() and len(path.name) in (1, 2)),
            key=lambda path: int(path.name),
            reverse=True,
        )

        for month_dir in months:
            month = int(month_dir.name)
            if year == today.year and month > today.month:
                continue

            days = sorted(
                (path for path in month_dir.iterdir() if path.is_dir() and path.name.isdigit() and len(path.name) in (1, 2)),
                key=lambda path: int(path.name),
                reverse=True,
            )

            for day_dir in days:
                day = int(day_dir.name)
                if year == today.year and month == today.month and day > today.day:
                    continue
                yield day_dir


def iter_day_dirs(
    base_dir: str | Path = ".",
    gtfs_rt_root: str | Path = DEFAULT_GTFS_RT_ROOT,
    feed_names: Iterable[str] = DEFAULT_FEED_NAMES,
    today: date | None = None,
) -> Generator[tuple[str, str, Path], None, None]:
    """Yield (agency, feed_name, day_dir) for every day directory, newest first.

    Resolves gtfs_rt_root relative to base_dir when it is not absolute. Operator
    directories are sorted alphabetically. For each (operator, feed) pair, day
    directories are yielded in descending date order starting from yesterday.

    Args:
        base_dir: Base directory used to resolve a relative gtfs_rt_root.
            Defaults to the current working directory.
        gtfs_rt_root: Root path containing operator subdirectories.
            Defaults to GTFS-RT/.
        feed_names: Feed directory names to look for under each operator.
            Defaults to ("trip_updates", "vehicle_positions").
        today: Reference date for determining yesterday. Defaults to date.today().

    Yields:
        Three-tuples of (agency name, feed directory name, day directory path),
        in alphabetical agency order and descending date order within each agency.
    """
    reference_day = today or date.today()
    current_day = reference_day - timedelta(days=1)
    base_path = Path(base_dir)

    for feed_root in _iter_discovery_roots(base_path, gtfs_rt_root, feed_names):
        agency = feed_root.parent.name
        feed_name = feed_root.name
        for day_dir in _iter_date_dirs_desc(feed_root, current_day):
            yield agency, feed_name, day_dir


def iter_pb_files(
    base_dir: str | Path = ".",
    gtfs_rt_root: str | Path = DEFAULT_GTFS_RT_ROOT,
    feed_names: Iterable[str] = DEFAULT_FEED_NAMES,
    today: date | None = None,
) -> Generator[Path, None, None]:
    """Yield .pb files by scanning GTFS-RT operators and feed roots in reverse time.

    Finds operator folders dynamically under gtfs_rt_root and traverses date
    directories as YYYY/MM/DD starting from yesterday (today - 1 day), in
    descending order. Recursively yields every .pb file found under each
    matching day directory.

    Args:
        base_dir: Base directory used to resolve a relative gtfs_rt_root.
            Defaults to the current working directory.
        gtfs_rt_root: Root path containing operator subdirectories.
            Defaults to GTFS-RT/.
        feed_names: Feed directory names to look for under each operator.
            Defaults to ("trip_updates", "vehicle_positions").
        today: Reference date for determining yesterday. Defaults to date.today().

    Yields:
        Paths to .pb files, newest first across all operators and feeds.
    """
    reference_day = today or date.today()
    current_day = reference_day - timedelta(days=1)
    base_path = Path(base_dir)

    for root_path in _iter_discovery_roots(base_path, gtfs_rt_root, feed_names):
        for day_dir in _iter_date_dirs_desc(root_path, current_day):
            for pb_file in sorted((path for path in day_dir.rglob("*.pb") if path.is_file()), reverse=True):
                yield pb_file
