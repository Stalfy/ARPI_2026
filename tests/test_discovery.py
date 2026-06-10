"""Unit tests for arpi.discovery."""

from datetime import date

from apex_transit_arpi.discovery import (
    _iter_date_dirs_desc,
    _iter_discovery_roots,
    iter_day_dirs,
    iter_pb_files,
)


class TestIterDiscoveryRoots:
    def test_yields_feed_roots_for_existing_operators(self, tmp_path):
        root = tmp_path / "GTFS-RT"
        (root / "agency_a" / "trip_updates").mkdir(parents=True)
        (root / "agency_a" / "vehicle_positions").mkdir(parents=True)
        (root / "agency_b" / "trip_updates").mkdir(parents=True)

        results = list(_iter_discovery_roots(tmp_path, "GTFS-RT", ["trip_updates", "vehicle_positions"]))

        assert len(results) == 3

    def test_operators_sorted_alphabetically(self, tmp_path):
        root = tmp_path / "GTFS-RT"
        (root / "z_agency" / "trip_updates").mkdir(parents=True)
        (root / "a_agency" / "trip_updates").mkdir(parents=True)
        (root / "m_agency" / "trip_updates").mkdir(parents=True)

        results = list(_iter_discovery_roots(tmp_path, "GTFS-RT", ["trip_updates"]))

        assert [r.parent.name for r in results] == ["a_agency", "m_agency", "z_agency"]

    def test_missing_feed_dir_is_skipped(self, tmp_path):
        root = tmp_path / "GTFS-RT"
        (root / "agency_a").mkdir(parents=True)  # operator dir exists, but no feed subdir

        results = list(_iter_discovery_roots(tmp_path, "GTFS-RT", ["trip_updates"]))

        assert results == []

    def test_nonexistent_root_returns_empty(self, tmp_path):
        results = list(_iter_discovery_roots(tmp_path, "NONEXISTENT", ["trip_updates"]))

        assert results == []

    def test_resolves_relative_root_against_base_path(self, tmp_path):
        root = tmp_path / "GTFS-RT"
        (root / "agency_a" / "trip_updates").mkdir(parents=True)

        results = list(_iter_discovery_roots(tmp_path, "GTFS-RT", ["trip_updates"]))

        assert len(results) == 1
        assert results[0] == root / "agency_a" / "trip_updates"

    def test_absolute_gtfs_rt_root_ignores_base_path(self, tmp_path):
        root = tmp_path / "absolute_root"
        (root / "agency_a" / "trip_updates").mkdir(parents=True)
        other_base = tmp_path / "ignored"

        results = list(_iter_discovery_roots(other_base, root, ["trip_updates"]))

        assert len(results) == 1

    def test_skips_files_at_operator_level(self, tmp_path):
        root = tmp_path / "GTFS-RT"
        root.mkdir(parents=True)
        (root / "not_a_dir.txt").write_bytes(b"")

        results = list(_iter_discovery_roots(tmp_path, "GTFS-RT", ["trip_updates"]))

        assert results == []

    def test_yields_correct_feed_root_path(self, tmp_path):
        root = tmp_path / "GTFS-RT"
        (root / "agency_a" / "trip_updates").mkdir(parents=True)

        results = list(_iter_discovery_roots(tmp_path, "GTFS-RT", ["trip_updates"]))

        assert results[0].name == "trip_updates"
        assert results[0].parent.name == "agency_a"


class TestIterDateDirsDesc:
    def test_yields_day_dirs_newest_first(self, tmp_path):
        (tmp_path / "2024" / "01" / "15").mkdir(parents=True)
        (tmp_path / "2024" / "01" / "10").mkdir(parents=True)
        (tmp_path / "2024" / "02" / "05").mkdir(parents=True)

        today = date(2024, 3, 1)
        results = list(_iter_date_dirs_desc(tmp_path, today))

        assert [d.name for d in results] == ["05", "15", "10"]

    def test_skips_future_months(self, tmp_path):
        (tmp_path / "2024" / "12" / "31").mkdir(parents=True)
        (tmp_path / "2024" / "01" / "01").mkdir(parents=True)

        today = date(2024, 6, 15)
        results = list(_iter_date_dirs_desc(tmp_path, today))

        assert len(results) == 1
        assert results[0].name == "01"

    def test_skips_future_days_in_current_month(self, tmp_path):
        (tmp_path / "2024" / "06" / "20").mkdir(parents=True)
        (tmp_path / "2024" / "06" / "14").mkdir(parents=True)

        today = date(2024, 6, 15)
        results = list(_iter_date_dirs_desc(tmp_path, today))

        assert len(results) == 1
        assert results[0].name == "14"

    def test_includes_today(self, tmp_path):
        (tmp_path / "2024" / "06" / "15").mkdir(parents=True)

        today = date(2024, 6, 15)
        results = list(_iter_date_dirs_desc(tmp_path, today))

        assert len(results) == 1
        assert results[0].name == "15"

    def test_skips_future_years(self, tmp_path):
        (tmp_path / "2025" / "01" / "01").mkdir(parents=True)
        (tmp_path / "2024" / "01" / "01").mkdir(parents=True)

        today = date(2024, 6, 15)
        results = list(_iter_date_dirs_desc(tmp_path, today))

        assert len(results) == 1

    def test_nonexistent_root_returns_empty(self, tmp_path):
        results = list(_iter_date_dirs_desc(tmp_path / "nonexistent", date.today()))

        assert results == []

    def test_skips_non_numeric_dirs(self, tmp_path):
        (tmp_path / "2024" / "01" / "15").mkdir(parents=True)
        (tmp_path / "2024" / "01" / "bad_dir").mkdir(parents=True)
        (tmp_path / "bad_year" / "01" / "15").mkdir(parents=True)

        today = date(2024, 3, 1)
        results = list(_iter_date_dirs_desc(tmp_path, today))

        assert len(results) == 1

    def test_years_sorted_descending(self, tmp_path):
        (tmp_path / "2023" / "12" / "31").mkdir(parents=True)
        (tmp_path / "2022" / "06" / "15").mkdir(parents=True)

        today = date(2024, 1, 1)
        results = list(_iter_date_dirs_desc(tmp_path, today))
        years = [int(d.parent.parent.name) for d in results]

        assert years == [2023, 2022]


class TestIterDayDirs:
    def test_yields_agency_feed_daydir_tuple(self, tmp_path):
        (tmp_path / "GTFS-RT" / "agency_a" / "trip_updates" / "2024" / "01" / "15").mkdir(parents=True)

        today = date(2024, 3, 1)
        results = list(iter_day_dirs(base_dir=tmp_path, today=today))

        assert len(results) == 1
        agency, feed_name, day_dir = results[0]
        assert agency == "agency_a"
        assert feed_name == "trip_updates"
        assert day_dir.name == "15"

    def test_excludes_today(self, tmp_path):
        today = date(2024, 6, 15)
        (tmp_path / "GTFS-RT" / "agency_a" / "trip_updates" / "2024" / "06" / "15").mkdir(parents=True)
        (tmp_path / "GTFS-RT" / "agency_a" / "trip_updates" / "2024" / "06" / "14").mkdir(parents=True)

        # iter_day_dirs uses current_day = today - 1 = 2024-06-14 as its cutoff,
        # so day 15 (> 14) is skipped; only day 14 is yielded.
        results = list(iter_day_dirs(base_dir=tmp_path, today=today))

        assert len(results) == 1
        assert results[0][2].name == "14"

    def test_multiple_agencies_sorted_alphabetically(self, tmp_path):
        for agency in ["z_agency", "a_agency"]:
            (tmp_path / "GTFS-RT" / agency / "trip_updates" / "2024" / "01" / "10").mkdir(parents=True)

        today = date(2024, 3, 1)
        results = list(iter_day_dirs(base_dir=tmp_path, today=today))

        assert [r[0] for r in results] == ["a_agency", "z_agency"]

    def test_returns_empty_when_no_dirs(self, tmp_path):
        results = list(iter_day_dirs(base_dir=tmp_path))

        assert results == []

    def test_custom_feed_names(self, tmp_path):
        (tmp_path / "GTFS-RT" / "agency_a" / "custom_feed" / "2024" / "01" / "10").mkdir(parents=True)

        today = date(2024, 3, 1)
        results = list(iter_day_dirs(base_dir=tmp_path, feed_names=["custom_feed"], today=today))

        assert len(results) == 1
        assert results[0][1] == "custom_feed"


class TestIterPbFiles:
    def test_yields_pb_files_from_day_dir(self, tmp_path):
        day_dir = tmp_path / "GTFS-RT" / "agency_a" / "trip_updates" / "2024" / "01" / "15"
        day_dir.mkdir(parents=True)
        (day_dir / "file1.pb").write_bytes(b"")
        (day_dir / "file2.pb").write_bytes(b"")

        today = date(2024, 3, 1)
        results = list(iter_pb_files(base_dir=tmp_path, today=today))

        assert len(results) == 2
        assert all(p.suffix == ".pb" for p in results)

    def test_skips_non_pb_files(self, tmp_path):
        day_dir = tmp_path / "GTFS-RT" / "agency_a" / "trip_updates" / "2024" / "01" / "15"
        day_dir.mkdir(parents=True)
        (day_dir / "data.txt").write_bytes(b"")
        (day_dir / "data.pb").write_bytes(b"")

        today = date(2024, 3, 1)
        results = list(iter_pb_files(base_dir=tmp_path, today=today))

        assert len(results) == 1
        assert results[0].name == "data.pb"

    def test_returns_empty_when_no_dirs(self, tmp_path):
        results = list(iter_pb_files(base_dir=tmp_path))

        assert results == []

    def test_excludes_today(self, tmp_path):
        today = date(2024, 6, 15)
        today_dir = tmp_path / "GTFS-RT" / "agency_a" / "trip_updates" / "2024" / "06" / "15"
        yesterday_dir = tmp_path / "GTFS-RT" / "agency_a" / "trip_updates" / "2024" / "06" / "14"
        today_dir.mkdir(parents=True)
        yesterday_dir.mkdir(parents=True)
        (today_dir / "feed.pb").write_bytes(b"")
        (yesterday_dir / "feed.pb").write_bytes(b"")

        results = list(iter_pb_files(base_dir=tmp_path, today=today))

        assert len(results) == 1
        assert results[0].parent.name == "14"
