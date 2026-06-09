"""Unit tests for arpi.gtfs_mapping."""

import hashlib
import io
import json
import zipfile

import polars as pl

from arpi.discovery.gtfs_mapping import (
    _ensure_cols,
    _find_entry,
    _map_agency,
    _iterate_trips,
    build_mapping,
    run,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_zip(files: dict[str, str]) -> bytes:
    """Build an in-memory zip archive from filename → CSV content pairs."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestFindEntry:
    def test_returns_exact_match(self):
        names = {"agency.txt", "routes.txt"}

        assert _find_entry(names, "agency.txt") == "agency.txt"

    def test_returns_nested_match(self):
        names = {"subdir/agency.txt", "routes.txt"}

        assert _find_entry(names, "agency.txt") == "subdir/agency.txt"

    def test_returns_none_when_not_found(self):
        names = {"routes.txt", "stops.txt"}

        assert _find_entry(names, "agency.txt") is None

    def test_prefers_exact_over_nested(self):
        names = {"agency.txt", "subdir/agency.txt"}

        assert _find_entry(names, "agency.txt") == "agency.txt"

    def test_matches_only_on_slash_boundary(self):
        # "subagency.txt" should NOT match a lookup for "agency.txt"
        names = {"subagency.txt"}

        assert _find_entry(names, "agency.txt") is None


class TestMapAgency:
    def test_returns_id_to_name_mapping(self):
        df = pl.DataFrame(
            {
                "agency_id": ["RTL", "STM"],
                "agency_name": ["Réseau de transport de Longueuil", "Société de transport de Montréal"],
            }
        )

        result = _map_agency(lambda _: df)

        assert result == {
            "RTL": "Réseau de transport de Longueuil",
            "STM": "Société de transport de Montréal",
        }

    def test_returns_empty_dict_for_empty_dataframe(self):
        result = _map_agency(lambda _: pl.DataFrame())

        assert result == {}

    def test_returns_empty_dict_when_agency_id_column_missing(self):
        df = pl.DataFrame({"agency_name": ["Agency A"]})

        result = _map_agency(lambda _: df)

        assert result == {}

    def test_returns_empty_dict_when_agency_name_column_missing(self):
        df = pl.DataFrame({"agency_id": ["RTL"]})

        result = _map_agency(lambda _: df)

        assert result == {}


class TestEnsureCols:
    def test_adds_missing_string_column_with_default(self):
        df = pl.DataFrame({"existing": ["a"]})

        result = _ensure_cols(df, {"new_col": ""})

        assert "new_col" in result.columns
        assert result["new_col"][0] == ""

    def test_adds_missing_int_column_with_default(self):
        df = pl.DataFrame({"existing": ["a"]})

        result = _ensure_cols(df, {"count": 0})

        assert result["count"][0] == 0

    def test_does_not_overwrite_existing_column(self):
        df = pl.DataFrame({"existing": ["original"]})

        result = _ensure_cols(df, {"existing": "default"})

        assert result["existing"][0] == "original"

    def test_adds_multiple_missing_columns(self):
        df = pl.DataFrame({"a": [1]})

        result = _ensure_cols(df, {"b": "", "c": 0})

        assert "b" in result.columns
        assert "c" in result.columns

    def test_returns_unchanged_df_when_all_cols_present(self):
        df = pl.DataFrame({"a": [1], "b": ["x"]})

        result = _ensure_cols(df, {"a": 0, "b": ""})

        assert result.shape == df.shape


class TestPrepareTrips:
    def test_builds_trip_name_from_time_pattern(self):
        trips = pl.DataFrame(
            {
                "trip_id": ["SVC_15:46"],
                "route_id": ["R1"],
                "trip_headsign": ["Downtown"],
            }
        )

        result = list(_iterate_trips(trips))

        # yields (route_id, trip_id, headsign)
        assert result[0][2] == "15:46 - Downtown"

    def test_builds_trip_name_from_dash_split(self):
        trips = pl.DataFrame(
            {
                "trip_id": ["ABC-123"],
                "route_id": ["R1"],
                "trip_headsign": ["Uptown"],
            }
        )

        result = list(_iterate_trips(trips))

        assert result[0][2] == "ABC - Uptown"

    def test_null_headsign_becomes_empty_string_in_name(self):
        trips = pl.DataFrame(
            {
                "trip_id": ["T1"],
                "route_id": ["R1"],
                "trip_headsign": [None],
            }
        )

        result = list(_iterate_trips(trips))

        assert result[0][2] is not None
        assert " - " in result[0][2]

    def test_preserves_trip_id_and_route_id(self):
        trips = pl.DataFrame(
            {
                "trip_id": ["T1"],
                "route_id": ["R1"],
                "trip_headsign": ["Dest"],
            }
        )

        result = list(_iterate_trips(trips))

        assert result[0][1] == "T1"
        assert result[0][0] == "R1"


class TestBuildMapping:
    def test_returns_expected_top_level_keys(self, tmp_path):
        zip_bytes = _make_zip({"agency.txt": "agency_id,agency_name\nRTL,RTL Agency\n"})
        zip_path = tmp_path / "GTFS.zip"
        zip_path.write_bytes(zip_bytes)

        result = build_mapping(zip_path)

        assert set(result.keys()) == {"gtfs_hash", "agency", "routes"}

    def test_gtfs_hash_matches_file_sha256(self, tmp_path):
        zip_bytes = _make_zip({"agency.txt": "agency_id,agency_name\nRTL,RTL\n"})
        zip_path = tmp_path / "GTFS.zip"
        zip_path.write_bytes(zip_bytes)

        result = build_mapping(zip_path)

        assert result["gtfs_hash"] == hashlib.sha256(zip_bytes).hexdigest()

    def test_agency_mapped_correctly(self, tmp_path):
        zip_bytes = _make_zip({"agency.txt": "agency_id,agency_name\nRTL,Réseau de transport de Longueuil\n"})
        zip_path = tmp_path / "GTFS.zip"
        zip_path.write_bytes(zip_bytes)

        result = build_mapping(zip_path)

        assert result["agency"] == {"RTL": "Réseau de transport de Longueuil"}

    def test_routes_built_from_routes_csv(self, tmp_path):
        agency_csv = "agency_id,agency_name\nRTL,RTL\n"
        routes_csv = "route_id,agency_id,route_short_name,route_long_name\n1,RTL,1,Desaulniers\n"
        trips_csv = "route_id,service_id,trip_id,block_id,shape_id,trip_short_name,trip_headsign\n1,SVC1,T1,,,,Downtown\n"
        stop_times_csv = "trip_id,stop_id,stop_sequence\nT1,S1,1\nT1,S2,2\n"
        stops_csv = "stop_id,stop_code,stop_name\nS1,001,Stop One\nS2,002,Stop Two\n"

        zip_bytes = _make_zip(
            {
                "agency.txt": agency_csv,
                "routes.txt": routes_csv,
                "trips.txt": trips_csv,
                "stop_times.txt": stop_times_csv,
                "stops.txt": stops_csv,
            }
        )
        zip_path = tmp_path / "GTFS.zip"
        zip_path.write_bytes(zip_bytes)

        result = build_mapping(zip_path)

        assert "1" in result["routes"]
        assert result["routes"]["1"]["route_name"] == "1 - Desaulniers"
        assert "T1" in result["routes"]["1"]["trips"]

    def test_returns_empty_agency_when_file_absent(self, tmp_path):
        routes_csv = "route_id,agency_id,route_short_name,route_long_name\n1,RTL,1,Line 1\n"
        zip_bytes = _make_zip({"routes.txt": routes_csv})
        zip_path = tmp_path / "GTFS.zip"
        zip_path.write_bytes(zip_bytes)

        result = build_mapping(zip_path)

        assert result["agency"] == {}

    def test_handles_nested_files_in_zip(self, tmp_path):
        agency_csv = "agency_id,agency_name\nSTM,STM Agency\n"
        zip_bytes = _make_zip({"subdir/agency.txt": agency_csv})
        zip_path = tmp_path / "GTFS.zip"
        zip_path.write_bytes(zip_bytes)

        result = build_mapping(zip_path)

        assert result["agency"] == {"STM": "STM Agency"}


    def test_stops_preserve_file_order_when_all_sequences_are_zero(self, tmp_path):
        agency_csv = "agency_id,agency_name\nRTL,RTL\n"
        routes_csv = "route_id,agency_id,route_short_name,route_long_name\n1,RTL,1,Line 1\n"
        trips_csv = "route_id,service_id,trip_id,block_id,shape_id,trip_short_name,trip_headsign\n1,SVC1,T1,,,,Downtown\n"
        # All sequence = 0; file order (S1, S2, S3) should be preserved via stable sort
        stop_times_csv = "trip_id,stop_id,stop_sequence\nT1,S1,0\nT1,S2,0\nT1,S3,0\n"
        stops_csv = "stop_id,stop_code,stop_name\nS1,001,Stop One\nS2,002,Stop Two\nS3,003,Stop Three\n"
        zip_bytes = _make_zip(
            {
                "agency.txt": agency_csv,
                "routes.txt": routes_csv,
                "trips.txt": trips_csv,
                "stop_times.txt": stop_times_csv,
                "stops.txt": stops_csv,
            }
        )
        zip_path = tmp_path / "GTFS.zip"
        zip_path.write_bytes(zip_bytes)

        result = build_mapping(zip_path)

        stops = list(result["routes"]["1"]["trips"]["T1"]["stops"].keys())
        assert stops == ["S1", "S2", "S3"]

    def test_stops_are_ordered_by_stop_sequence(self, tmp_path):
        agency_csv = "agency_id,agency_name\nRTL,RTL\n"
        routes_csv = "route_id,agency_id,route_short_name,route_long_name\n1,RTL,1,Line 1\n"
        trips_csv = "route_id,service_id,trip_id,block_id,shape_id,trip_short_name,trip_headsign\n1,SVC1,T1,,,,Downtown\n"
        # Deliberately out-of-order: sequence 3, 1, 2
        stop_times_csv = "trip_id,stop_id,stop_sequence\nT1,S3,3\nT1,S1,1\nT1,S2,2\n"
        stops_csv = "stop_id,stop_code,stop_name\nS1,001,Stop One\nS2,002,Stop Two\nS3,003,Stop Three\n"
        zip_bytes = _make_zip(
            {
                "agency.txt": agency_csv,
                "routes.txt": routes_csv,
                "trips.txt": trips_csv,
                "stop_times.txt": stop_times_csv,
                "stops.txt": stops_csv,
            }
        )
        zip_path = tmp_path / "GTFS.zip"
        zip_path.write_bytes(zip_bytes)

        result = build_mapping(zip_path)

        stops = list(result["routes"]["1"]["trips"]["T1"]["stops"].keys())
        assert stops == ["S1", "S2", "S3"]


class TestRun:
    def test_writes_valid_parquet_to_output_file(self, tmp_path):
        agency_csv = "agency_id,agency_name\nRTL,Réseau de transport de Longueuil\n"
        routes_csv = "route_id,agency_id,route_short_name,route_long_name\n1,RTL,1,Desaulniers\n"
        trips_csv = "route_id,service_id,trip_id,block_id,shape_id,trip_short_name,trip_headsign\n1,SVC1,T1,,,,Downtown\n"
        stop_times_csv = "trip_id,stop_id,stop_sequence\nT1,S1,1\nT1,S2,2\n"
        stops_csv = "stop_id,stop_code,stop_name\nS1,001,Stop One\nS2,002,Stop Two\n"
        zip_bytes = _make_zip(
            {
                "agency.txt": agency_csv,
                "routes.txt": routes_csv,
                "trips.txt": trips_csv,
                "stop_times.txt": stop_times_csv,
                "stops.txt": stops_csv,
            }
        )

        zip_path = tmp_path / "GTFS.zip"
        zip_path.write_bytes(zip_bytes)

        output_file = tmp_path / "mapping.json.zip"
        run(zip_path, output_file)

        with zipfile.ZipFile(output_file) as zf:
            mapping = json.loads(zf.read("mapping.json"))
        assert mapping["gtfs_hash"]
        assert mapping["routes"]

    def test_output_file_is_utf8_encoded(self, tmp_path):
        # Desaulniers contains a voluntarily placed non-ASCII character to verify UTF-8 encoding is preserved.
        agency_csv = "agency_id,agency_name\nRTL,Réseau de transport de Longueuil\n"
        routes_csv = "route_id,agency_id,route_short_name,route_long_name\n1,RTL,1,Désaulniers\n"
        trips_csv = "route_id,service_id,trip_id,block_id,shape_id,trip_short_name,trip_headsign\n1,SVC1,T1,,,,Downtown\n"
        stop_times_csv = "trip_id,stop_id,stop_sequence\nT1,S1,1\nT1,S2,2\n"
        stops_csv = "stop_id,stop_code,stop_name\nS1,001,Stop One\nS2,002,Stop Two\n"
        zip_bytes = _make_zip(
            {
                "agency.txt": agency_csv,
                "routes.txt": routes_csv,
                "trips.txt": trips_csv,
                "stop_times.txt": stop_times_csv,
                "stops.txt": stops_csv,
            }
        )

        zip_path = tmp_path / "GTFS.zip"
        zip_path.write_bytes(zip_bytes)

        output_file = tmp_path / "mapping.json.zip"
        run(zip_path, output_file)

        with zipfile.ZipFile(output_file) as zf:
            content = zf.read("mapping.json").decode("utf-8")
        assert "Désaulniers" in content

    def test_reuses_cached_file_when_hash_matches(self, tmp_path):
        agency_csv = "agency_id,agency_name\nRTL,RTL\n"
        routes_csv = "route_id,agency_id,route_short_name,route_long_name\n1,RTL,1,Line 1\n"
        trips_csv = "route_id,service_id,trip_id,block_id,shape_id,trip_short_name,trip_headsign\n1,SVC1,T1,,,,Downtown\n"
        stop_times_csv = "trip_id,stop_id,stop_sequence\nT1,S1,1\n"
        stops_csv = "stop_id,stop_code,stop_name\nS1,001,Stop One\n"
        zip_bytes = _make_zip({"agency.txt": agency_csv, "routes.txt": routes_csv, "trips.txt": trips_csv, "stop_times.txt": stop_times_csv, "stops.txt": stops_csv})
        zip_path = tmp_path / "GTFS.zip"
        zip_path.write_bytes(zip_bytes)
        output_file = tmp_path / "mapping.json.zip"

        run(zip_path, output_file)
        mtime_after_first = output_file.stat().st_mtime

        run(zip_path, output_file)
        mtime_after_second = output_file.stat().st_mtime

        assert mtime_after_first == mtime_after_second  # file not rewritten on cache hit

    def test_rebuilds_when_zip_changes(self, tmp_path):
        def _zip(name: str) -> bytes:
            return _make_zip({"agency.txt": f"agency_id,agency_name\nRTL,{name}\n"})

        zip_path = tmp_path / "GTFS.zip"
        output_file = tmp_path / "mapping.json.zip"

        zip_path.write_bytes(_zip("Agency V1"))
        run(zip_path, output_file)
        mtime_v1 = output_file.stat().st_mtime

        zip_path.write_bytes(_zip("Agency V2"))
        run(zip_path, output_file)
        mtime_v2 = output_file.stat().st_mtime

        assert mtime_v2 > mtime_v1  # file was rebuilt


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])