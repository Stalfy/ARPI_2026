"""Unit tests for arpi.main helpers."""

import zipfile
from datetime import date, datetime
from unittest.mock import MagicMock

import polars as pl

from apex_transit_arpi.main import _diff_day, _step_legacy, _zip_gtfs


class TestZipGtfs:
    def test_creates_zip_file(self, tmp_path):
        src = tmp_path / "GTFS"
        src.mkdir()
        (src / "agency.txt").write_text("agency_id,agency_name\nRTL,RTL\n")
        out = tmp_path / "GTFS.zip"

        _zip_gtfs(src, out)

        assert out.exists()

    def test_zip_contains_all_source_files(self, tmp_path):
        src = tmp_path / "GTFS"
        src.mkdir()
        (src / "agency.txt").write_text("agency_id\nRTL\n")
        (src / "routes.txt").write_text("route_id\n1\n")
        out = tmp_path / "GTFS.zip"

        _zip_gtfs(src, out)

        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
        assert "agency.txt" in names
        assert "routes.txt" in names

    def test_paths_inside_zip_are_relative_to_source(self, tmp_path):
        src = tmp_path / "GTFS"
        (src / "subdir").mkdir(parents=True)
        (src / "subdir" / "extra.txt").write_text("data")
        (src / "top.txt").write_text("data")
        out = tmp_path / "GTFS.zip"

        _zip_gtfs(src, out)

        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
        assert "top.txt" in names
        assert "subdir/extra.txt" in names
        # absolute paths must not appear
        assert not any(str(tmp_path) in n for n in names)

    def test_uses_deflate_compression(self, tmp_path):
        src = tmp_path / "GTFS"
        src.mkdir()
        # Use repetitive content so deflate actually compresses it
        (src / "agency.txt").write_text("agency_id\n" + "A" * 1000)
        out = tmp_path / "GTFS.zip"

        _zip_gtfs(src, out)

        with zipfile.ZipFile(out) as zf:
            for info in zf.infolist():
                assert info.compress_type == zipfile.ZIP_DEFLATED

    def test_creates_empty_zip_for_empty_source_dir(self, tmp_path):
        src = tmp_path / "GTFS"
        src.mkdir()
        out = tmp_path / "GTFS.zip"

        _zip_gtfs(src, out)

        assert out.exists()
        with zipfile.ZipFile(out) as zf:
            assert zf.namelist() == []

    def test_file_content_preserved_in_zip(self, tmp_path):
        src = tmp_path / "GTFS"
        src.mkdir()
        content = "agency_id,agency_name\nRTL,Réseau de transport de Longueuil\n"
        (src / "agency.txt").write_text(content, encoding="utf-8")
        out = tmp_path / "GTFS.zip"

        _zip_gtfs(src, out)

        with zipfile.ZipFile(out) as zf:
            extracted = zf.read("agency.txt").decode("utf-8")
        assert extracted == content


# ── Helpers ────────────────────────────────────────────────────────────────────

_ANALYSIS_SCHEMA = {
    "trip_id": pl.String,
    "stop_id": pl.String,
    "pred_time": pl.Datetime("us", None),
    "error_sec": pl.Float64,
    "time_to_arrival_sec": pl.Float64,
    "time_bucket": pl.String,
    "is_accurate": pl.Boolean,
    "transit_agency": pl.String,
    "route_id": pl.String,
}


def _make_analysis_df(n: int = 1) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trip_id": ["TRIP_A"] * n,
            "stop_id": ["S1"] * n,
            "pred_time": [datetime(2024, 1, 1, 10, 0, 0)] * n,
            "error_sec": [5.0] * n,
            "time_to_arrival_sec": [300.0] * n,
            "time_bucket": ["5min"] * n,
            "is_accurate": [True] * n,
            "transit_agency": ["TEST"] * n,
            "route_id": ["R1"] * n,
        }
    )


def _make_orchestrator(status, df: pl.DataFrame | None = None):
    orchestrator = MagicMock()
    orchestrator.benchmark_day.return_value = (status, df if df is not None else pl.DataFrame())
    return orchestrator


# ── _step_legacy ───────────────────────────────────────────────────────────────


class TestStepLegacy:
    def test_always_writes_file_on_skip(self, tmp_path):
        from apex_transit_arpi.analyzer import DailyAnalysisStatus

        out = tmp_path / "out.parquet"
        result = _step_legacy(_make_orchestrator(DailyAnalysisStatus.SKIP), None, date(2024, 1, 1), out)

        assert result is True
        assert out.exists()
        assert pl.read_parquet(out).is_empty()

    def test_always_writes_file_on_fail(self, tmp_path):
        from apex_transit_arpi.analyzer import DailyAnalysisStatus

        out = tmp_path / "out.parquet"
        result = _step_legacy(_make_orchestrator(DailyAnalysisStatus.FAIL), None, date(2024, 1, 1), out)

        assert result is True
        assert out.exists()
        assert pl.read_parquet(out).is_empty()

    def test_always_writes_file_on_exception(self, tmp_path):
        orchestrator = MagicMock()
        orchestrator.benchmark_day.side_effect = RuntimeError("boom")

        out = tmp_path / "out.parquet"
        result = _step_legacy(orchestrator, None, date(2024, 1, 1), out)

        assert result is True
        assert out.exists()
        assert pl.read_parquet(out).is_empty()

    def test_writes_data_on_success(self, tmp_path):
        from apex_transit_arpi.analyzer import DailyAnalysisStatus

        data = _make_analysis_df(3)
        out = tmp_path / "out.parquet"
        result = _step_legacy(_make_orchestrator(DailyAnalysisStatus.SUCCESS, data), None, date(2024, 1, 1), out)

        assert result is True
        assert out.exists()
        assert pl.read_parquet(out).height == 3

    def test_skips_if_output_already_exists(self, tmp_path):
        from apex_transit_arpi.analyzer import DailyAnalysisStatus

        out = tmp_path / "out.parquet"
        _make_analysis_df(1).write_parquet(out)

        orchestrator = _make_orchestrator(DailyAnalysisStatus.SUCCESS, _make_analysis_df(99))
        _step_legacy(orchestrator, None, date(2024, 1, 1), out)

        orchestrator.benchmark_day.assert_not_called()
        assert pl.read_parquet(out).height == 1


# ── _diff_day ─────────────────────────────────────────────────────────────────


class TestDiffDay:
    def test_no_issues_for_identical_files(self, tmp_path):
        df = _make_analysis_df(2)
        a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
        df.write_parquet(a)
        df.write_parquet(b)

        issues = _diff_day("test", a, b)
        assert issues == []

    def test_reports_row_count_mismatch(self, tmp_path):
        a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
        _make_analysis_df(2).write_parquet(a)
        _make_analysis_df(3).write_parquet(b)

        issues = _diff_day("test", a, b)
        assert any("row count" in i for i in issues)

    def test_handles_empty_legacy_without_error(self, tmp_path):
        a = tmp_path / "modern.parquet"
        b = tmp_path / "legacy.parquet"
        _make_analysis_df(2).write_parquet(a)
        pl.DataFrame().write_parquet(b)

        issues = _diff_day("test", a, b)
        assert any("row count" in i for i in issues)

    def test_handles_both_empty_without_error(self, tmp_path):
        a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
        pl.DataFrame().write_parquet(a)
        pl.DataFrame().write_parquet(b)

        issues = _diff_day("test", a, b)
        assert issues == []
