"""Pytest tests proving the legacy eta_benchmark_analysis pipeline works end-to-end."""

import datetime as dt
import io
import zipfile
from unittest.mock import MagicMock

import polars as pl
from google.transit import gtfs_realtime_pb2

from apex_transit_arpi.analyzer import BenchmarkOrchestrator, DailyAnalysisStatus
from apex_transit_arpi.analyzer.eta_benchmark_analysis.benchmark_builder import build as benchmark_build
from apex_transit_arpi.analyzer.gtfs.duckdb import GtfsRtFetchService
from apex_transit_arpi.models.transit import FeedType, TimePeriod, TransitAgency

# ── epoch constants ────────────────────────────────────────────────────────────
# 2024-01-15 15:00:00 UTC  ==  2024-01-15 10:00:00 EST (Toronto, naive)
_T = 1705330800
# Prediction horizon = 300 s → bucket "3-6" (180–360 s), perfectly on-time (error = 0)
_T_PLUS_300 = _T + 300
_DAY = dt.date(2024, 1, 15)
_AGENCY = TransitAgency.BRAMPTON


# ── protobuf factories ─────────────────────────────────────────────────────────


def _vp_bytes() -> bytes:
    """Single vehicle TRIP_A at STOP_1 observed at T+300."""
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.timestamp = _T_PLUS_300
    e = feed.entity.add()
    e.id = "1"
    e.vehicle.trip.trip_id = "TRIP_A"
    e.vehicle.vehicle.id = "VEH_1"
    e.vehicle.stop_id = "STOP_1"
    e.vehicle.timestamp = _T_PLUS_300
    return feed.SerializeToString()


def _tu_bytes() -> bytes:
    """Single TripUpdate: TRIP_A predicted at STOP_1 to arrive at T+300, feed timestamp T."""
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.timestamp = _T
    e = feed.entity.add()
    e.id = "1"
    tu = e.trip_update
    tu.trip.trip_id = "TRIP_A"
    tu.trip.route_id = "R1"
    tu.timestamp = _T
    stu = tu.stop_time_update.add()
    stu.stop_id = "STOP_1"
    stu.arrival.time = _T_PLUS_300
    return feed.SerializeToString()


def _vp_bytes_seq() -> bytes:
    """VP feed: TRIP_A at current_stop_sequence=1, no stop_id (forces FROM_STOPSEQUENCE strategy)."""
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.timestamp = _T_PLUS_300
    e = feed.entity.add()
    e.id = "1"
    e.vehicle.trip.trip_id = "TRIP_A"
    e.vehicle.vehicle.id = "VEH_1"
    e.vehicle.current_stop_sequence = 1
    e.vehicle.timestamp = _T_PLUS_300
    return feed.SerializeToString()


def _gtfs_zip_bytes() -> bytes:
    """Minimal GTFS zip with stop_times.txt mapping (TRIP_A, seq=1) → STOP_1."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "stop_times.txt",
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n" "TRIP_A,STOP_1,1,10:05:00,10:05:00\n",
        )
    return buf.getvalue()


# ── mock helpers ───────────────────────────────────────────────────────────────


def _mock_clients():
    """Return (file_client, gtfs_fetcher) mocks wired to VP and TU protobuf bytes."""
    vp = _vp_bytes()
    tu = _tu_bytes()

    file_client = MagicMock()
    gtfs_fetcher = MagicMock()

    file_client.count_files_for_date.return_value = 1

    def _by_period(*args, **kwargs):
        # called as (req, ...) positionally or (fetch_request=req, ...)
        fr = args[0] if args else kwargs.get("fetch_request")
        if fr.feed_type == FeedType.VEHICLE_POSITIONS:
            return [{"file_path": "vp.pb"}]
        return [{"file_path": "tu.pb"}]

    def _content(file_path):
        return vp if file_path == "vp.pb" else tu

    file_client.retrieve_file_by_time_period.side_effect = _by_period
    file_client.retrieve_file_content.side_effect = _content
    # load_stop_times_df is called for stop-id strategy detection; empty DF is fine
    # because the VP feed carries stop_id directly → FROM_VEHICLE_POSITION strategy
    gtfs_fetcher.load_stop_times_df.return_value = pl.DataFrame()

    return file_client, gtfs_fetcher


def _make_orchestrator(tmp_path):
    file_client, gtfs_fetcher = _mock_clients()
    # batch_size=1 → single processing worker, avoids the sentinel race condition
    gtfs_rt_fetcher = GtfsRtFetchService(file_client=file_client, gtfs_fetcher=gtfs_fetcher, batch_size=1)
    return BenchmarkOrchestrator(
        file_client=file_client,
        gtfs_fetcher=gtfs_fetcher,
        gtfs_rt_fetcher=gtfs_rt_fetcher,
        output_dir=tmp_path,
    )


# ── tests ──────────────────────────────────────────────────────────────────────


def test_benchmark_builder_accuracy():
    """benchmark_builder.build produces correct error, bucket, and accuracy with no I/O."""
    # Toronto-local naive datetimes matching _T and _T_PLUS_300
    pred_time = dt.datetime(2024, 1, 15, 10, 0, 0)  # T
    pred_arrival = dt.datetime(2024, 1, 15, 10, 5, 0)  # T+300
    actual_arrival = dt.datetime(2024, 1, 15, 10, 5, 0)  # exactly on time

    predictions = pl.DataFrame(
        {
            "trip_id": ["TRIP_A"],
            "stop_id": ["STOP_1"],
            "pred_time": [pred_time],
            "pred_arrival": [pred_arrival],
            "route_id": ["R1"],
            "prediction_horizon_sec": [300.0],
        }
    )
    actuals = pl.DataFrame(
        {
            "trip_id": ["TRIP_A"],
            "stop_id": ["STOP_1"],
            "actual_arrival": [actual_arrival],
            "transit_agency": ["BRAMPTON"],
        }
    )
    period = TimePeriod(
        start_time=dt.datetime(2024, 1, 15, 0, 0, 0),
        end_time=dt.datetime(2024, 1, 15, 23, 59, 59),
    )

    result = benchmark_build(predictions, actuals, "BRAMPTON", period)

    assert result.df.height == 1
    assert result.statistics["total_predictions"] == 1
    assert result.statistics["merged_predictions"] == 1
    assert result.statistics["benchmark_predictions"] == 1

    row = result.df.row(0, named=True)
    assert row["trip_id"] == "TRIP_A"
    assert row["stop_id"] == "STOP_1"
    assert row["route_id"] == "R1"
    assert row["time_bucket"] == "3-6"  # horizon 300 s is in 180–360 s range
    assert row["error_sec"] == 0.0  # predicted == actual
    assert row["is_accurate"] is True  # "3-6" tolerance: early=-60, late=150


def test_benchmark_day_success(tmp_path):
    """benchmark_day returns SUCCESS and one benchmark row with the expected field values."""
    orchestrator = _make_orchestrator(tmp_path)

    status, result_df = orchestrator.benchmark_day(_AGENCY, _DAY)

    assert status == DailyAnalysisStatus.SUCCESS
    assert result_df.height == 1

    row = result_df.row(0, named=True)
    assert row["trip_id"] == "TRIP_A"
    assert row["stop_id"] == "STOP_1"
    assert row["route_id"] == "R1"
    assert row["time_bucket"] == "3-6"
    assert row["error_sec"] == 0.0
    assert row["is_accurate"] is True
    assert row["transit_agency"] == "BRAMPTON"


def test_benchmark_day_writes_parquet(tmp_path):
    """benchmark_period writes the result to output_dir/{agency}/{yyyy-mm}/{day}.analysis.parquet."""
    orchestrator = _make_orchestrator(tmp_path)
    period = TimePeriod(
        start_time=dt.datetime(2024, 1, 15, 0, 0, 0),
        end_time=dt.datetime(2024, 1, 15, 23, 59, 59),
    )

    orchestrator.benchmark_period(_AGENCY, period)

    parquet_path = tmp_path / "BRAMPTON" / "2024-01" / "2024-01-15.analysis.parquet"
    assert parquet_path.exists()

    saved = pl.read_parquet(parquet_path)
    assert saved.height == 1
    assert saved["trip_id"][0] == "TRIP_A"
    assert saved["is_accurate"][0] is True


def test_benchmark_period_summary(tmp_path):
    """benchmark_period processes one day and returns a summary with days_processed=1."""
    orchestrator = _make_orchestrator(tmp_path)
    period = TimePeriod(
        start_time=dt.datetime(2024, 1, 15, 0, 0, 0),
        end_time=dt.datetime(2024, 1, 15, 23, 59, 59),
    )

    summary = orchestrator.benchmark_period(_AGENCY, period)

    assert summary["days_processed"] == 1
    assert summary["days_skipped"] == 0
    assert summary["days_failed"] == 0
    assert summary["errors"] == []
    assert summary["execution_time"] != ""


def test_benchmark_day_skip_when_no_files(tmp_path):
    """benchmark_day returns SKIP when no trip-update files are available for the day."""
    file_client, gtfs_fetcher = _mock_clients()
    file_client.count_files_for_date.return_value = 0

    gtfs_rt_fetcher = GtfsRtFetchService(file_client=file_client, gtfs_fetcher=gtfs_fetcher)
    orchestrator = BenchmarkOrchestrator(
        file_client=file_client,
        gtfs_fetcher=gtfs_fetcher,
        gtfs_rt_fetcher=gtfs_rt_fetcher,
        output_dir=tmp_path,
    )

    status, df = orchestrator.benchmark_day(_AGENCY, _DAY)

    assert status == DailyAnalysisStatus.SKIP
    assert df.is_empty()


def test_process_trip_updates_deduplication(tmp_path):
    """process_trip_updates deduplicates across files: two files with the same prediction produce one result row."""
    vp = _vp_bytes()
    tu = _tu_bytes()

    file_client = MagicMock()
    gtfs_fetcher = MagicMock()
    gtfs_fetcher.load_stop_times_df.return_value = pl.DataFrame()
    file_client.count_files_for_date.return_value = 2

    def _by_period(*args, **kwargs):
        fr = args[0] if args else kwargs.get("fetch_request")
        if fr.feed_type == FeedType.VEHICLE_POSITIONS:
            return [{"file_path": "vp.pb"}]
        # Two TU files with identical content → same (trip_id, stop_id, pred_time)
        return [{"file_path": "tu1.pb"}, {"file_path": "tu2.pb"}]

    file_client.retrieve_file_by_time_period.side_effect = _by_period
    file_client.retrieve_file_content.side_effect = lambda p: vp if p == "vp.pb" else tu

    gtfs_rt_fetcher = GtfsRtFetchService(file_client=file_client, gtfs_fetcher=gtfs_fetcher)
    orchestrator = BenchmarkOrchestrator(
        file_client=file_client,
        gtfs_fetcher=gtfs_fetcher,
        gtfs_rt_fetcher=gtfs_rt_fetcher,
        output_dir=tmp_path,
    )

    status, result_df = orchestrator.benchmark_day(_AGENCY, _DAY)

    assert status == DailyAnalysisStatus.SUCCESS
    # Two files with the same prediction must collapse to exactly one benchmark row.
    assert result_df.height == 1
    row = result_df.row(0, named=True)
    assert row["trip_id"] == "TRIP_A"
    assert row["stop_id"] == "STOP_1"
    assert row["is_accurate"] is True


def test_duckdb_refactor_matches_threadpool_single_file(tmp_path):
    """DuckDB implementation produces bit-for-bit identical output to the original thread-pool version (single file)."""
    from apex_transit_arpi.analyzer.gtfs.legacy import GtfsRtFetchService as GtfsRtFetchServiceBak
    from apex_transit_arpi.analyzer.gtfs.legacy import GtfsRtFetchService as GtfsRtFetchServiceZero

    def _make_mocks():
        file_client = MagicMock()
        gtfs_fetcher = MagicMock()
        gtfs_fetcher.load_stop_times_df.return_value = pl.DataFrame()
        file_client.count_files_for_date.return_value = 1

        def _by_period(*args, **kwargs):
            fr = args[0] if args else kwargs.get("fetch_request")
            return [{"file_path": "vp.pb"}] if fr.feed_type == FeedType.VEHICLE_POSITIONS else [{"file_path": "tu.pb"}]

        file_client.retrieve_file_by_time_period.side_effect = _by_period
        file_client.retrieve_file_content.side_effect = lambda p: _vp_bytes() if p == "vp.pb" else _tu_bytes()
        return file_client, gtfs_fetcher

    # New (DuckDB) implementation
    fc_new, gf_new = _make_mocks()
    gtfs_rt_new = GtfsRtFetchService(file_client=fc_new, gtfs_fetcher=gf_new)
    orch_new = BenchmarkOrchestrator(file_client=fc_new, gtfs_fetcher=gf_new, gtfs_rt_fetcher=gtfs_rt_new, output_dir=tmp_path / "new")
    status_new, df_new = orch_new.benchmark_day(_AGENCY, _DAY)

    # Zero (sequential, Polars dedup) implementation
    fc_zero, gf_zero = _make_mocks()
    gtfs_rt_zero = GtfsRtFetchServiceZero(file_client=fc_zero, gtfs_fetcher=gf_zero)
    orch_zero = BenchmarkOrchestrator(file_client=fc_zero, gtfs_fetcher=gf_zero, gtfs_rt_fetcher=gtfs_rt_zero, output_dir=tmp_path / "zero")
    status_zero, df_zero = orch_zero.benchmark_day(_AGENCY, _DAY)

    # Old (thread-pool) implementation — batch_size=1 for deterministic ordering
    fc_old, gf_old = _make_mocks()
    gtfs_rt_old = GtfsRtFetchServiceBak(file_client=fc_old, gtfs_fetcher=gf_old, batch_size=1)
    orch_old = BenchmarkOrchestrator(file_client=fc_old, gtfs_fetcher=gf_old, gtfs_rt_fetcher=gtfs_rt_old, output_dir=tmp_path / "old")
    status_old, df_old = orch_old.benchmark_day(_AGENCY, _DAY)

    assert status_new == status_zero == status_old == DailyAnalysisStatus.SUCCESS

    sort_cols = ["trip_id", "stop_id", "pred_time"]
    assert df_new.sort(sort_cols).equals(df_old.sort(sort_cols)), f"DuckDB vs thread-pool diverged (single file):\nnew:\n{df_new}\n\nold:\n{df_old}"
    assert df_zero.sort(sort_cols).equals(
        df_old.sort(sort_cols)
    ), f"zero vs thread-pool diverged (single file):\nzero:\n{df_zero}\n\nold:\n{df_old}"


def test_duckdb_refactor_matches_threadpool_with_deduplication(tmp_path):
    """DuckDB implementation produces bit-for-bit identical output to the original thread-pool version (3 duplicate files)."""
    from apex_transit_arpi.analyzer.gtfs.legacy import GtfsRtFetchService as GtfsRtFetchServiceBak
    from apex_transit_arpi.analyzer.gtfs.legacy import GtfsRtFetchService as GtfsRtFetchServiceZero

    def _make_mocks():
        file_client = MagicMock()
        gtfs_fetcher = MagicMock()
        gtfs_fetcher.load_stop_times_df.return_value = pl.DataFrame()
        file_client.count_files_for_date.return_value = 3

        def _by_period(*args, **kwargs):
            fr = args[0] if args else kwargs.get("fetch_request")
            if fr.feed_type == FeedType.VEHICLE_POSITIONS:
                return [{"file_path": "vp.pb"}]
            # Three files with identical content → cross-file deduplication exercised
            return [{"file_path": "tu1.pb"}, {"file_path": "tu2.pb"}, {"file_path": "tu3.pb"}]

        file_client.retrieve_file_by_time_period.side_effect = _by_period
        file_client.retrieve_file_content.side_effect = lambda p: _vp_bytes() if p == "vp.pb" else _tu_bytes()
        return file_client, gtfs_fetcher

    # New (DuckDB) implementation
    fc_new, gf_new = _make_mocks()
    gtfs_rt_new = GtfsRtFetchService(file_client=fc_new, gtfs_fetcher=gf_new)
    orch_new = BenchmarkOrchestrator(file_client=fc_new, gtfs_fetcher=gf_new, gtfs_rt_fetcher=gtfs_rt_new, output_dir=tmp_path / "new")
    status_new, df_new = orch_new.benchmark_day(_AGENCY, _DAY)

    # Zero (sequential, Polars dedup) implementation
    fc_zero, gf_zero = _make_mocks()
    gtfs_rt_zero = GtfsRtFetchServiceZero(file_client=fc_zero, gtfs_fetcher=gf_zero)
    orch_zero = BenchmarkOrchestrator(file_client=fc_zero, gtfs_fetcher=gf_zero, gtfs_rt_fetcher=gtfs_rt_zero, output_dir=tmp_path / "zero")
    status_zero, df_zero = orch_zero.benchmark_day(_AGENCY, _DAY)

    # Old (thread-pool) implementation — batch_size=1 for deterministic ordering
    fc_old, gf_old = _make_mocks()
    gtfs_rt_old = GtfsRtFetchServiceBak(file_client=fc_old, gtfs_fetcher=gf_old, batch_size=1)
    orch_old = BenchmarkOrchestrator(file_client=fc_old, gtfs_fetcher=gf_old, gtfs_rt_fetcher=gtfs_rt_old, output_dir=tmp_path / "old")
    status_old, df_old = orch_old.benchmark_day(_AGENCY, _DAY)

    assert status_new == status_zero == status_old == DailyAnalysisStatus.SUCCESS

    sort_cols = ["trip_id", "stop_id", "pred_time"]
    assert df_new.sort(sort_cols).equals(
        df_old.sort(sort_cols)
    ), f"DuckDB vs thread-pool diverged (deduplication):\nnew:\n{df_new}\n\nold:\n{df_old}"
    assert df_zero.sort(sort_cols).equals(
        df_old.sort(sort_cols)
    ), f"zero vs thread-pool diverged (deduplication):\nzero:\n{df_zero}\n\nold:\n{df_old}"
