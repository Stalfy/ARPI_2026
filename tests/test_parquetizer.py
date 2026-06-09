"""Unit tests for arpi.parquetizer (helpers, trip_updates, vehicle_positions, __init__)."""

from unittest.mock import MagicMock

import pytest
from google.transit import gtfs_realtime_pb2

from arpi.parquetizer.helpers import attr, hf, parse_header, str_or_ts, ts
from arpi.parquetizer.trip_updates import parse_file as parse_tu_file
from arpi.parquetizer.trip_updates import parse_stop_time_event, parse_stop_time_properties
from arpi.parquetizer.trip_updates import parse_trip_descriptor as parse_tu_trip
from arpi.parquetizer.trip_updates import parse_trip_properties, parse_vehicle
from arpi.parquetizer.vehicle_positions import parse_file as parse_vp_file
from arpi.parquetizer.vehicle_positions import parse_position
from arpi.parquetizer.vehicle_positions import parse_trip_descriptor as parse_vp_trip
from arpi.parquetizer.vehicle_positions import parse_vehicle_descriptor

# ── helpers ─────────────────────────────────────────────────────────────────


class TestHf:
    def test_returns_true_when_singular_message_field_is_set(self):
        tu = gtfs_realtime_pb2.TripUpdate()
        tu.trip.trip_id = "T1"

        assert hf(tu, "trip") is True

    def test_returns_false_when_singular_message_field_not_set(self):
        tu = gtfs_realtime_pb2.TripUpdate()

        assert hf(tu, "vehicle") is False

    def test_returns_false_for_repeated_field(self):
        # HasField raises ValueError for repeated fields; hf must return False
        tu = gtfs_realtime_pb2.TripUpdate()

        assert hf(tu, "stop_time_update") is False


class TestAttr:
    def test_returns_attribute_value(self):
        obj = MagicMock()
        obj.some_field = 42

        assert attr(obj, "some_field") == 42

    def test_returns_none_default_when_attribute_missing(self):
        obj = MagicMock(spec=[])  # spec=[] → no attributes defined

        assert attr(obj, "missing") is None

    def test_returns_custom_default_when_attribute_missing(self):
        obj = MagicMock(spec=[])

        assert attr(obj, "missing", 0) == 0


class TestTs:
    def test_returns_first_translation_text(self):
        msg = gtfs_realtime_pb2.TranslatedString()
        msg.translation.add().text = "Hello"

        assert ts(msg) == "Hello"

    def test_returns_none_for_empty_translation_list(self):
        msg = gtfs_realtime_pb2.TranslatedString()

        assert ts(msg) is None


class TestStrOrTs:
    def test_returns_plain_string_field(self):
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.header.gtfs_realtime_version = "2.0"

        assert str_or_ts(feed.header, "gtfs_realtime_version") == "2.0"

    def test_returns_none_for_empty_string(self):
        # default value of gtfs_realtime_version is ""
        feed = gtfs_realtime_pb2.FeedMessage()

        assert str_or_ts(feed.header, "gtfs_realtime_version") is None

    def test_returns_none_when_attribute_absent(self):
        obj = MagicMock(spec=[])

        assert str_or_ts(obj, "nonexistent") is None

    def test_returns_translated_string_text(self):
        msg = gtfs_realtime_pb2.Alert()
        msg.header_text.translation.add().text = "Service Alert"

        assert str_or_ts(msg, "header_text") == "Service Alert"


class TestParseHeader:
    def test_maps_all_header_fields(self):
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.header.gtfs_realtime_version = "2.0"
        feed.header.timestamp = 1_700_000_000
        feed.header.incrementality = 0

        result = parse_header(feed.header)

        assert result["header.gtfs_realtime_version"] == "2.0"
        assert result["header.timestamp"] == 1_700_000_000
        assert result["header.incrementality"] == 0

    def test_normalizes_empty_version_to_none(self):
        feed = gtfs_realtime_pb2.FeedMessage()  # version not set → ""

        result = parse_header(feed.header)

        assert result["header.gtfs_realtime_version"] is None


# ── trip_updates parsers ─────────────────────────────────────────────────────


class TestParseTuTripDescriptor:
    def test_extracts_trip_fields(self):
        trip = gtfs_realtime_pb2.TripDescriptor()
        trip.trip_id = "T1"
        trip.route_id = "R1"
        trip.start_time = "08:00:00"
        trip.start_date = "20240115"

        result = parse_tu_trip(trip)

        assert result["entity.trip_update.trip.trip_id"] == "T1"
        assert result["entity.trip_update.trip.route_id"] == "R1"
        assert result["entity.trip_update.trip.start_time"] == "08:00:00"
        assert result["entity.trip_update.trip.start_date"] == "20240115"

    def test_empty_strings_normalized_to_none(self):
        trip = gtfs_realtime_pb2.TripDescriptor()  # all fields empty

        result = parse_tu_trip(trip)

        assert result["entity.trip_update.trip.trip_id"] is None
        assert result["entity.trip_update.trip.route_id"] is None

    def test_modified_trip_fields_are_none_when_absent(self):
        trip = gtfs_realtime_pb2.TripDescriptor()

        result = parse_tu_trip(trip)

        assert result["entity.trip_update.trip.modified_trip.service_id"] is None
        assert result["entity.trip_update.trip.modified_trip.trip_modification_id"] is None


class TestParseVehicle:
    def test_returns_none_values_when_vehicle_absent(self):
        tu = gtfs_realtime_pb2.TripUpdate()

        result = parse_vehicle(tu)

        assert result["entity.trip_update.vehicle.id"] is None
        assert result["entity.trip_update.vehicle.label"] is None
        assert result["entity.trip_update.vehicle.license_plate"] is None

    def test_extracts_vehicle_fields(self):
        tu = gtfs_realtime_pb2.TripUpdate()
        tu.vehicle.id = "V1"
        tu.vehicle.label = "Bus 42"
        tu.vehicle.license_plate = "ABC123"

        result = parse_vehicle(tu)

        assert result["entity.trip_update.vehicle.id"] == "V1"
        assert result["entity.trip_update.vehicle.label"] == "Bus 42"
        assert result["entity.trip_update.vehicle.license_plate"] == "ABC123"


class TestParseStopTimeEvent:
    def test_returns_none_values_when_event_absent(self):
        stu = gtfs_realtime_pb2.TripUpdate.StopTimeUpdate()

        result = parse_stop_time_event(stu, "arrival")

        assert result["entity.trip_update.stop_time_update.arrival.delay"] is None
        assert result["entity.trip_update.stop_time_update.arrival.time"] is None
        assert result["entity.trip_update.stop_time_update.arrival.uncertainty"] is None

    def test_extracts_arrival_event_fields(self):
        stu = gtfs_realtime_pb2.TripUpdate.StopTimeUpdate()
        stu.arrival.delay = 30
        stu.arrival.time = 1_700_000_000

        result = parse_stop_time_event(stu, "arrival")

        assert result["entity.trip_update.stop_time_update.arrival.delay"] == 30
        assert result["entity.trip_update.stop_time_update.arrival.time"] == 1_700_000_000

    def test_extracts_departure_event_fields(self):
        stu = gtfs_realtime_pb2.TripUpdate.StopTimeUpdate()
        stu.departure.delay = -60
        stu.departure.time = 1_700_001_000

        result = parse_stop_time_event(stu, "departure")

        assert result["entity.trip_update.stop_time_update.departure.delay"] == -60
        assert result["entity.trip_update.stop_time_update.departure.time"] == 1_700_001_000


class TestParseStopTimeProperties:
    def test_returns_none_values_when_absent(self):
        stu = gtfs_realtime_pb2.TripUpdate.StopTimeUpdate()

        result = parse_stop_time_properties(stu)

        prefix = "entity.trip_update.stop_time_update.stop_time_properties"
        assert result[f"{prefix}.assigned_stop_id"] is None
        assert result[f"{prefix}.stop_headsign"] is None
        assert result[f"{prefix}.drop_off_type"] is None
        assert result[f"{prefix}.pickup_type"] is None


class TestParseTripProperties:
    def test_returns_none_values_when_absent(self):
        tu = gtfs_realtime_pb2.TripUpdate()

        result = parse_trip_properties(tu)

        prefix = "entity.trip_update.trip_properties"
        assert result[f"{prefix}.trip_id"] is None
        assert result[f"{prefix}.start_date"] is None
        assert result[f"{prefix}.start_time"] is None
        assert result[f"{prefix}.shape_id"] is None


class TestParseTuFile:
    def test_parses_feed_to_rows(self, tmp_path):
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.header.gtfs_realtime_version = "2.0"
        feed.header.timestamp = 1_700_000_000
        entity = feed.entity.add()
        entity.id = "1"
        tu = entity.trip_update
        tu.trip.trip_id = "T1"
        tu.trip.route_id = "R1"
        tu.timestamp = 1_700_000_000
        stu = tu.stop_time_update.add()
        stu.stop_id = "S1"
        stu.arrival.time = 1_700_001_000

        pb_path = tmp_path / "feed.pb"
        pb_path.write_bytes(feed.SerializeToString())

        rows = parse_tu_file(pb_path)

        assert len(rows) == 1
        assert rows[0]["entity.trip_update.trip.trip_id"] == "T1"
        assert rows[0]["entity.trip_update.stop_time_update.stop_id"] == "S1"
        assert rows[0]["file"] == "feed.pb"

    def test_emits_one_row_per_stop_time_update(self, tmp_path):
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.header.gtfs_realtime_version = "2.0"
        feed.header.timestamp = 1_700_000_000
        entity = feed.entity.add()
        entity.id = "1"
        tu = entity.trip_update
        tu.trip.trip_id = "T1"
        for i, stop_id in enumerate(["S1", "S2", "S3"], 1):
            stu = tu.stop_time_update.add()
            stu.stop_id = stop_id
            stu.stop_sequence = i
            stu.arrival.time = 1_700_000_000 + i * 60

        pb_path = tmp_path / "feed.pb"
        pb_path.write_bytes(feed.SerializeToString())

        rows = parse_tu_file(pb_path)

        assert len(rows) == 3

    def test_skips_entities_without_trip_update(self, tmp_path):
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.header.gtfs_realtime_version = "2.0"
        feed.header.timestamp = 1_700_000_000
        entity = feed.entity.add()
        entity.id = "1"
        entity.vehicle.trip.trip_id = "T1"  # vehicle entity, not trip_update

        pb_path = tmp_path / "feed.pb"
        pb_path.write_bytes(feed.SerializeToString())

        rows = parse_tu_file(pb_path)

        assert rows == []

    def test_header_fields_duplicated_into_each_row(self, tmp_path):
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.header.gtfs_realtime_version = "2.0"
        feed.header.timestamp = 1_700_000_000
        entity = feed.entity.add()
        entity.id = "1"
        tu = entity.trip_update
        tu.trip.trip_id = "T1"
        for i in range(2):
            stu = tu.stop_time_update.add()
            stu.stop_id = f"S{i}"

        pb_path = tmp_path / "feed.pb"
        pb_path.write_bytes(feed.SerializeToString())

        rows = parse_tu_file(pb_path)

        assert all(r["header.gtfs_realtime_version"] == "2.0" for r in rows)
        assert all(r["header.timestamp"] == 1_700_000_000 for r in rows)


# ── vehicle_positions parsers ────────────────────────────────────────────────


class TestParseVpTripDescriptor:
    def test_returns_none_values_when_trip_absent(self):
        vp = gtfs_realtime_pb2.VehiclePosition()

        result = parse_vp_trip(vp)

        assert result["entity.vehicle.trip.trip_id"] is None
        assert result["entity.vehicle.trip.route_id"] is None

    def test_extracts_trip_fields(self):
        vp = gtfs_realtime_pb2.VehiclePosition()
        vp.trip.trip_id = "T1"
        vp.trip.route_id = "R1"

        result = parse_vp_trip(vp)

        assert result["entity.vehicle.trip.trip_id"] == "T1"
        assert result["entity.vehicle.trip.route_id"] == "R1"

    def test_modified_trip_fields_are_none_when_absent(self):
        vp = gtfs_realtime_pb2.VehiclePosition()
        vp.trip.trip_id = "T1"  # trip is set, but no modified_trip

        result = parse_vp_trip(vp)

        assert result["entity.vehicle.trip.modified_trip.service_id"] is None
        assert result["entity.vehicle.trip.modified_trip.trip_modification_id"] is None


class TestParseVehicleDescriptor:
    def test_returns_none_values_when_vehicle_absent(self):
        vp = gtfs_realtime_pb2.VehiclePosition()

        result = parse_vehicle_descriptor(vp)

        assert result["entity.vehicle.vehicle.id"] is None
        assert result["entity.vehicle.vehicle.label"] is None
        assert result["entity.vehicle.vehicle.license_plate"] is None

    def test_extracts_vehicle_fields(self):
        vp = gtfs_realtime_pb2.VehiclePosition()
        vp.vehicle.id = "V1"
        vp.vehicle.label = "Bus 99"
        vp.vehicle.license_plate = "XYZ789"

        result = parse_vehicle_descriptor(vp)

        assert result["entity.vehicle.vehicle.id"] == "V1"
        assert result["entity.vehicle.vehicle.label"] == "Bus 99"
        assert result["entity.vehicle.vehicle.license_plate"] == "XYZ789"


class TestParsePosition:
    def test_returns_none_values_when_position_absent(self):
        vp = gtfs_realtime_pb2.VehiclePosition()

        result = parse_position(vp)

        assert result["entity.vehicle.position.latitude"] is None
        assert result["entity.vehicle.position.longitude"] is None

    def test_extracts_lat_lng(self):
        vp = gtfs_realtime_pb2.VehiclePosition()
        vp.position.latitude = 45.5
        vp.position.longitude = -73.6

        result = parse_position(vp)

        assert result["entity.vehicle.position.latitude"] == pytest.approx(45.5, abs=1e-4)
        assert result["entity.vehicle.position.longitude"] == pytest.approx(-73.6, abs=1e-4)


class TestParseVpFile:
    def test_parses_feed_to_rows(self, tmp_path):
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.header.gtfs_realtime_version = "2.0"
        feed.header.timestamp = 1_700_000_000
        entity = feed.entity.add()
        entity.id = "1"
        vp = entity.vehicle
        vp.trip.trip_id = "T1"
        vp.vehicle.id = "V1"
        vp.stop_id = "S1"
        vp.timestamp = 1_700_000_000

        pb_path = tmp_path / "feed.pb"
        pb_path.write_bytes(feed.SerializeToString())

        rows = list(parse_vp_file(pb_path))

        assert len(rows) == 1
        assert rows[0]["entity.vehicle.trip.trip_id"] == "T1"
        assert rows[0]["entity.vehicle.vehicle.id"] == "V1"
        assert rows[0]["file"] == "feed.pb"

    def test_emits_one_row_per_entity(self, tmp_path):
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.header.gtfs_realtime_version = "2.0"
        feed.header.timestamp = 1_700_000_000
        for i in range(3):
            entity = feed.entity.add()
            entity.id = str(i)
            entity.vehicle.vehicle.id = f"V{i}"

        pb_path = tmp_path / "feed.pb"
        pb_path.write_bytes(feed.SerializeToString())

        rows = list(parse_vp_file(pb_path))

        assert len(rows) == 3

    def test_skips_entities_without_vehicle(self, tmp_path):
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.header.gtfs_realtime_version = "2.0"
        feed.header.timestamp = 1_700_000_000
        entity = feed.entity.add()
        entity.id = "1"
        entity.trip_update.trip.trip_id = "T1"  # trip_update entity, not vehicle

        pb_path = tmp_path / "feed.pb"
        pb_path.write_bytes(feed.SerializeToString())

        rows = list(parse_vp_file(pb_path))

        assert rows == []
