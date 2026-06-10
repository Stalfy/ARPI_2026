"""Schemas and deduplication keys for parquetized GTFS-RT datasets."""

import polars as pl

FLUSH_EVERY = 200_000  # rows in pending before flushing to DuckDB

PL_TO_DUCKDB: dict = {
    pl.String: "VARCHAR",
    pl.Int32: "INTEGER",
    pl.Int64: "BIGINT",
    pl.UInt32: "UINTEGER",
    pl.Float32: "FLOAT",
    pl.Float64: "DOUBLE",
    pl.Boolean: "BOOLEAN",
}

TU_SCHEMA = {
    # source (outside spec)
    "file": pl.String,
    # FeedHeader
    "header.gtfs_realtime_version": pl.String,
    "header.incrementality": pl.Int32,
    "header.timestamp": pl.Int64,
    "header.feed_version": pl.String,
    # FeedEntity
    "entity.id": pl.String,
    "entity.is_deleted": pl.Boolean,
    # TripDescriptor
    "entity.trip_update.trip.trip_id": pl.String,
    "entity.trip_update.trip.route_id": pl.String,
    "entity.trip_update.trip.direction_id": pl.UInt32,
    "entity.trip_update.trip.start_time": pl.String,
    "entity.trip_update.trip.start_date": pl.String,
    "entity.trip_update.trip.schedule_relationship": pl.Int32,
    # ModifiedTripSelector
    "entity.trip_update.trip.modified_trip.service_id": pl.String,
    "entity.trip_update.trip.modified_trip.trip_modification_id": pl.String,
    # VehicleDescriptor (TripUpdate.vehicle)
    "entity.trip_update.vehicle.id": pl.String,
    "entity.trip_update.vehicle.label": pl.String,
    "entity.trip_update.vehicle.license_plate": pl.String,
    "entity.trip_update.vehicle.wheelchair_accessible": pl.Int32,
    # StopTimeUpdate
    "entity.trip_update.stop_time_update.stop_sequence": pl.UInt32,
    "entity.trip_update.stop_time_update.stop_id": pl.String,
    # StopTimeEvent – arrival
    "entity.trip_update.stop_time_update.arrival.delay": pl.Int32,
    "entity.trip_update.stop_time_update.arrival.time": pl.Int64,
    "entity.trip_update.stop_time_update.arrival.scheduled_time": pl.Int64,
    "entity.trip_update.stop_time_update.arrival.uncertainty": pl.Int32,
    # StopTimeEvent – departure
    "entity.trip_update.stop_time_update.departure.delay": pl.Int32,
    "entity.trip_update.stop_time_update.departure.time": pl.Int64,
    "entity.trip_update.stop_time_update.departure.scheduled_time": pl.Int64,
    "entity.trip_update.stop_time_update.departure.uncertainty": pl.Int32,
    # StopTimeUpdate – other fields
    "entity.trip_update.stop_time_update.departure_occupancy_status": pl.Int32,
    "entity.trip_update.stop_time_update.schedule_relationship": pl.Int32,
    # StopTimeProperties
    "entity.trip_update.stop_time_update.stop_time_properties.assigned_stop_id": pl.String,
    "entity.trip_update.stop_time_update.stop_time_properties.stop_headsign": pl.String,
    "entity.trip_update.stop_time_update.stop_time_properties.drop_off_type": pl.Int32,
    "entity.trip_update.stop_time_update.stop_time_properties.pickup_type": pl.Int32,
    # TripUpdate – scalar fields
    "entity.trip_update.timestamp": pl.Int64,
    "entity.trip_update.delay": pl.Int32,
    # TripProperties
    "entity.trip_update.trip_properties.trip_id": pl.String,
    "entity.trip_update.trip_properties.start_date": pl.String,
    "entity.trip_update.trip_properties.start_time": pl.String,
    "entity.trip_update.trip_properties.trip_headsign": pl.String,
    "entity.trip_update.trip_properties.trip_short_name": pl.String,
    "entity.trip_update.trip_properties.shape_id": pl.String,
}

TU_DEDUP_KEYS = [
    "entity.id",
    "entity.trip_update.timestamp",
]

VP_SCHEMA = {
    # source (outside spec)
    "file": pl.String,
    # FeedHeader
    "header.gtfs_realtime_version": pl.String,
    "header.incrementality": pl.Int32,
    "header.timestamp": pl.Int64,
    "header.feed_version": pl.String,
    # FeedEntity
    "entity.id": pl.String,
    "entity.is_deleted": pl.Boolean,
    # TripDescriptor
    "entity.vehicle.trip.trip_id": pl.String,
    "entity.vehicle.trip.route_id": pl.String,
    "entity.vehicle.trip.direction_id": pl.UInt32,
    "entity.vehicle.trip.start_time": pl.String,
    "entity.vehicle.trip.start_date": pl.String,
    "entity.vehicle.trip.schedule_relationship": pl.Int32,
    # ModifiedTripSelector
    "entity.vehicle.trip.modified_trip.service_id": pl.String,
    "entity.vehicle.trip.modified_trip.trip_modification_id": pl.String,
    # VehicleDescriptor
    "entity.vehicle.vehicle.id": pl.String,
    "entity.vehicle.vehicle.label": pl.String,
    "entity.vehicle.vehicle.license_plate": pl.String,
    "entity.vehicle.vehicle.wheelchair_accessible": pl.Int32,
    # Position
    "entity.vehicle.position.latitude": pl.Float32,
    "entity.vehicle.position.longitude": pl.Float32,
    "entity.vehicle.position.bearing": pl.Float32,
    "entity.vehicle.position.odometer": pl.Float64,
    "entity.vehicle.position.speed": pl.Float32,
    # VehiclePosition – scalar fields
    "entity.vehicle.current_stop_sequence": pl.UInt32,
    "entity.vehicle.stop_id": pl.String,
    "entity.vehicle.current_status": pl.Int32,
    "entity.vehicle.timestamp": pl.Int64,
    "entity.vehicle.congestion_level": pl.Int32,
    "entity.vehicle.occupancy_status": pl.Int32,
    "entity.vehicle.occupancy_percentage": pl.UInt32,
}

VP_DEDUP_KEYS = [
    "entity.vehicle.trip.trip_id",
    "entity.vehicle.stop_id",  # populated for FROM_VEHICLE_POSITION agencies
    "entity.vehicle.current_stop_sequence",  # fallback for FROM_STOPSEQUENCE agencies
]
