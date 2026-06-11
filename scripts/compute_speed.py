import pandas as pd

from compute_distance import RouteDistanceCalculator


def compute_distance(row):
    calc = calculators[row["gtfs_shape_id"]]

    return calc.route_distance(
        row["gtfsrt_vp_position_latitude_previous"],
        row["gtfsrt_vp_position_longitude_previous"],
        row["gtfsrt_vp_position_latitude_current"],
        row["gtfsrt_vp_position_longitude_current"],
    )

def compute_distance_stop(row):
    calc = calculators[row["gtfs_shape_id"]]

    return calc.route_distance(
        row["gtfsrt_vp_position_latitude_current"],
        row["gtfsrt_vp_position_longitude_current"],
        row["gtfs_next_stop_latitude"],
        row["gtfs_next_stop_longitude"],
    )

df = pd.read_csv("travel_times.csv")

shape_ids = df["gtfs_shape_id"].unique()

calculators = {
    shape_id: RouteDistanceCalculator(shape_id)
    for shape_id in shape_ids
}

# Drop the rows where there are no next stop defined for now - TODO: check when this occurs
df = df.dropna(subset=["gtfs_next_stop_latitude"])
df["gtfsrt_vp_position_timestamp_current"] = pd.to_datetime(df["gtfsrt_vp_position_timestamp_current"])
df["gtfsrt_vp_position_timestamp_previous"] = pd.to_datetime(df["gtfsrt_vp_position_timestamp_previous"])
df["gtfs_next_stop_arrival_time"] = pd.to_datetime(df["gtfs_next_stop_arrival_time"])


df = df.sort_values(["gtfs_trip_id", "gtfsrt_vp_position_timestamp_current"], ignore_index=True)

df["distance_travelled_along_route"] = df.apply(
    compute_distance,
    axis=1
)

df["speed"] = df["distance_travelled_along_route"]/df["travel_time_seconds"]

# Ensure there are no edge case of not moving (and we are very close to/at the stop)
df["speed"] = df["speed"].clip(lower=0.01)

df["distance_to_next_stop"] = df.apply(
    compute_distance_stop,
    axis=1
)

g = df.groupby("gtfs_trip_id", sort=False)

change_mask = g["gtfs_next_stop_sequence_number"].transform(
    lambda s: s != s.shift(-1)
)

next_speed = g["speed"].shift(-1)

df["estimated_stop_pass_time"] = pd.Series(
    pd.NaT,
    index=df.index,
    dtype="datetime64[ns, UTC]"
)

mask = change_mask & df["distance_to_next_stop"].notna() & next_speed.notna() & (next_speed > 0)

df.loc[mask, "estimated_stop_pass_time"] = (
    df.loc[mask, "gtfsrt_vp_position_timestamp_current"]
    + pd.to_timedelta(
        df.loc[mask, "distance_to_next_stop"] / next_speed[mask],
        unit="s"
    )
)

df["estimated_stop_pass_time"] = (
    df.groupby(["gtfs_trip_id", "gtfs_next_stop_sequence_number"])["estimated_stop_pass_time"]
      .bfill()
)

df.to_csv("travel_times_updated.csv", index=False)