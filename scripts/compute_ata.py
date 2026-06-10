import pandas as pd
from compute_distance import RouteDistanceCalculator

df = pd.read_csv("vehicle_position_timesteps.csv")

stops = pd.read_csv("ref/GTFS/RTL/stops.txt")[["stop_id", "stop_lat", "stop_lon"]]
stops = stops.rename(columns={
    "stop_id": "gtfs_stop_id",
    "stop_lat": "gtfs_stop_latitude",
    "stop_lon": "gtfs_stop_longitude",
})


df = df.merge(stops, on="gtfs_stop_id", how="left")

for col in [
    "gtfsrt_vp_position_timestamp_previous",
    "gtfsrt_vp_position_timestamp_current",
    "gtfsrt_tu_stop_predicted_arrival",
    "gtfsrt_tu_stop_predicted_departure",
]:
    df[col] = pd.to_datetime(df[col], unit="s", utc=True, errors="coerce")

# Define a calculator for each unique shape id
shape_ids = df["gtfs_shape_id"].dropna().unique()
calculators = {
    shape_id: RouteDistanceCalculator(shape_id)
    for shape_id in shape_ids
}

def enrich_frame(df):
    pieces = []

    for shape_id, idx in df.groupby("gtfs_shape_id").groups.items():
        calc = calculators[shape_id]
        sub = df.loc[idx]

        pieces.append(
            calc.add_distance_to_stops(
                sub,
                current_lat_col="gtfsrt_vp_position_latitude_current",
                current_lon_col="gtfsrt_vp_position_longitude_current",
                stop_sequence_col="gtfs_stop_sequence",
            )
        )

    return pd.concat(pieces).sort_index()

def compute_distance(row):
    calc = calculators[row["gtfs_shape_id"]]
    return calc.distance_between_points_m(
        row["gtfsrt_vp_position_latitude_previous"],
        row["gtfsrt_vp_position_longitude_previous"],
        row["gtfsrt_vp_position_latitude_current"],
        row["gtfsrt_vp_position_longitude_current"],
    )

# def compute_distance_to_stop(row):
#     calc = calculators[row["gtfs_shape_id"]]
#     return calc.distance_between_points_m(
#         row["gtfsrt_vp_position_latitude_current"],
#         row["gtfsrt_vp_position_longitude_current"],
#         row["gtfs_stop_latitude"],
#         row["gtfs_stop_longitude"],
#     )

df = df.dropna(subset=["gtfs_stop_latitude", "gtfs_stop_longitude"]).copy()

df = df.sort_values(
    ["gtfs_trip_id", "gtfsrt_vp_position_timestamp_current"],
    ignore_index=True
)

df = enrich_frame(df)

df["distance_travelled_along_route"] = df.apply(compute_distance, axis=1)
df["speed"] = df["distance_travelled_along_route"] / df["travel_time_seconds"]
df["speed"] = df["speed"].clip(lower=0.01)

# Infer the next stop:
closest_idx = (
    df.groupby(["gtfs_trip_id", "gtfsrt_vp_position_timestamp_current"])["distance_to_stop_m"]
      .idxmin()
)

df_sub = df.loc[closest_idx].copy()

# Re-sort after collapsing to one stop per vehicle observation
df_sub = df_sub.sort_values(
    ["gtfs_trip_id", "gtfsrt_vp_position_timestamp_current"],
    ignore_index=True
)

# Estimate stop pass time
g = df_sub.groupby("gtfs_trip_id", sort=False)

next_speed = g["speed"].shift(-1)
change_mask = g["gtfs_stop_id"].transform(lambda s: s != s.shift(-1))

df_sub["estimated_stop_pass_time"] = pd.Series(
    pd.NaT,
    index=df_sub.index,
    dtype="datetime64[ns, UTC]"
)

mask = (
    change_mask
    & df_sub["distance_to_stop_m"].notna()
    & next_speed.notna()
    & (next_speed > 0)
)

df_sub.loc[mask, "estimated_stop_pass_time"] = (
    df_sub.loc[mask, "gtfsrt_vp_position_timestamp_current"]
    + pd.to_timedelta(
        df_sub.loc[mask, "distance_to_stop_m"] / next_speed[mask],
        unit="s"
    )
)

obs_cols = [
    "gtfs_trip_id",
    "gtfsrt_vp_position_timestamp_previous",
    "gtfsrt_vp_position_latitude_previous",
    "gtfsrt_vp_position_longitude_previous",
    "gtfsrt_vp_position_timestamp_current",
    "gtfsrt_vp_position_latitude_current",
    "gtfsrt_vp_position_longitude_current",
]

df = df.merge(
    df_sub[obs_cols + ["gtfs_stop_id", "gtfs_stop_sequence", "estimated_stop_pass_time"]],
    on=obs_cols + ["gtfs_stop_id", "gtfs_stop_sequence"],
    how="left"
    )

# Fill backward within each trip and stop
df["estimated_stop_pass_time"] = (
    df.groupby(["gtfs_trip_id", "gtfs_stop_id"])["estimated_stop_pass_time"]
      .bfill()
)

df.to_csv("vehicle_position_timesteps_updated.csv", index=False)