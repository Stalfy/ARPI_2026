import numpy as np
import pandas as pd
from compute_distance import RouteDistanceCalculator


def enrich_frame(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []

    for (shape_id, trip_id), idx in frame.groupby(["gtfs_shape_id", "gtfs_trip_id"], sort=False).groups.items():
        calc = calculators[(shape_id, trip_id)]
        sub = frame.loc[idx].copy()

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
    calc = calculators[(row["gtfs_shape_id"], row["gtfs_trip_id"])]
    return calc.distance_between_points_m(
        row["gtfsrt_vp_position_latitude_previous"],
        row["gtfsrt_vp_position_longitude_previous"],
        row["gtfsrt_vp_position_latitude_current"],
        row["gtfsrt_vp_position_longitude_current"],
    )

def estimate_stop_pass_times_for_trip(obs_trip: pd.DataFrame, calc: RouteDistanceCalculator, pos_tol_m: float = 5.0) -> pd.DataFrame:
    """
    Estimate one pass time per stop for a single trip.

    Idea is that we use consecutive vehicle observations, detect when the route distance 
    # crosses a stop's route distance and interpolate time between the bracketing observations.
    """
    obs_trip = obs_trip.sort_values("gtfsrt_vp_position_timestamp_current").reset_index(drop=True)

    stop_trip = calc.stop_positions_df.copy().rename(columns={
        "stop_sequence": "gtfs_stop_sequence",
        "distance_along_route_m": "stop_distance_along_route_m",
        "offset_from_route_m": "stop_offset_from_route_m",
    })

    stop_trip = stop_trip.sort_values("stop_distance_along_route_m").reset_index(drop=True)
    stop_trip["estimated_stop_pass_time"] = pd.Series(
        pd.NaT,
        index=stop_trip.index,
        dtype="datetime64[ns, UTC]"
    )

    if stop_trip.empty or len(obs_trip) < 2:
        return stop_trip

    stop_positions = stop_trip["stop_distance_along_route_m"].astype(float).to_numpy()
    times = obs_trip["gtfsrt_vp_position_timestamp_current"].tolist()
    route_positions = obs_trip["current_distance_along_route_m"].astype(float).to_numpy()

    # For each consecutive pair of vehicle observations, assign any stops crossed in that interval.
    for i in range(1, len(obs_trip)):
        p0 = route_positions[i - 1]
        p1 = route_positions[i]

        if not np.isfinite(p0) or not np.isfinite(p1):
            continue

        # If the projected route position goes backward substantially, skip that interval.
        # Small negative jitter can happen so adjust pos_tol_m if needed.
        if p1 < p0 - pos_tol_m:
            continue

        t0 = pd.Timestamp(times[i - 1])
        t1 = pd.Timestamp(times[i])

        lower = min(p0, p1)
        upper = max(p0, p1)

        # Candidate stops in this interval
        start = np.searchsorted(stop_positions, lower - pos_tol_m, side="left")
        end = np.searchsorted(stop_positions, upper + pos_tol_m, side="right")

        for j in range(start, end):
            if pd.notna(stop_trip.at[j, "estimated_stop_pass_time"]):
                continue

            s = stop_positions[j]

            # Outside the interval
            if s < lower - pos_tol_m or s > upper + pos_tol_m:
                continue

            # If the bus is essentially stationary, assign the timestamp if the stop is at that position
            if abs(p1 - p0) <= 1e-9:
                if abs(s - p0) <= pos_tol_m:
                    est = t1
                else:
                    continue
            else:
                frac = (s - p0) / (p1 - p0)
                frac = float(np.clip(frac, 0.0, 1.0))
                est = t0 + (t1 - t0) * frac

            stop_trip.at[j, "estimated_stop_pass_time"] = est

    return stop_trip


df = pd.read_csv("vehicle_position_timesteps.csv")

stops = pd.read_csv("ref/GTFS/RTL/stops.txt")[["stop_id", "stop_lat", "stop_lon"]]
stops = stops.rename(columns={
    "stop_id": "gtfs_stop_id",
    "stop_lat": "gtfs_stop_latitude",
    "stop_lon": "gtfs_stop_longitude",
})

df = df.merge(stops, on="gtfs_stop_id", how="left")

TIME_COLS = [
    "gtfs_service_date",
    "gtfsrt_vp_position_timestamp_previous",
    "gtfsrt_vp_position_timestamp_current",
    "gtfsrt_tu_stop_predicted_arrival",
    "gtfsrt_tu_stop_predicted_departure",
    "gtfs_stop_scheduled_arrival",
    "gtfs_stop_scheduled_departure",
]

for col in TIME_COLS:
    if col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], unit="s", utc=True, errors="coerce")
        else:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")


shape_trip_pairs = (
    df[["gtfs_shape_id", "gtfs_trip_id"]]
    .dropna()
    .drop_duplicates()
)

calculators = {}
for shape_id, trip_id in shape_trip_pairs.itertuples(index=False, name=None):
    calculators[(shape_id, trip_id)] = RouteDistanceCalculator(shape_id, trip_id=trip_id)

df = df.dropna(subset=["gtfs_stop_sequence"]).copy()
df = df.sort_values(
    ["gtfs_trip_id", "gtfsrt_vp_position_timestamp_current"],
    ignore_index=True
)

df = enrich_frame(df)

df["distance_travelled_along_route"] = df.apply(compute_distance, axis=1)
df["speed"] = df["distance_travelled_along_route"] / df["travel_time_seconds"]
df["speed"] = df["speed"].clip(lower=0.01)

obs_key_cols = [
    "gtfs_shape_id",
    "gtfs_trip_id",
    "gtfsrt_vp_position_timestamp_previous",
    "gtfsrt_vp_position_latitude_previous",
    "gtfsrt_vp_position_longitude_previous",
    "gtfsrt_vp_position_timestamp_current",
    "gtfsrt_vp_position_latitude_current",
    "gtfsrt_vp_position_longitude_current",
]

obs = (
    df[obs_key_cols + ["current_distance_along_route_m"]]
    .dropna(subset=["gtfs_trip_id", "gtfsrt_vp_position_timestamp_current", "current_distance_along_route_m"])
    .drop_duplicates(subset=obs_key_cols)
    .sort_values(["gtfs_trip_id", "gtfsrt_vp_position_timestamp_current"])
    .reset_index(drop=True)
)

stop_estimates = []

for (shape_id, trip_id), calc in calculators.items():
    obs_trip = obs[
        (obs["gtfs_shape_id"] == shape_id) &
        (obs["gtfs_trip_id"] == trip_id)
    ].copy()

    if obs_trip.empty:
        continue

    stop_trip = estimate_stop_pass_times_for_trip(obs_trip, calc)
    stop_trip["gtfs_shape_id"] = shape_id
    stop_trip["gtfs_trip_id"] = trip_id

    stop_estimates.append(
        stop_trip[[
            "gtfs_shape_id",
            "gtfs_trip_id",
            "gtfs_stop_sequence",
            "stop_id",
            "stop_name",
            "estimated_stop_pass_time",
        ]]
    )

stop_estimates_df = pd.concat(stop_estimates, ignore_index=True) if stop_estimates else pd.DataFrame(
    columns=[
        "gtfs_shape_id",
        "gtfs_trip_id",
        "gtfs_stop_sequence",
        "stop_id",
        "stop_name",
        "estimated_stop_pass_time",
    ]
)

df = df.merge(
    stop_estimates_df[[
        "gtfs_trip_id",
        "gtfs_stop_sequence",
        "estimated_stop_pass_time",
    ]],
    on=["gtfs_trip_id", "gtfs_stop_sequence"],
    how="left"
)

df.to_csv("vehicle_position_timesteps_updated.csv", index=False)