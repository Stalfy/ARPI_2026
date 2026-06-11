from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="GTFS RT Trip Explorer", layout="wide")

DEFAULT_CSV = Path("vehicle_position_timesteps_updated.csv")
DEFAULT_STOPS = Path("ref/GTFS/RTL/stops.txt")

TIME_COLS = [
    "gtfs_service_date",
    "gtfsrt_vp_position_timestamp_previous",
    "gtfsrt_vp_position_timestamp_current",
    "gtfsrt_tu_stop_predicted_arrival",
    "gtfsrt_tu_stop_predicted_departure",
    "estimated_stop_pass_time",
]

TEXT_TIME_COLS = [
    "gtfs_stop_scheduled_arrival",
    "gtfs_stop_scheduled_departure",
]


def parse_time_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in TIME_COLS:
        if col in df.columns:
            df[col] = parse_time_column(df[col])

    for col in TEXT_TIME_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


@st.cache_data(show_spinner=False)
def load_data_from_path(csv_path: str) -> pd.DataFrame:
    return normalize_dataframe(pd.read_csv(csv_path))


@st.cache_data(show_spinner=False)
def load_data_from_bytes(csv_bytes: bytes) -> pd.DataFrame:
    return normalize_dataframe(pd.read_csv(pd.io.common.BytesIO(csv_bytes)))


@st.cache_data(show_spinner=False)
def load_stops_from_path(stops_path: str) -> pd.DataFrame:
    stops = pd.read_csv(stops_path)[["stop_id", "stop_lat", "stop_lon"]]
    stops = stops.rename(
        columns={
            "stop_id": "gtfs_stop_id",
            "stop_lat": "gtfs_stop_latitude",
            "stop_lon": "gtfs_stop_longitude",
        }
    )
    return stops


def fmt_ts(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return "<missing>"
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%d %H:%M:%S %Z")


def fmt_any(value) -> str:
    if pd.isna(value):
        return "<missing>"
    if isinstance(value, pd.Timestamp):
        return fmt_ts(value)
    return str(value)


def format_distance(value) -> str:
    if pd.isna(value):
        return "<missing>"
    try:
        return f"{float(value):,.1f}"
    except Exception:
        return str(value)


def get_stop_key_cols(df: pd.DataFrame) -> list[str]:
    preferred = ["gtfs_stop_id", "gtfs_stop_sequence", "gtfs_stop_name"]
    return [c for c in preferred if c in df.columns]


def make_stop_key_series(df: pd.DataFrame, key_cols: list[str]) -> pd.Series:
    if not key_cols:
        return pd.Series(["__all__"] * len(df), index=df.index, dtype="object")

    def row_key(row) -> tuple:
        return tuple("" if pd.isna(row[c]) else str(row[c]) for c in key_cols)

    return df.apply(row_key, axis=1)


def build_stop_hover(r: pd.Series, is_past: bool) -> str:
    stop_name = fmt_any(r.get("gtfs_stop_name", ""))
    state_label = "past stop" if is_past else "active stop"

    lines = [
        f"<b>{stop_name}</b>",
        f"status: {state_label}",
        f"sequence: {fmt_any(r.get('gtfs_stop_sequence', ''))}",
    ]

    if not is_past:
        lines.append(f"distance_to_stop_m: {format_distance(r.get('distance_to_stop_m'))}")

    pred_arr_label = "final predicted arrival" if is_past else "predicted arrival"
    pred_dep_label = "final predicted departure" if is_past else "predicted departure"

    lines.extend(
        [
            f"{pred_arr_label}: {fmt_ts(r.get('gtfsrt_tu_stop_predicted_arrival'))}",
            f"{pred_dep_label}: {fmt_ts(r.get('gtfsrt_tu_stop_predicted_departure'))}",
            f"scheduled arrival: {fmt_any(r.get('gtfs_stop_scheduled_arrival'))}",
            f"scheduled departure: {fmt_any(r.get('gtfs_stop_scheduled_departure'))}",
            f"estimated pass time: {fmt_ts(r.get('estimated_stop_pass_time'))}",
        ]
    )

    return "<br>".join(lines)


def build_map(current_row: pd.Series, active_stops: pd.DataFrame, past_stops: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if not past_stops.empty:
        hover_text = [build_stop_hover(r, is_past=True) for _, r in past_stops.iterrows()]
        fig.add_trace(
            go.Scattermapbox(
                lat=past_stops["gtfs_stop_latitude"],
                lon=past_stops["gtfs_stop_longitude"],
                mode="markers",
                marker=dict(size=10),
                opacity=0.35,
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                name="Past stops",
            )
        )

    if not active_stops.empty:
        hover_text = [build_stop_hover(r, is_past=False) for _, r in active_stops.iterrows()]
        fig.add_trace(
            go.Scattermapbox(
                lat=active_stops["gtfs_stop_latitude"],
                lon=active_stops["gtfs_stop_longitude"],
                mode="markers",
                marker=dict(size=11),
                opacity=0.95,
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                name="Active stops",
            )
        )

    fig.add_trace(
        go.Scattermapbox(
            lat=[current_row["gtfsrt_vp_position_latitude_current"]],
            lon=[current_row["gtfsrt_vp_position_longitude_current"]],
            mode="markers",
            marker=dict(size=16),
            text=[
                "<b>Current vehicle position</b>"
                f"<br>timestamp: {fmt_ts(current_row.get('gtfsrt_vp_position_timestamp_current'))}"
                f"<br>gtfs_trip_id: {fmt_any(current_row.get('gtfs_trip_id', ''))}"
            ],
            hovertemplate="%{text}<extra></extra>",
            name="Current position",
        )
    )

    center_lat = float(current_row["gtfsrt_vp_position_latitude_current"])
    center_lon = float(current_row["gtfsrt_vp_position_longitude_current"])

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=13,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h"),
        height=650,
    )
    return fig


def build_stop_display_frame(active_stops: pd.DataFrame, past_stops: pd.DataFrame) -> pd.DataFrame:
    frames = []

    if not active_stops.empty:
        active = active_stops.copy()
        active["stop_state"] = "active"
        active["distance_to_stop_m_display"] = active["distance_to_stop_m"].apply(format_distance)
        active["predicted_arrival_display"] = active["gtfsrt_tu_stop_predicted_arrival"].apply(fmt_ts)
        active["predicted_departure_display"] = active["gtfsrt_tu_stop_predicted_departure"].apply(fmt_ts)
        active["estimated_stop_pass_time_display"] = active["estimated_stop_pass_time"].apply(fmt_ts)
        frames.append(active)

    if not past_stops.empty:
        past = past_stops.copy()
        past["stop_state"] = "past"
        past["distance_to_stop_m_display"] = ""
        past["predicted_arrival_display"] = past["gtfsrt_tu_stop_predicted_arrival"].apply(lambda v: f"final predicted: {fmt_ts(v)}")
        past["predicted_departure_display"] = past["gtfsrt_tu_stop_predicted_departure"].apply(lambda v: f"final predicted: {fmt_ts(v)}")
        past["estimated_stop_pass_time_display"] = past["estimated_stop_pass_time"].apply(fmt_ts)
        frames.append(past)

    if not frames:
        return pd.DataFrame()

    display_df = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ["stop_state", "gtfs_stop_sequence", "distance_to_stop_m"] if c in display_df.columns]
    if sort_cols:
        display_df = display_df.sort_values(sort_cols, na_position="last")
    return display_df


st.title("GTFS RT trip explorer")
st.caption("Pick a trip, then step through vehicle-position timestamps to inspect the live position and downstream stops.")

uploaded = st.sidebar.file_uploader("Upload a CSV", type=["csv"])

if uploaded is not None:
    df = load_data_from_bytes(uploaded.getvalue())
    source_name = uploaded.name
else:
    df = load_data_from_path(str(DEFAULT_CSV))
    source_name = DEFAULT_CSV.name

stops = load_stops_from_path(str(DEFAULT_STOPS))
df = df.merge(stops, on="gtfs_stop_id", how="left")

# Normalize stop coordinate column names after merge
if "gtfs_stop_latitude" not in df.columns or "gtfs_stop_longitude" not in df.columns:
    if "gtfs_stop_latitude_x" in df.columns and "gtfs_stop_longitude_x" in df.columns:
        df = df.rename(
            columns={
                "gtfs_stop_latitude_x": "gtfs_stop_latitude",
                "gtfs_stop_longitude_x": "gtfs_stop_longitude",
            }
        )
        drop_cols = [c for c in ["gtfs_stop_latitude_y", "gtfs_stop_longitude_y"] if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

st.sidebar.markdown(f"**Data source:** `{source_name}`")
st.sidebar.markdown(f"**Rows:** {len(df):,}")

required = {
    "gtfs_trip_id",
    "gtfsrt_vp_position_timestamp_current",
    "gtfsrt_vp_position_latitude_current",
    "gtfsrt_vp_position_longitude_current",
    "gtfs_stop_id",
    "gtfs_stop_name",
    "gtfs_stop_sequence",
    "distance_to_stop_m",
    "gtfs_stop_scheduled_arrival",
    "gtfsrt_tu_stop_predicted_arrival",
}

missing = sorted(required - set(df.columns))
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

trip_ids = sorted(df["gtfs_trip_id"].dropna().astype(str).unique().tolist())
if not trip_ids:
    st.warning("No trip IDs found in the file.")
    st.stop()

selected_trip = st.sidebar.selectbox("gtfs_trip_id", trip_ids)
trip_df = df[df["gtfs_trip_id"].astype(str) == selected_trip].copy()

sort_cols = ["gtfsrt_vp_position_timestamp_current"]
if "gtfs_stop_sequence" in trip_df.columns:
    sort_cols.append("gtfs_stop_sequence")
trip_df = trip_df.sort_values(sort_cols, na_position="last")

# One row per current position timestamp
ts_df = (
    trip_df[["gtfsrt_vp_position_timestamp_current"]]
    .dropna()
    .drop_duplicates()
    .sort_values("gtfsrt_vp_position_timestamp_current")
    .reset_index(drop=True)
)

if ts_df.empty:
    st.warning("No timestamps available for the selected trip.")
    st.stop()

options = [f"{i:03d} | {fmt_ts(ts)}" for i, ts in enumerate(ts_df["gtfsrt_vp_position_timestamp_current"].tolist())]
selected_option = st.sidebar.selectbox(
    "gtfsrt_vp_position_timestamp_current",
    options,
    index=0,
)
selected_idx = int(selected_option.split("|", 1)[0].strip())
selected_ts = ts_df.loc[selected_idx, "gtfsrt_vp_position_timestamp_current"]

slice_df = trip_df[trip_df["gtfsrt_vp_position_timestamp_current"] == selected_ts].copy()

valid_pos = slice_df.dropna(subset=["gtfsrt_vp_position_latitude_current", "gtfsrt_vp_position_longitude_current"])
if valid_pos.empty:
    st.warning("No valid current position for the selected timestamp.")
    st.stop()

current_row = valid_pos.iloc[0]

# Build active stops from the selected timestamp slice
active_stops = slice_df.dropna(subset=["gtfs_stop_latitude", "gtfs_stop_longitude"]).copy()
active_stop_key_cols = get_stop_key_cols(active_stops)
if not active_stops.empty:
    active_stops = active_stops.sort_values(
        ["gtfs_stop_sequence", "distance_to_stop_m"],
        na_position="last",
    )
    active_stops = active_stops.assign(_stop_key=make_stop_key_series(active_stops, active_stop_key_cols))
    active_stops = active_stops.drop_duplicates(subset=["_stop_key"], keep="first").copy()

# Build the full stop history up to the selected timestamp so past stops remain visible
history_df = trip_df[trip_df["gtfsrt_vp_position_timestamp_current"] <= selected_ts].copy()
past_stops = history_df.dropna(subset=["gtfs_stop_latitude", "gtfs_stop_longitude"]).copy()

past_stop_key_cols = get_stop_key_cols(past_stops)
if not past_stops.empty:
    past_stops = past_stops.sort_values(
        ["gtfsrt_vp_position_timestamp_current", "gtfs_stop_sequence", "distance_to_stop_m"],
        na_position="last",
    )
    past_stops = past_stops.assign(_stop_key=make_stop_key_series(past_stops, past_stop_key_cols))
    past_stops = past_stops.drop_duplicates(subset=["_stop_key"], keep="last").copy()

# Remove active stops from the past-stop layer so they stay at normal opacity/color
if not active_stops.empty and not past_stops.empty:
    active_keys = set(active_stops["_stop_key"].tolist())
    past_stops = past_stops[~past_stops["_stop_key"].isin(active_keys)].copy()

# Clean helper columns before display/plotting
for frame_name in ["active_stops", "past_stops"]:
    frame = locals()[frame_name]
    if not frame.empty and "_stop_key" in frame.columns:
        frame.drop(columns=["_stop_key"], inplace=True)

display_stops = build_stop_display_frame(active_stops, past_stops)

st.subheader("Selection")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Trip", selected_trip)
col2.metric("Position index", f"{selected_idx}")
col3.metric("Active stops", f"{len(active_stops):,}")
col4.metric("Past stops", f"{len(past_stops):,}")
col5.metric("Vehicle ID", str(current_row.get("gtfsrt_vp_vehicle_id", "")))

left, right = st.columns([1.25, 1])
with left:
    st.plotly_chart(build_map(current_row, active_stops, past_stops), use_container_width=True)

with right:
    st.subheader("Current location")
    current_summary = pd.DataFrame(
        {
            "field": [
                "gtfsrt_vp_position_timestamp_current",
                "gtfsrt_vp_position_latitude_current",
                "gtfsrt_vp_position_longitude_current",
                "current_distance_along_route_m",
                "current_offset_from_route_m",
            ],
            "value": [
                fmt_ts(current_row.get("gtfsrt_vp_position_timestamp_current")),
                current_row.get("gtfsrt_vp_position_latitude_current"),
                current_row.get("gtfsrt_vp_position_longitude_current"),
                current_row.get("current_distance_along_route_m"),
                current_row.get("current_offset_from_route_m"),
            ],
        }
    )
    st.dataframe(current_summary, use_container_width=True, hide_index=True)

    st.subheader("Stops")
    if display_stops.empty:
        st.info("No stops available for this selection.")
    else:
        display_cols = [
            "stop_state",
            "gtfs_stop_sequence",
            "gtfs_stop_name",
            "distance_to_stop_m_display",
            "predicted_arrival_display",
            "gtfs_stop_scheduled_arrival",
            "predicted_departure_display",
            "gtfs_stop_scheduled_departure",
            "estimated_stop_pass_time_display",
        ]
        existing_display_cols = [c for c in display_cols if c in display_stops.columns]
        st.dataframe(
            display_stops[existing_display_cols].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

st.caption("Hover a stop marker to see distance_to_stop_m, predicted arrival/departure, scheduled arrival/departure, and estimated pass time.")
