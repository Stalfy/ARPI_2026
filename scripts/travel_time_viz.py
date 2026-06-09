# visualize.py
#
# Usage:
#   uv run streamlit run visualize.py -- --file travel_times.csv --gtfs GTFS.zip

import argparse
import zipfile
from pathlib import Path

import folium
import polars as pl
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium


COL = {
    "agency": "gtfs_agency",
    "service_date": "gtfs_service_date",
    "route": "gtfs_route_id",
    "service_id": "gtfs_service_id",
    "trip": "gtfs_trip_id",
    "shape": "gtfs_shape_id",
    "headsign": "gtfs_trip_headsign",
    "direction": "gtfs_direction_id",
    "block": "gtfs_block_id",
    "vehicle_id": "gtfsrt_vp_vehicle_id",
    "vehicle_label": "gtfsrt_vp_vehicle_label",
    "travel_time": "travel_time_seconds",
    "from_ts": "gtfsrt_vp_position_timestamp_previous",
    "from_lat": "gtfsrt_vp_position_latitude_previous",
    "from_lon": "gtfsrt_vp_position_longitude_previous",
    "to_ts": "gtfsrt_vp_position_timestamp_current",
    "to_lat": "gtfsrt_vp_position_latitude_current",
    "to_lon": "gtfsrt_vp_position_longitude_current",
    "stop_id": "gtfs_next_stop_id",
    "stop_seq": "gtfs_next_stop_sequence_number",
    "stop_name": "gtfs_next_stop_name",
    "stop_arrival": "gtfs_next_stop_arrival_time",
    "stop_departure": "gtfs_next_stop_departure_time",
    "stop_lat": "gtfs_next_stop_latitude",
    "stop_lon": "gtfs_next_stop_longitude",
    "entity_id": "gtfsrt_vp_entity_id",
    "rt_route": "gtfsrt_vp_trip_route_id",
    "file_prev": "gtfs_rt_file_previous",
    "file_cur": "gtfs_rt_file_current",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--gtfs", type=Path, required=True)
    return parser.parse_args()


@st.cache_data
def load_csv(path: str) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        try_parse_dates=True,
        infer_schema_length=0,
        schema_overrides={
            COL["agency"]: pl.String,
            COL["service_date"]: pl.String,
            COL["route"]: pl.String,
            COL["service_id"]: pl.String,
            COL["trip"]: pl.String,
            COL["shape"]: pl.String,
            COL["headsign"]: pl.String,
            COL["direction"]: pl.String,
            COL["block"]: pl.String,
            COL["vehicle_id"]: pl.String,
            COL["vehicle_label"]: pl.String,
            COL["stop_id"]: pl.String,
            COL["stop_seq"]: pl.Int64,
            COL["stop_name"]: pl.String,
            COL["entity_id"]: pl.String,
            COL["rt_route"]: pl.String,
            COL["file_prev"]: pl.String,
            COL["file_cur"]: pl.String,
        },
    )

    return df.with_columns([
        pl.col(COL["travel_time"]).cast(pl.Int64, strict=False),
        pl.col(COL["from_lat"]).cast(pl.Float64, strict=False),
        pl.col(COL["from_lon"]).cast(pl.Float64, strict=False),
        pl.col(COL["to_lat"]).cast(pl.Float64, strict=False),
        pl.col(COL["to_lon"]).cast(pl.Float64, strict=False),
        pl.col(COL["stop_lat"]).cast(pl.Float64, strict=False),
        pl.col(COL["stop_lon"]).cast(pl.Float64, strict=False),
    ])


@st.cache_data
def load_gtfs_shapes(gtfs_zip: str) -> pl.DataFrame:
    with zipfile.ZipFile(gtfs_zip) as zf:
        with zf.open("shapes.txt") as f:
            shapes = pl.read_csv(f, infer_schema_length=0)

    return (
        shapes.select([
            pl.col("shape_id").cast(pl.String).alias("shape"),
            pl.col("shape_pt_lat").cast(pl.Float64),
            pl.col("shape_pt_lon").cast(pl.Float64),
            pl.col("shape_pt_sequence").cast(pl.Int64),
        ])
        .sort(["shape", "shape_pt_sequence"])
    )


def apply_filters(df: pl.DataFrame) -> pl.DataFrame:
    with st.sidebar:
        st.header("Filters")

        routes = sorted(df[COL["route"]].cast(pl.String).drop_nulls().unique().to_list())
        route = st.selectbox("Route", ["All"] + routes)
        if route != "All":
            df = df.filter(pl.col(COL["route"]).cast(pl.String) == route)

        shapes = sorted(df[COL["shape"]].cast(pl.String).drop_nulls().unique().to_list())
        shape = st.selectbox("Shape", ["All"] + shapes)
        if shape != "All":
            df = df.filter(pl.col(COL["shape"]).cast(pl.String) == shape)

        trips = sorted(df[COL["trip"]].cast(pl.String).drop_nulls().unique().to_list())
        trip = st.selectbox("Trip", ["All"] + trips)
        if trip != "All":
            df = df.filter(pl.col(COL["trip"]).cast(pl.String) == trip)

        min_obs = st.number_input("Min observations per segment", min_value=1, value=2, step=1)

        max_time = int(df[COL["travel_time"]].max() or 1)
        max_segment_time = st.slider(
            "Max segment travel time (seconds)",
            min_value=1,
            max_value=max(max_time, 1),
            value=min(7200, max(max_time, 1)),
        )
        df = df.filter(pl.col(COL["travel_time"]) <= max_segment_time)

    st.session_state["min_obs"] = int(min_obs)
    return df


def summarize_worst_segments(df: pl.DataFrame, min_obs: int) -> pl.DataFrame:
    return (
        df.drop_nulls([
            COL["route"],
            COL["shape"],
            COL["from_lat"],
            COL["from_lon"],
            COL["to_lat"],
            COL["to_lon"],
            COL["travel_time"],
        ])
        .group_by([
            COL["route"],
            COL["shape"],
            COL["from_lat"],
            COL["from_lon"],
            COL["to_lat"],
            COL["to_lon"],
        ])
        .agg([
            pl.len().alias("observations"),
            pl.col(COL["travel_time"]).mean().round(2).alias("avg_sec"),
            pl.col(COL["travel_time"]).median().round(2).alias("median_sec"),
            pl.col(COL["travel_time"]).quantile(0.95).round(2).alias("p95_sec"),
            pl.col(COL["travel_time"]).max().alias("max_sec"),
            pl.col(COL["stop_id"]).drop_nulls().first().alias(COL["stop_id"]),
            pl.col(COL["stop_name"]).drop_nulls().first().alias(COL["stop_name"]),
        ])
        .filter(pl.col("observations") >= min_obs)
        .sort(["p95_sec", "avg_sec"], descending=True)
    )


def colormap(df: pl.DataFrame, column: str, caption: str) -> LinearColormap:
    if df.is_empty() or column not in df.columns:
        return LinearColormap(["green", "yellow", "red"], vmin=0, vmax=1, caption=caption)

    vmin = float(df[column].min() or 0)
    vmax = float(df[column].max() or 1)

    if vmin == vmax:
        vmax = vmin + 1

    return LinearColormap(["green", "yellow", "red"], vmin=vmin, vmax=vmax, caption=caption)


def render_gtfs_shapes(fmap: folium.Map, shapes: pl.DataFrame, selected_shapes: set[str]) -> None:
    if shapes.is_empty():
        return

    if selected_shapes:
        shapes = shapes.filter(pl.col("shape").is_in(list(selected_shapes)))

    for key, group in shapes.group_by("shape"):
        shape_id = key[0] if isinstance(key, tuple) else key
        pts = list(zip(group["shape_pt_lat"].to_list(), group["shape_pt_lon"].to_list()))

        if len(pts) < 2:
            continue

        folium.PolyLine(
            pts,
            color="#777777",
            weight=3,
            opacity=0.35,
            tooltip=f"GTFS shape {shape_id}",
        ).add_to(fmap)


def map_center(df: pl.DataFrame, from_lat: str, from_lon: str, to_lat: str, to_lon: str) -> tuple[float, float]:
    lats = pl.concat([df[from_lat], df[to_lat]])
    lons = pl.concat([df[from_lon], df[to_lon]])
    return float(lats.mean()), float(lons.mean())


def render_worst_segments_map(worst: pl.DataFrame, shapes: pl.DataFrame) -> None:
    if worst.is_empty():
        st.info("No road segments after filtering.")
        return

    sample = worst.head(500)
    selected_shapes = set(sample[COL["shape"]].cast(pl.String).drop_nulls().unique().to_list())
    center_lat, center_lon = map_center(sample, COL["from_lat"], COL["from_lon"], COL["to_lat"], COL["to_lon"])

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap")
    render_gtfs_shapes(fmap, shapes, selected_shapes)

    scale = colormap(sample, "p95_sec", "P95 travel time (seconds)")
    scale.add_to(fmap)

    for row in sample.iter_rows(named=True):
        color = scale(float(row["p95_sec"]))

        tooltip = (
            f"Route: {row[COL['route']]}<br>"
            f"Shape: {row[COL['shape']]}<br>"
            f"Obs: {row['observations']}<br>"
            f"Avg: {row['avg_sec']}s<br>"
            f"Median: {row['median_sec']}s<br>"
            f"P95: {row['p95_sec']}s<br>"
            f"Max: {row['max_sec']}s<br>"
            f"Next stop: {row.get(COL['stop_name'])}"
        )

        folium.PolyLine(
            [
                [row[COL["from_lat"]], row[COL["from_lon"]]],
                [row[COL["to_lat"]], row[COL["to_lon"]]],
            ],
            color=color,
            weight=7,
            opacity=0.9,
            tooltip=tooltip,
        ).add_to(fmap)

    st_folium(fmap, width="100%", height=650, returned_objects=[])


def render_selected_segment(row: dict, scale_df: pl.DataFrame) -> None:
    scale = colormap(scale_df, COL["travel_time"], "Travel time (seconds)")
    color = scale(float(row[COL["travel_time"]]))

    center_lat = (float(row[COL["from_lat"]]) + float(row[COL["to_lat"]])) / 2
    center_lon = (float(row[COL["from_lon"]]) + float(row[COL["to_lon"]])) / 2

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles="OpenStreetMap")
    scale.add_to(fmap)

    folium.CircleMarker(
        location=[row[COL["from_lat"]], row[COL["from_lon"]]],
        radius=8,
        color="blue",
        fill=True,
        fill_color="blue",
        fill_opacity=0.85,
        tooltip="Previous VP",
        popup=folium.Popup(
            f"<b>Previous VP</b><br>"
            f"{row[COL['from_lat']]}, {row[COL['from_lon']]}<br>"
            f"{row.get(COL['from_ts'])}",
            max_width=350,
        ),
    ).add_to(fmap)

    folium.CircleMarker(
        location=[row[COL["to_lat"]], row[COL["to_lon"]]],
        radius=8,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.9,
        tooltip="Current VP",
        popup=folium.Popup(
            f"<b>Current VP</b><br>"
            f"{row[COL['to_lat']]}, {row[COL['to_lon']]}<br>"
            f"{row.get(COL['to_ts'])}<br>"
            f"Travel time: {row[COL['travel_time']]}s<br>"
            f"Next stop: {row.get(COL['stop_name'])}<br>"
            f"Arrival: {row.get(COL['stop_arrival'])}",
            max_width=350,
        ),
    ).add_to(fmap)

    if row.get(COL["stop_lat"]) is not None and row.get(COL["stop_lon"]) is not None:
        folium.CircleMarker(
            location=[row[COL["stop_lat"]], row[COL["stop_lon"]]],
            radius=7,
            color="black",
            fill=True,
            fill_color="black",
            fill_opacity=0.75,
            tooltip="Next stop",
            popup=folium.Popup(
                f"<b>Next stop</b><br>"
                f"{row.get(COL['stop_id'])} - {row.get(COL['stop_name'])}<br>"
                f"Arrival: {row.get(COL['stop_arrival'])}<br>"
                f"Departure: {row.get(COL['stop_departure'])}",
                max_width=350,
            ),
        ).add_to(fmap)

    folium.PolyLine(
        [
            [row[COL["from_lat"]], row[COL["from_lon"]]],
            [row[COL["to_lat"]], row[COL["to_lon"]]],
        ],
        color=color,
        weight=7,
        opacity=0.95,
        tooltip=f"{row[COL['travel_time']]}s",
    ).add_to(fmap)

    st_folium(fmap, width="100%", height=520, returned_objects=[])


def render_clickable_table(df: pl.DataFrame) -> None:
    st.subheader("Clickable vehicle-position segments")

    display_cols = [
        COL["route"],
        COL["trip"],
        COL["shape"],
        COL["travel_time"],
        COL["from_lat"],
        COL["from_lon"],
        COL["to_lat"],
        COL["to_lon"],
        COL["stop_id"],
        COL["stop_name"],
        COL["stop_arrival"],
        COL["stop_departure"],
        COL["from_ts"],
        COL["to_ts"],
        COL["vehicle_id"],
        COL["vehicle_label"],
    ]

    display_cols = [c for c in display_cols if c in df.columns]

    table_df = (
        df.select(display_cols)
        .sort([COL["route"], COL["trip"], COL["shape"], COL["from_ts"]])
    )

    event = st.dataframe(
        table_df.to_pandas(),
        width="stretch",
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        key="segment_table",
        column_config={
            COL["travel_time"]: st.column_config.ProgressColumn(
                "Travel time (s)",
                min_value=int(df[COL["travel_time"]].min() or 0),
                max_value=int(df[COL["travel_time"]].max() or 1),
                format="%d s",
            ),
            COL["from_lat"]: st.column_config.NumberColumn("From lat", format="%.6f"),
            COL["from_lon"]: st.column_config.NumberColumn("From lon", format="%.6f"),
            COL["to_lat"]: st.column_config.NumberColumn("To lat", format="%.6f"),
            COL["to_lon"]: st.column_config.NumberColumn("To lon", format="%.6f"),
        },
    )

    st.divider()
    st.subheader("Selected segment")

    selected = event.selection.rows
    if not selected:
        st.info("Click a row to show individual points.")
        return

    row = table_df.row(selected[0], named=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Route", row.get(COL["route"]))
    c2.metric("Trip", row.get(COL["trip"]))
    c3.metric("Shape", row.get(COL["shape"]))
    c4.metric("Travel time", f"{row.get(COL['travel_time'])}s")

    render_selected_segment(row, df)


def render_worst_table(worst: pl.DataFrame) -> None:
    st.subheader("Worst performing road segments")

    if worst.is_empty():
        st.info("No segments to show.")
        return

    st.dataframe(
        worst.head(250).to_pandas(),
        width="stretch",
        hide_index=True,
        column_config={
            "p95_sec": st.column_config.ProgressColumn(
                "P95 seconds",
                min_value=float(worst["p95_sec"].min() or 0),
                max_value=float(worst["p95_sec"].max() or 1),
                format="%.1f s",
            ),
            "avg_sec": st.column_config.NumberColumn("Avg seconds", format="%.1f s"),
            "median_sec": st.column_config.NumberColumn("Median seconds", format="%.1f s"),
            "max_sec": st.column_config.NumberColumn("Max seconds", format="%d s"),
            "observations": st.column_config.NumberColumn("Observations", format="%d"),
        },
    )


def render_metrics(df: pl.DataFrame, worst: pl.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VP segments", f"{df.height:,}")
    c2.metric("Trips", f"{df[COL['trip']].n_unique():,}")
    c3.metric("Road segments", f"{worst.height:,}")
    c4.metric("Avg travel time", f"{df[COL['travel_time']].mean():.1f}s")


def main() -> None:
    args = parse_args()

    st.set_page_config(page_title="GTFS-RT Road Segment Performance", layout="wide")
    st.title("GTFS-RT Road Segment Performance")

    if not args.file.exists():
        st.error(f"CSV file not found: {args.file}")
        st.stop()

    if not args.gtfs.exists():
        st.error(f"GTFS zip not found: {args.gtfs}")
        st.stop()

    df = load_csv(str(args.file))
    shapes = load_gtfs_shapes(str(args.gtfs))

    required = {
        COL["route"],
        COL["trip"],
        COL["shape"],
        COL["from_lat"],
        COL["from_lon"],
        COL["to_lat"],
        COL["to_lon"],
        COL["travel_time"],
    }

    missing = required - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {sorted(missing)}")
        st.stop()

    filtered = apply_filters(df)

    if filtered.is_empty():
        st.warning("No rows after filtering.")
        st.stop()

    min_obs = st.session_state.get("min_obs", 2)
    worst = summarize_worst_segments(filtered, min_obs=min_obs)

    render_metrics(filtered, worst)

    tab_worst, tab_table, tab_raw = st.tabs([
        "Worst road segments",
        "Clickable segment table",
        "Raw data",
    ])

    with tab_worst:
        render_worst_segments_map(worst, shapes)
        render_worst_table(worst)

    with tab_table:
        render_clickable_table(filtered)

    with tab_raw:
        st.dataframe(filtered.to_pandas(), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()