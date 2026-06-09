# visualize.py
#
# Usage:
#   uv run streamlit run travel_time_viz.py -- --file travel_times.csv --gtfs GTFS.zip

import argparse
import zipfile
from pathlib import Path

import folium
import polars as pl
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--gtfs", type=Path, required=True)
    return parser.parse_args()


@st.cache_data
def load_csv(path: str) -> pl.DataFrame:
    return pl.read_csv(
        path,
        try_parse_dates=True,
        infer_schema_length=10000,
        schema_overrides={
            "route": pl.String,
            "trip": pl.String,
            "shape": pl.String,
            "destination_stop_id": pl.String,
            "vehicle_id": pl.String,
            "vehicle_label": pl.String,
            "entity_id": pl.String,
            "service_id": pl.String,
            "direction_id": pl.String,
            "block_id": pl.String,
            "route_id_rt": pl.String,
        },
    )


@st.cache_data
def load_gtfs_shapes(gtfs_zip: str) -> pl.DataFrame:
    with zipfile.ZipFile(gtfs_zip) as zf:
        with zf.open("shapes.txt") as f:
            shapes = pl.read_csv(f, infer_schema_length=0)

    return shapes.select(
        [
            pl.col("shape_id").cast(pl.String).alias("shape"),
            pl.col("shape_pt_lat").cast(pl.Float64),
            pl.col("shape_pt_lon").cast(pl.Float64),
            pl.col("shape_pt_sequence").cast(pl.Int64),
        ]
    ).sort(["shape", "shape_pt_sequence"])


def apply_filters(df: pl.DataFrame) -> pl.DataFrame:
    with st.sidebar:
        st.header("Filters")

        routes = sorted(df["route"].cast(pl.String).drop_nulls().unique().to_list())
        selected_route = st.selectbox("Route", ["All"] + routes)

        if selected_route != "All":
            df = df.filter(pl.col("route").cast(pl.String) == selected_route)

        shapes = sorted(df["shape"].cast(pl.String).drop_nulls().unique().to_list())
        selected_shape = st.selectbox("Shape", ["All"] + shapes)

        if selected_shape != "All":
            df = df.filter(pl.col("shape").cast(pl.String) == selected_shape)

        min_obs = st.number_input(
            "Min observations per road segment",
            min_value=1,
            value=2,
            step=1,
        )

        max_time = int(df["travel_time_seconds"].max() or 1)
        max_segment_time = st.slider(
            "Max segment travel time (seconds)",
            min_value=1,
            max_value=max(max_time, 1),
            value=min(1300, max(max_time, 1)),
        )

        df = df.filter(pl.col("travel_time_seconds") <= max_segment_time)

    st.session_state["min_obs"] = int(min_obs)
    return df


def summarize_worst_segments(df: pl.DataFrame, min_obs: int) -> pl.DataFrame:
    return (
        df.drop_nulls(
            [
                "route",
                "shape",
                "from_latitude",
                "from_longitude",
                "to_latitude",
                "to_longitude",
                "travel_time_seconds",
            ]
        )
        .group_by(
            [
                "route",
                "shape",
                "from_latitude",
                "from_longitude",
                "to_latitude",
                "to_longitude",
            ]
        )
        .agg(
            [
                pl.len().alias("observations"),
                pl.col("travel_time_seconds").mean().round(2).alias("avg_sec"),
                pl.col("travel_time_seconds").median().round(2).alias("median_sec"),
                pl.col("travel_time_seconds").quantile(0.95).round(2).alias("p95_sec"),
                pl.col("travel_time_seconds").max().alias("max_sec"),
                pl.col("destination_stop_id").drop_nulls().first().alias("destination_stop_id"),
                pl.col("destination_stop_name").drop_nulls().first().alias("destination_stop_name"),
            ]
        )
        .filter(pl.col("observations") >= min_obs)
        .sort(["p95_sec", "avg_sec"], descending=True)
    )


def travel_time_colormap(df: pl.DataFrame, column: str) -> LinearColormap:
    if df.is_empty() or column not in df.columns:
        return LinearColormap(["green", "yellow", "red"], vmin=0, vmax=1)

    vmin = float(df[column].min() or 0)
    vmax = float(df[column].max() or 1)

    if vmin == vmax:
        vmax = vmin + 1

    return LinearColormap(
        ["green", "yellow", "red"],
        vmin=vmin,
        vmax=vmax,
        caption=f"{column} travel time (seconds)",
    )


def map_center_from_segments(df: pl.DataFrame) -> tuple[float, float]:
    lats = pl.concat([df["from_latitude"], df["to_latitude"]])
    lons = pl.concat([df["from_longitude"], df["to_longitude"]])
    return float(lats.mean()), float(lons.mean())


def render_gtfs_shapes(fmap: folium.Map, shapes: pl.DataFrame, selected_shapes: set[str]) -> None:
    if shapes.is_empty():
        return

    if selected_shapes:
        shapes = shapes.filter(pl.col("shape").is_in(list(selected_shapes)))

    for key, group in shapes.group_by("shape"):
        shape_id = key[0] if isinstance(key, tuple) else key

        pts = list(
            zip(
                group["shape_pt_lat"].to_list(),
                group["shape_pt_lon"].to_list(),
            )
        )

        if len(pts) < 2:
            continue

        folium.PolyLine(
            pts,
            color="#777777",
            weight=3,
            opacity=0.35,
            tooltip=f"GTFS shape {shape_id}",
        ).add_to(fmap)


def render_worst_segments_map(worst: pl.DataFrame, shapes: pl.DataFrame) -> None:
    if worst.is_empty():
        st.info("No road segments after filtering.")
        return

    sample = worst.head(500)
    selected_shapes = set(sample["shape"].cast(pl.String).drop_nulls().unique().to_list())

    center_lat, center_lon = map_center_from_segments(sample)

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="OpenStreetMap",
    )

    render_gtfs_shapes(fmap, shapes, selected_shapes)

    colormap = travel_time_colormap(sample, "p95_sec")
    colormap.add_to(fmap)

    for row in sample.iter_rows(named=True):
        color = colormap(float(row["p95_sec"]))

        tooltip = (
            f"Route: {row['route']}<br>"
            f"Shape: {row['shape']}<br>"
            f"Obs: {row['observations']}<br>"
            f"Avg: {row['avg_sec']}s<br>"
            f"Median: {row['median_sec']}s<br>"
            f"P95: {row['p95_sec']}s<br>"
            f"Max: {row['max_sec']}s<br>"
            f"Destination: {row.get('destination_stop_name')}"
        )

        folium.PolyLine(
            [
                [row["from_latitude"], row["from_longitude"]],
                [row["to_latitude"], row["to_longitude"]],
            ],
            color=color,
            weight=7,
            opacity=0.9,
            tooltip=tooltip,
        ).add_to(fmap)

    st_folium(fmap, width="100%", height=650, returned_objects=[])


def render_selected_segment(row: dict, scale_df: pl.DataFrame) -> None:
    colormap = travel_time_colormap(scale_df, "travel_time_seconds")
    color = colormap(float(row["travel_time_seconds"]))

    center_lat = (float(row["from_latitude"]) + float(row["to_latitude"])) / 2
    center_lon = (float(row["from_longitude"]) + float(row["to_longitude"])) / 2

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=16,
        tiles="OpenStreetMap",
    )
    colormap.add_to(fmap)

    folium.CircleMarker(
        location=[row["from_latitude"], row["from_longitude"]],
        radius=8,
        color="blue",
        fill=True,
        fill_color="blue",
        fill_opacity=0.85,
        tooltip="From",
        popup=folium.Popup(
            f"<b>From</b><br>" f"{row['from_latitude']}, {row['from_longitude']}<br>" f"{row.get('from_timestamp')}",
            max_width=350,
        ),
    ).add_to(fmap)

    folium.CircleMarker(
        location=[row["to_latitude"], row["to_longitude"]],
        radius=8,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.9,
        tooltip="To",
        popup=folium.Popup(
            f"<b>To</b><br>"
            f"{row['to_latitude']}, {row['to_longitude']}<br>"
            f"{row.get('to_timestamp')}<br>"
            f"Travel time: {row['travel_time_seconds']}s<br>"
            f"Destination: {row.get('destination_stop_name')}",
            max_width=350,
        ),
    ).add_to(fmap)

    folium.PolyLine(
        [
            [row["from_latitude"], row["from_longitude"]],
            [row["to_latitude"], row["to_longitude"]],
        ],
        color=color,
        weight=7,
        opacity=0.95,
        tooltip=f"{row['travel_time_seconds']}s",
    ).add_to(fmap)

    st_folium(fmap, width="100%", height=520, returned_objects=[])


def render_selectable_table(df: pl.DataFrame) -> None:
    st.subheader("Individual segment table")

    display_cols = [
        c
        for c in [
            "route",
            "trip",
            "shape",
            "travel_time_seconds",
            "from_latitude",
            "from_longitude",
            "to_latitude",
            "to_longitude",
            "destination_stop_id",
            "destination_stop_name",
            "destination_arrival_time",
            "destination_departure_time",
            "destination_stop_sequence",
            "from_timestamp",
            "to_timestamp",
            "vehicle_id",
            "vehicle_label",
        ]
        if c in df.columns
    ]

    table_df = df.select(display_cols).sort(["route", "trip", "shape", "from_timestamp"])

    event = st.dataframe(
        table_df.to_pandas(),
        width="stretch",
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        key="segment_table",
        column_config={
            "travel_time_seconds": st.column_config.ProgressColumn(
                "Travel time (s)",
                min_value=int(df["travel_time_seconds"].min() or 0),
                max_value=int(df["travel_time_seconds"].max() or 1),
                format="%d s",
            ),
            "from_latitude": st.column_config.NumberColumn("From lat", format="%.6f"),
            "from_longitude": st.column_config.NumberColumn("From lon", format="%.6f"),
            "to_latitude": st.column_config.NumberColumn("To lat", format="%.6f"),
            "to_longitude": st.column_config.NumberColumn("To lon", format="%.6f"),
        },
    )

    st.divider()
    st.subheader("Selected segment")

    selected = event.selection.rows
    if not selected:
        st.info("Click a row to show the segment.")
        return

    row = table_df.row(selected[0], named=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Route", row.get("route"))
    c2.metric("Trip", row.get("trip"))
    c3.metric("Shape", row.get("shape"))
    c4.metric("Travel time", f"{row.get('travel_time_seconds')}s")

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

    c1.metric("Segments", f"{df.height:,}")
    c2.metric("Trips", f"{df['trip'].n_unique():,}")
    c3.metric("Road segments", f"{worst.height:,}")

    avg_time = df["travel_time_seconds"].mean() if df.height else 0
    c4.metric("Avg travel time", f"{avg_time:.1f}s")


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
        "route",
        "trip",
        "shape",
        "from_latitude",
        "from_longitude",
        "to_latitude",
        "to_longitude",
        "travel_time_seconds",
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

    tab_worst, tab_table, tab_raw = st.tabs(
        [
            "Worst road segments",
            "Clickable segment table",
            "Raw data",
        ]
    )

    with tab_worst:
        render_worst_segments_map(worst, shapes)
        render_worst_table(worst)

    with tab_table:
        render_selectable_table(filtered)

    with tab_raw:
        st.dataframe(filtered.to_pandas(), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
