"""ETA Benchmark Visualizer.

Usage:
    DATA_DIRECTORY=/path/to/output DATA_CACHE=/path/to/cache uv run streamlit run scripts/viz.py
"""

import hashlib
import math
import os
import zipfile
from datetime import date
from pathlib import Path

import folium
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

_REPO_ROOT = Path(__file__).parent.parent
DATA_DIRECTORY = Path(os.environ.get("DATA_DIRECTORY", str(_REPO_ROOT / "ref" / "output")))
DATA_CACHE = Path(os.environ.get("DATA_CACHE", str(_REPO_ROOT / "ref" / "viz_cache")))

# Bucket ranges: (name, t_min_sec, t_max_sec, early_threshold, late_threshold)
BUCKET_ORDER = ["0-3", "3-6", "6-10", "10-15"]
BUCKET_COLORS = {"0-3": "#9BC631", "3-6": "#6E9B28", "6-10": "#557630", "10-15": "#1D3C34"}
_BUCKET_RANGES = [
    ("0-3", 0, 180, -30, 90),
    ("3-6", 180, 360, -60, 150),
    ("6-10", 360, 600, -60, 210),
    ("10-15", 600, 900, -90, 270),
]

# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _discover_agencies(output_dir: Path) -> list[str]:
    return sorted(p.name for p in output_dir.iterdir() if p.is_dir() and (p / "GTFS.zip").exists())


def _discover_dates(output_dir: Path, agency: str) -> list[str]:
    dates: list[str] = []
    agency_dir = output_dir / agency
    for month_dir in sorted(agency_dir.iterdir()):
        if not month_dir.is_dir():
            continue
        for f in sorted(month_dir.glob("*.analysis.parquet")):
            dd = f.stem.split(".")[0]
            dates.append(f"{month_dir.name}-{dd.zfill(2)}")
    return dates


def _analysis_path(output_dir: Path, agency: str, date_str: str) -> Path:
    ym, dd = date_str.rsplit("-", 1)
    return output_dir / agency / ym / f"{dd}.analysis.parquet"


def _vp_parquet_path(output_dir: Path, agency: str, date_str: str) -> Path:
    ym, dd = date_str.rsplit("-", 1)
    return output_dir / agency / ym / f"{dd}.vehicle-positions.parquet"


def _tu_parquet_path(output_dir: Path, agency: str, date_str: str) -> Path:
    ym, dd = date_str.rsplit("-", 1)
    return output_dir / agency / ym / f"{dd}.trip-updates.parquet"


def _render_paginated_dataframe(
    df: pl.DataFrame,
    key_prefix: str,
    *,
    column_config: dict | None = None,
    default_page_size: int = 100,
) -> None:
    if df.is_empty():
        st.info("Aucune ligne à afficher.")
        return

    page_sizes = [50, 100, 250, 500, 1000]
    if default_page_size not in page_sizes:
        page_sizes.append(default_page_size)
        page_sizes.sort()

    default_idx = page_sizes.index(default_page_size)
    controls_col1, controls_col2, controls_col3 = st.columns([2, 2, 4])
    with controls_col1:
        page_size = int(st.selectbox("Lignes par page", page_sizes, index=default_idx, key=f"{key_prefix}_page_size"))

    total_rows = df.height
    total_pages = max(1, math.ceil(total_rows / page_size))

    with controls_col2:
        page = int(st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, key=f"{key_prefix}_page"))

    start = (page - 1) * page_size
    length = min(page_size, total_rows - start)
    end = start + length

    with controls_col3:
        st.caption(f"Affichage {start + 1:,}–{end:,} sur {total_rows:,} lignes ({total_pages} pages)")

    page_df = df.slice(start, length)
    st.dataframe(page_df.to_pandas(), width="stretch", hide_index=True, column_config=column_config)


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------


def _cache_dir(path: Path) -> Path:
    mtime = path.stat().st_mtime
    key = hashlib.sha1(f"{path}{mtime}".encode()).hexdigest()[:12]
    return DATA_CACHE / key


def _read_cache(cache_dir: Path, name: str) -> pl.DataFrame | None:
    f = cache_dir / f"{name}.parquet"
    return pl.read_parquet(f) if f.exists() else None


def _write_cache(cache_dir: Path, name: str, df: pl.DataFrame) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache_dir / f"{name}.parquet", compression="zstd")


@st.cache_data
def _load_stop_names(gtfs_zip_str: str) -> pl.DataFrame:
    with zipfile.ZipFile(gtfs_zip_str) as zf:
        with zf.open("stops.txt") as f:
            df = pl.read_csv(f, infer_schema_length=0)
    return df.select(
        [
            pl.col("stop_id"),
            pl.col("stop_name"),
            pl.col("stop_lat").cast(pl.Float64),
            pl.col("stop_lon").cast(pl.Float64),
        ]
    )


@st.cache_data
def _load_route_shapes(gtfs_zip_str: str) -> pl.DataFrame:
    with zipfile.ZipFile(gtfs_zip_str) as zf:
        names = set(zf.namelist())
        if "trips.txt" not in names or "shapes.txt" not in names:
            return pl.DataFrame({"route_id": [], "lats": [], "lons": []})
        with zf.open("trips.txt") as f:
            trips = pl.read_csv(f, infer_schema_length=0).select(["route_id", "shape_id"])
        with zf.open("shapes.txt") as f:
            shapes = pl.read_csv(f, infer_schema_length=0).select(
                [
                    pl.col("shape_id"),
                    pl.col("shape_pt_lat").cast(pl.Float64),
                    pl.col("shape_pt_lon").cast(pl.Float64),
                    pl.col("shape_pt_sequence").cast(pl.Int64),
                ]
            )

    route_shape_map = (
        trips.filter(pl.col("shape_id").is_not_null() & (pl.col("shape_id") != ""))
        .group_by(["route_id", "shape_id"])
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .unique(subset=["route_id"], keep="first")
        .select(["route_id", "shape_id"])
    )

    return (
        shapes.join(route_shape_map, on="shape_id", how="inner")
        .group_by("route_id")
        .agg(
            [
                pl.col("shape_pt_lat").sort_by("shape_pt_sequence").alias("lats"),
                pl.col("shape_pt_lon").sort_by("shape_pt_sequence").alias("lons"),
            ]
        )
    )


# ---------------------------------------------------------------------------
# Chart data (disk-cached + Streamlit in-memory cached)
# ---------------------------------------------------------------------------


@st.cache_data
def _load_chart_data(path_str: str) -> dict[str, pl.DataFrame]:
    """Compute all chart aggregations for an analysis parquet.

    Results are persisted to DATA_CACHE on first run and reloaded on
    subsequent runs (keyed by file path + mtime).
    """
    path = Path(path_str)
    cd = _cache_dir(path)

    _FRAMES = [
        "summary",
        "bucket_stats",
        "route_overall",
        "route_bucket",
        "stop_overall",
        "stop_bucket",
        "hourly_overall",
        "hourly_bucket",
        "error_sample",
    ]
    cached = {n: _read_cache(cd, n) for n in _FRAMES}
    if all(v is not None for v in cached.values()):
        return cached  # type: ignore[return-value]

    df = pl.read_parquet(path)
    df_ts = df.with_columns(pl.col("pred_time").dt.hour().alias("hour"))

    total = df.height
    accurate = int(df["is_accurate"].sum())
    mean_abs_err = float(df["error_sec"].abs().mean())

    summary = pl.DataFrame({"total": [total], "accurate": [accurate], "mean_abs_err": [mean_abs_err]})

    bucket_stats = (
        df.group_by("time_bucket")
        .agg([pl.len().alias("n"), pl.col("is_accurate").mean().alias("accuracy")])
        .with_columns(pl.col("time_bucket").replace_strict({b: i for i, b in enumerate(BUCKET_ORDER)}, default=99).alias("_ord"))
        .sort("_ord")
        .drop("_ord")
    )

    route_overall = (
        df.group_by("route_id").agg([pl.len().alias("n"), pl.col("is_accurate").mean().alias("accuracy")]).sort("accuracy", descending=True)
    )

    route_bucket = df.group_by(["route_id", "time_bucket"]).agg(pl.col("is_accurate").mean().alias("accuracy"))

    stop_overall = df.group_by("stop_id").agg([pl.len().alias("n"), pl.col("is_accurate").mean().alias("accuracy")]).sort("stop_id")

    stop_bucket = df.group_by(["stop_id", "time_bucket"]).agg(pl.col("is_accurate").mean().alias("accuracy"))

    hourly_overall = df_ts.group_by("hour").agg([pl.len().alias("n"), pl.col("is_accurate").mean().alias("accuracy")]).sort("hour")

    hourly_bucket = (
        df_ts.group_by(["hour", "time_bucket"])
        .agg([pl.len().alias("n"), pl.col("is_accurate").mean().alias("accuracy")])
        .sort(["time_bucket", "hour"])
    )

    error_sample = df_ts.sample(n=min(5000, df_ts.height), seed=42)

    frames = {
        "summary": summary,
        "bucket_stats": bucket_stats,
        "route_overall": route_overall,
        "route_bucket": route_bucket,
        "stop_overall": stop_overall,
        "stop_bucket": stop_bucket,
        "hourly_overall": hourly_overall,
        "hourly_bucket": hourly_bucket,
        "error_sample": error_sample,
    }
    for name, frame in frames.items():
        _write_cache(cd, name, frame)

    return frames


# ---------------------------------------------------------------------------
# Predictions detail (disk-cached + Streamlit in-memory cached)
# ---------------------------------------------------------------------------


@st.cache_data
def _load_predictions_detail(output_dir_str: str, agency: str, date_str: str) -> tuple[pl.DataFrame, str | None]:
    """Join analysis + TU + VP + stop names into one detail table, cached on disk."""
    output_dir = Path(output_dir_str)
    ym, dd = date_str.rsplit("-", 1)

    an_path = output_dir / agency / ym / f"{dd}.analysis.parquet"
    tu_path = output_dir / agency / ym / f"{dd}.trip-updates.parquet"
    vp_path = output_dir / agency / ym / f"{dd}.vehicle-positions.parquet"

    for p, label in [(an_path, "analysis"), (tu_path, "trip-updates"), (vp_path, "vehicle-positions")]:
        if not p.exists():
            return pl.DataFrame(), f"{label}.parquet introuvable : {p}"

    cd = _cache_dir(an_path)
    cached = _read_cache(cd, "predictions_detail")
    if cached is not None:
        return cached, None

    analysis = pl.read_parquet(an_path)

    tu_schema_cols = set(pl.scan_parquet(tu_path).collect_schema().names())
    tu_keep = [
        c
        for c in [
            "trip_id",
            "stop_id",
            "pred_time",
            "pred_arrival",
            "pb_feed_entity_trip_update_vehicle_label",
            "pb_feed_entity_trip_update_vehicle_id",
        ]
        if c in tu_schema_cols
    ]
    tu = (
        pl.read_parquet(tu_path, columns=tu_keep)
        .unique(subset=["trip_id", "stop_id", "pred_time"], keep="first")
        .rename(
            {
                k: v
                for k, v in {
                    "pb_feed_entity_trip_update_vehicle_label": "vehicle_label",
                    "pb_feed_entity_trip_update_vehicle_id": "vehicle_id_tu",
                }.items()
                if k in tu_keep
            }
        )
    )

    vp_schema_cols = set(pl.scan_parquet(vp_path).collect_schema().names())
    vp_keep = [
        c
        for c in [
            "trip_id",
            "stop_id",
            "actual_arrival",
            "pb_feed_entity_vehicle_position_latitude",
            "pb_feed_entity_vehicle_position_longitude",
        ]
        if c in vp_schema_cols
    ]
    vp = pl.read_parquet(vp_path, columns=vp_keep).rename(
        {
            k: v
            for k, v in {
                "pb_feed_entity_vehicle_position_latitude": "vp_lat",
                "pb_feed_entity_vehicle_position_longitude": "vp_lon",
            }.items()
            if k in vp_keep
        }
    )

    df = analysis.join(tu, on=["trip_id", "stop_id", "pred_time"], how="left")
    df = df.join(vp, on=["trip_id", "stop_id"], how="left")

    gtfs_zip = output_dir / agency / "GTFS.zip"
    if gtfs_zip.exists():
        with zipfile.ZipFile(str(gtfs_zip)) as zf:
            with zf.open("stops.txt") as f:
                stops = pl.read_csv(f, infer_schema_length=0).select(
                    [
                        pl.col("stop_id"),
                        pl.col("stop_name"),
                        pl.col("stop_lat").cast(pl.Float64),
                        pl.col("stop_lon").cast(pl.Float64),
                    ]
                )
        df = df.join(stops, on="stop_id", how="left")

    _write_cache(cd, "predictions_detail", df)
    return df, None


# ---------------------------------------------------------------------------
# Benchmark scatter chart
# ---------------------------------------------------------------------------


def _fmt_sec(s: int) -> str:
    sign = "+" if s >= 0 else "-"
    s = abs(s)
    m, sec = divmod(s, 60)
    if m and sec:
        return f"{sign}{m}m {sec}s"
    return f"{sign}{m}m" if m else f"{sign}{sec}s"


def _build_benchmark_chart(error_sample: pl.DataFrame, bucket_stats: pl.DataFrame) -> go.Figure:
    df = error_sample.with_columns(
        pl.when(pl.col("is_accurate"))
        .then(pl.lit("Précis"))
        .when(pl.col("error_sec") > 0)
        .then(pl.lit("Attente excessive"))
        .otherwise(pl.lit("Manqué"))
        .alias("outcome")
    )

    fig = go.Figure()

    for i, (name, t_min, t_max, early, late) in enumerate(_BUCKET_RANGES):
        if i % 2 == 1:
            fig.add_shape(type="rect", x0=t_min, x1=t_max, y0=-600, y1=600, fillcolor="rgba(0,0,0,0.022)", line=dict(width=0), layer="below")
        fig.add_shape(type="rect", x0=t_min, x1=t_max, y0=early, y1=late, fillcolor="rgba(62, 207, 173, 0.28)", line=dict(width=0), layer="below")
        for y in (early, late):
            fig.add_shape(type="line", x0=t_min, x1=t_max, y0=y, y1=y, line=dict(color="rgba(50, 180, 150, 0.7)", width=1.5))
        fig.add_annotation(
            x=t_max, y=late, text=_fmt_sec(late), showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=10, color="#aaa")
        )
        fig.add_annotation(
            x=t_max, y=early, text=_fmt_sec(early), showarrow=False, xanchor="right", yanchor="top", font=dict(size=10, color="#aaa")
        )

    for x in [180, 360, 600]:
        fig.add_vline(x=x, line_dash="dot", line_color="rgba(160,160,160,0.4)", line_width=1)
    fig.add_hline(y=0, line_color="rgba(0,0,0,0.45)", line_width=1.2)

    fig.add_vline(x=0, line_color="#3B7DD8", line_width=2)
    fig.add_annotation(
        x=0, y=1.04, xref="x", yref="paper", text="<b>ARRIVÉE</b>", showarrow=False, font=dict(size=11, color="#3B7DD8"), xanchor="center"
    )

    _COLORS = {"Précis": "#3ecfad", "Attente excessive": "#f5c518", "Manqué": "#e84b56"}
    for outcome, color in _COLORS.items():
        subset = df.filter(pl.col("outcome") == outcome)
        if subset.is_empty():
            continue
        tta = subset["time_to_arrival_sec"].to_list()
        customdata = list(
            zip(
                [_fmt_sec(int(e)) for e in subset["error_sec"].to_list()],
                subset["time_bucket"].to_list(),
                [f"{int(t) // 60}m {int(t) % 60:02d}s" for t in tta],
            )
        )
        fig.add_trace(
            go.Scatter(
                x=tta,
                y=subset["error_sec"].to_list(),
                mode="markers",
                name=outcome,
                customdata=customdata,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Erreur : %{customdata[0]}<br>"
                    "Temps avant arrivée : %{customdata[2]}<br>"
                    "Tranche : %{customdata[1]} min avant"
                    "<extra></extra>"
                ),
                marker=dict(color=color, size=6, opacity=0.55, line=dict(width=0)),
            )
        )

    bstats = dict(zip(bucket_stats["time_bucket"].to_list(), bucket_stats["accuracy"].to_list()))
    for name, t_min, t_max, _, _ in _BUCKET_RANGES:
        mid = (t_min + t_max) / 2
        acc = bstats.get(name, 0)
        fig.add_annotation(
            x=mid, y=-0.08, xref="x", yref="paper", text=f"<span style='color:#999;font-size:11px'>{name} min avant</span>", showarrow=False
        )
        fig.add_annotation(x=mid, y=-0.18, xref="x", yref="paper", text=f"<b style='font-size:20px'>{acc:.0%}</b>", showarrow=False)

    overall = sum(bstats.get(b[0], 0) for b in _BUCKET_RANGES) / len(_BUCKET_RANGES)
    fig.add_annotation(
        x=0.5,
        y=-0.29,
        xref="paper",
        yref="paper",
        text=f"<b>{overall:.1%}</b>  <span style='color:#999;font-size:13px'>global</span>",
        showarrow=False,
        font=dict(size=22),
    )

    y_ticks = list(range(-360, 361, 60))
    fig.update_layout(
        xaxis=dict(
            range=[960, -30],
            tickvals=[0, 180, 360, 600, 900],
            ticktext=["Arrivée", "3 min", "6 min", "10 min", "15 min"],
            showgrid=False,
            tickfont=dict(size=11, color="#555"),
            ticklen=0,
        ),
        yaxis=dict(
            tickvals=y_ticks,
            ticktext=[_fmt_sec(s) for s in y_ticks],
            tickfont=dict(size=10, color="#999"),
            zeroline=False,
            gridcolor="rgba(220,220,220,0.5)",
            range=[-420, 360],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12)),
        margin=dict(b=130, t=50, l=70, r=20),
        height=560,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="ARPI Viz Benchmark", layout="wide")

st.markdown(
    """
<style>
/* sidebar — always dark */
[data-testid="stSidebar"] {
    background: #1D3C34;
    border-right: 3px solid #9BC631;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #FAFFF8; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label { color: #9BC631 !important; }
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] .stCaption p { color: #C4D9E4 !important; }
[data-testid="stSidebar"] code {
    background: rgba(155, 198, 49, 0.12);
    color: #C4D9E4 !important;
    border-radius: 2px;
    padding: 1px 5px;
}

/* ground — light default */
.stApp { background: #FAFFF8; }

/* typography — light default */
h1 {
    color: #1D3C34;
    font-weight: 700;
    letter-spacing: -0.4px;
    border-bottom: 3px solid #9BC631;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}
h2, h3 { color: #1D3C34; font-weight: 600; }

/* tabs — light default */
[data-testid="stTabs"] [role="tablist"] { border-bottom: 2px solid #C4D9E4; gap: 2px; }
[data-testid="stTab"] {
    font-weight: 500;
    padding: 8px 22px;
    border-radius: 4px 4px 0 0;
    border-bottom: 3px solid transparent;
    transition: color 0.15s, border-color 0.15s, background 0.15s;
}
[data-testid="stTab"] p { color: #6D777A !important; }
[data-testid="stTab"][aria-selected="true"] {
    font-weight: 700;
    border-bottom-color: #9BC631;
    background: rgba(155, 198, 49, 0.14);
}
[data-testid="stTab"][aria-selected="true"],
[data-testid="stTab"][aria-selected="true"] * { color: #1D3C34 !important; }
[data-testid="stTab"]:hover p { color: #557630 !important; }

/* metrics — light default */
[data-testid="stMetric"] {
    background: #FAFFF8;
    border: 1px solid #C4D9E4;
    border-left: 4px solid #9BC631;
    border-radius: 4px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] p {
    color: #6D777A !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
[data-testid="stMetricValue"] { color: #1D3C34 !important; font-weight: 700 !important; }

/* divider */
hr { border: none; border-top: 1px solid #C4D9E4; margin: 1.5rem 0; }

/* form labels — light default */
[data-testid="stSelectbox"] label,
[data-testid="stRadio"] label {
    color: #6D777A;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* data table */
[data-testid="stDataFrame"] { border: 1px solid #C4D9E4; border-radius: 4px; overflow: hidden; }

/* caption */
.stCaption, .stCaption p { color: #6D777A !important; font-size: 0.75rem; }

/* scrollbar — light default */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #FAFFF8; }
::-webkit-scrollbar-thumb { background: #C4D9E4; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #557630; }

/* ── dark mode overrides ────────────────────────── */
@media (prefers-color-scheme: dark) {
    .stApp { background: unset; }

    h1 { color: #9BC631; border-bottom-color: #557630; }
    h2, h3 { color: #C4D9E4; }

    [data-testid="stTabs"] [role="tablist"] { border-bottom-color: rgba(196, 217, 228, 0.2); }
    [data-testid="stTab"] p { color: #6D777A !important; }
    [data-testid="stTab"][aria-selected="true"] { background: rgba(155, 198, 49, 0.1); }
    [data-testid="stTab"][aria-selected="true"] p { color: #FAFFF8 !important; }
    [data-testid="stTab"]:hover p { color: #9BC631 !important; }

    [data-testid="stMetric"] { background: rgba(29, 60, 52, 0.35); border-color: rgba(196, 217, 228, 0.15); }
    [data-testid="stMetricLabel"] p { color: #C4D9E4 !important; }
    [data-testid="stMetricValue"] { color: #FAFFF8 !important; }

    hr { border-top-color: rgba(196, 217, 228, 0.15); }

    [data-testid="stSelectbox"] label,
    [data-testid="stRadio"] label { color: #C4D9E4; }

    [data-testid="stDataFrame"] { border-color: rgba(196, 217, 228, 0.15); }
    .stCaption, .stCaption p { color: #6D777A !important; }

    ::-webkit-scrollbar-track { background: #1a1a1a; }
    ::-webkit-scrollbar-thumb { background: #557630; }
    ::-webkit-scrollbar-thumb:hover { background: #9BC631; }
}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("ARPI")
    st.caption("Visualiseur de benchmark ETA")
    st.divider()
    st.header("Données")
    output_dir = DATA_DIRECTORY

    if not output_dir.exists():
        st.error(f"DATA_DIRECTORY introuvable : {output_dir}")
        st.stop()

    agencies = _discover_agencies(output_dir)
    if not agencies:
        st.error("Aucune agence trouvée (DATA_DIRECTORY/{AGENCE}/GTFS.zip attendu).")
        st.stop()

    agency = st.selectbox("Agence", agencies)
    dates = _discover_dates(output_dir, agency)
    if not dates:
        st.error(f"Aucune analyse trouvée pour {agency}.")
        st.stop()

    dates_dt = [date.fromisoformat(d) for d in dates]
    selected_date = st.date_input(
        "Date",
        value=dates_dt[-1],
        min_value=dates_dt[0],
        max_value=dates_dt[-1],
    )
    date_str = selected_date.isoformat()
    if date_str not in dates:
        st.warning("Aucune donnée pour cette date.")
        st.stop()
    st.caption(f"`DATA_DIRECTORY` {output_dir}")
    st.caption(f"`DATA_CACHE` {DATA_CACHE}")

path = _analysis_path(output_dir, agency, date_str)
if not path.exists():
    st.error(f"Fichier introuvable : {path}")
    st.stop()

data = _load_chart_data(str(path))

summary = data["summary"].row(0, named=True)
bucket_stats = data["bucket_stats"]
route_overall = data["route_overall"]
route_bucket = data["route_bucket"]
stop_overall = data["stop_overall"]
stop_bucket = data["stop_bucket"]
hourly_overall = data["hourly_overall"]
hourly_bucket = data["hourly_bucket"]
error_sample = data["error_sample"]

tab_summary, tab_routes, tab_stops, tab_timeline, tab_predictions, tab_vp_raw, tab_tu_raw = st.tabs(
    ["Sommaire", "Par ligne", "Par arrêt", "Chronologie", "Prédictions", "Vehicle Positions", "Trip Updates"]
)

# ---------------------------------------------------------------------------
# Tab — Prédictions
# ---------------------------------------------------------------------------

with tab_predictions:
    st.subheader("Prédictions — Tableau détaillé")
    vp_path = _vp_parquet_path(output_dir, agency, date_str)
    tu_path = _tu_parquet_path(output_dir, agency, date_str)
    analysis_path = _analysis_path(output_dir, agency, date_str)
    if not tu_path.exists():
        st.info(f"Fichier trip-updates.parquet introuvable : {tu_path}")
    elif not vp_path.exists():
        st.info(f"Fichier vehicle-positions.parquet introuvable : {vp_path}")
    elif not analysis_path.exists():
        st.info(f"Fichier analysis.parquet introuvable : {analysis_path}")
    else:
        with st.spinner("Chargement des prédictions..."):
            pred_detail, pred_err = _load_predictions_detail(str(output_dir), agency, date_str)

        if pred_err:
            st.error(pred_err)
            st.stop()

        if pred_detail.is_empty():
            st.info("Aucune prédiction disponible.")
            st.stop()

        # Status bar on its own row
        st.success(f"{pred_detail.height:,} prédictions")

        # Buttons underneath
        col_clear, col_download = st.columns([1, 1])
        with col_download:
            st.download_button(
                "Télécharger (CSV)",
                data=pred_detail.write_csv().encode("utf-8"),
                file_name=f"{agency}_{date_str}_predictions_detail.csv",
                mime="text/csv",
                key="dl_pred_detail",
                width="stretch",
            )

        with col_clear:
            if st.button("Décharger la carte", key="btn_clear_pred_detail", width="stretch"):
                _load_predictions_detail.clear()
                st.rerun()

        st.divider()

        # --- Paginated table with row selection ---
        _DISPLAY_COLS = [
            "route_id",
            "trip_id",
            "stop_id",
            "stop_name",
            "time_bucket",
            "is_accurate",
            "pred_time",
            "pred_arrival",
            "actual_arrival",
            "error_sec",
            "time_to_arrival_sec",
            "vehicle_label",
        ]
        display_cols = [c for c in _DISPLAY_COLS if c in pred_detail.columns]

        page_sizes_pred = [50, 100, 250, 500]
        ctrl1, ctrl2 = st.columns([1, 1])
        with ctrl1:
            pred_page_size = int(st.selectbox("Lignes par page", page_sizes_pred, index=1, key="pred_page_size"))
        total_pred_rows = pred_detail.height
        total_pred_pages = max(1, math.ceil(total_pred_rows / pred_page_size))

        with ctrl2:
            pred_page = int(st.number_input("Page", min_value=1, max_value=total_pred_pages, value=1, step=1, key="pred_page"))
        pred_start = (pred_page - 1) * pred_page_size
        pred_length = min(pred_page_size, total_pred_rows - pred_start)
        pred_end = pred_start + pred_length

        page_full = pred_detail.slice(pred_start, pred_length)
        page_display = page_full.select(display_cols)

        event = st.dataframe(
            page_display.to_pandas(),
            selection_mode="single-row",
            on_select="rerun",
            hide_index=True,
            width="stretch",
            key=f"pred_table_{pred_page}_{pred_page_size}",
            column_config={
                "is_accurate": st.column_config.CheckboxColumn("Précis"),
                "error_sec": st.column_config.NumberColumn("Erreur (s)", format="%d s"),
                "time_to_arrival_sec": st.column_config.NumberColumn("TTA (s)", format="%d s"),
            },
        )
        st.caption(f"Affichage {pred_start + 1:,}–{pred_end:,} sur {total_pred_rows:,} lignes ({total_pred_pages} pages)")

        # --- Map ---
        st.divider()
        st.subheader("Carte — Position véhicule et arrêt")

        selected_rows = event.selection.rows
        if not selected_rows:
            st.info("Cliquez sur une ligne du tableau pour afficher les positions sur la carte.")
        else:
            row = page_full.row(selected_rows[0], named=True)

            vp_lat = row.get("vp_lat")
            vp_lon = row.get("vp_lon")
            stop_lat = row.get("stop_lat")
            stop_lon = row.get("stop_lon")

            has_vp = vp_lat is not None and vp_lon is not None
            has_stop = stop_lat is not None and stop_lon is not None

            if not has_vp and not has_stop:
                st.warning("Aucune coordonnée disponible pour cette ligne.")
            else:
                lats = [v for v in [vp_lat, stop_lat] if v is not None]
                lons = [v for v in [vp_lon, stop_lon] if v is not None]
                center_lat = sum(lats) / len(lats)
                center_lon = sum(lons) / len(lons)

                zoom = 14 if (has_vp and has_stop) else 15
                fmap = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="OpenStreetMap")

                error_s = row.get("error_sec") or 0
                trip_id = row.get("trip_id", "N/A")
                stop_id = row.get("stop_id", "N/A")
                stop_name = row.get("stop_name") or stop_id
                pred_time = row.get("pred_time", "N/A")
                actual_arr = row.get("actual_arrival", "N/A")
                pred_arr = row.get("pred_arrival", "N/A")
                vehicle_label = row.get("vehicle_label") or row.get("vehicle_id_tu") or "N/A"
                bucket = row.get("time_bucket", "N/A")
                accurate = row.get("is_accurate", False)

                if has_vp:
                    folium.CircleMarker(
                        location=[float(vp_lat), float(vp_lon)],
                        radius=9,
                        color="#1D3C34",
                        fill=True,
                        fill_color="#1D3C34",
                        fill_opacity=0.85,
                        weight=2,
                        popup=folium.Popup(
                            f"<b>Véhicule</b><br>"
                            f"Trip : {trip_id}<br>"
                            f"Véhicule : {vehicle_label}<br>"
                            f"Tranche : {bucket}<br>"
                            f"Erreur : {int(error_s)} s<br>"
                            f"Précis : {'oui' if accurate else 'non'}",
                            max_width=280,
                        ),
                        tooltip="Véhicule",
                    ).add_to(fmap)

                if has_stop:
                    folium.CircleMarker(
                        location=[float(stop_lat), float(stop_lon)],
                        radius=9,
                        color="#9BC631",
                        fill=True,
                        fill_color="#9BC631",
                        fill_opacity=0.85,
                        weight=2,
                        popup=folium.Popup(
                            f"<b>Arrêt : {stop_name}</b><br>"
                            f"ID : {stop_id}<br>"
                            f"Prédiction : {pred_time}<br>"
                            f"Arrivée prédite : {pred_arr}<br>"
                            f"Arrivée réelle : {actual_arr}",
                            max_width=280,
                        ),
                        tooltip=f"Arrêt : {stop_name}",
                    ).add_to(fmap)

                if has_vp and has_stop:
                    folium.PolyLine(
                        [[float(vp_lat), float(vp_lon)], [float(stop_lat), float(stop_lon)]],
                        color="#C4D9E4",
                        weight=2,
                        dash_array="6",
                        opacity=0.7,
                    ).add_to(fmap)

                st_folium(fmap, width="100%", height=420, returned_objects=[], key=f"pred_map_{pred_page}_{pred_page_size}_{selected_rows[0]}")
                st.caption("Marqueur vert = Arrêt  |  Marqueur sombre = Position véhicule")

# ---------------------------------------------------------------------------
# Tab — Raw Vehicle Positions
# ---------------------------------------------------------------------------

with tab_vp_raw:
    st.subheader("Vehicle Positions — données brutes")
    vp_path = _vp_parquet_path(output_dir, agency, date_str)
    if not vp_path.exists():
        st.info(f"Fichier vehicle-positions.parquet introuvable : {vp_path}")
    else:
        st.download_button(
            "Télécharger Vehicle Positions (CSV)",
            data=pl.scan_parquet(vp_path).collect().write_csv().encode("utf-8"),
            file_name=f"{agency}_{date_str}_vp.csv",
            mime="text/csv",
            key="download_vp_raw_csv",
        )

        total_vp_rows = pl.scan_parquet(vp_path).select(pl.len()).collect().item()
        col_ps, col_pg = st.columns(2)
        with col_ps:
            vp_page_size = int(st.number_input("Lignes par page", min_value=100, max_value=2000, value=400, step=100, key="vp_page_size"))

        total_vp_pages = max(1, math.ceil(total_vp_rows / vp_page_size))
        with col_pg:
            vp_page = int(st.number_input("Page", min_value=1, max_value=total_vp_pages, value=1, step=1, key="vp_page"))

        vp_start = (vp_page - 1) * vp_page_size
        st.caption(
            f"Page {vp_page}/{total_vp_pages} · {vp_start + 1:,}–{min(vp_start + vp_page_size, total_vp_rows):,} sur {total_vp_rows:,} lignes"
        )

        vp_raw = pl.scan_parquet(vp_path).slice(vp_start, vp_page_size).collect()
        lat_col = "pb_feed_entity_vehicle_position_latitude"
        lon_col = "pb_feed_entity_vehicle_position_longitude"
        if lat_col in vp_raw.columns and lon_col in vp_raw.columns:
            map_df = (
                vp_raw.select([lat_col, lon_col])
                .drop_nulls()
                .rename({lat_col: "lat", lon_col: "lon"})
                .to_pandas()
                .astype({"lat": "float64", "lon": "float64"})
            )

            if not map_df.empty:
                st.map(map_df, width="stretch", color="#9BC631", size=1)

        st.divider()
        st.dataframe(vp_raw.to_pandas(), width="stretch", hide_index=True)

        file_col = next((c for c in ["pb_feed_file", "file"] if c in vp_raw.columns), None)
        if file_col:
            fig_vp = px.bar(
                vp_raw.group_by(file_col).agg(pl.len().alias("n")).sort(file_col).to_pandas(),
                x=file_col,
                y="n",
                labels={file_col: "Fichier", "n": "Lignes"},
                title="Lignes par fichier Vehicle Positions",
            )
            st.plotly_chart(fig_vp, width="stretch", key="vp_raw_by_file")


# ---------------------------------------------------------------------------
# Tab — Raw Trip Updates
# ---------------------------------------------------------------------------

with tab_tu_raw:
    st.subheader("Trip Updates — données brutes")
    tu_path = _tu_parquet_path(output_dir, agency, date_str)
    if not tu_path.exists():
        st.info(f"Fichier trip-updates.parquet introuvable : {tu_path}")
    else:
        total_tu_rows = pl.scan_parquet(tu_path).select(pl.len()).collect().item()
        st.download_button(
            "Télécharger Trip Updates (CSV)",
            data=pl.scan_parquet(tu_path).collect().write_csv().encode("utf-8"),
            file_name=f"{agency}_{date_str}_tu.csv",
            mime="text/csv",
            key="download_tu_raw_csv",
        )

        col_ps, col_pg = st.columns(2)
        with col_ps:
            tu_page_size = int(st.number_input("Lignes par page", min_value=100, max_value=2000, value=400, step=100, key="tu_page_size"))
        total_tu_pages = max(1, math.ceil(total_tu_rows / tu_page_size))
        with col_pg:
            tu_page = int(st.number_input("Page", min_value=1, max_value=total_tu_pages, value=1, step=1, key="tu_page"))

        tu_start = (tu_page - 1) * tu_page_size
        tu_raw = pl.scan_parquet(tu_path).slice(tu_start, tu_page_size).collect()
        st.dataframe(tu_raw.to_pandas(), width="stretch", hide_index=True)
        st.caption(
            f"Page {tu_page}/{total_tu_pages} · {tu_start + 1:,}–{min(tu_start + tu_page_size, total_tu_rows):,} sur {total_tu_rows:,} lignes"
        )

        file_col = next((c for c in ["pb_feed_file", "file"] if c in tu_raw.columns), None)
        if file_col:
            fig_tu_file = px.bar(
                tu_raw.group_by(file_col).agg(pl.len().alias("n")).sort(file_col).to_pandas(),
                x=file_col,
                y="n",
                labels={file_col: "Fichier", "n": "Lignes"},
                title="Lignes par fichier Trip Updates",
            )
            st.plotly_chart(fig_tu_file, width="stretch", key="tu_raw_by_file")


# ---------------------------------------------------------------------------
# Tab — Summary
# ---------------------------------------------------------------------------

with tab_summary:
    c1, c2, c3 = st.columns(3)
    c1.metric("Prédictions totales", f"{summary['total']:,}")
    c2.metric("Précision globale", f"{summary['accurate'] / summary['total'] * 100:.1f}%")
    c3.metric("Erreur absolue moyenne", f"{summary['mean_abs_err']:.1f}s")

    st.divider()
    st.subheader("Précision ETA par tranche")
    st.plotly_chart(_build_benchmark_chart(error_sample, bucket_stats), width="stretch", key="summary_benchmark")

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Précision par tranche")
        fig = px.bar(
            bucket_stats.to_pandas(),
            x="time_bucket",
            y="accuracy",
            text_auto=".1%",
            color="accuracy",
            color_continuous_scale="RdYlGn",
            range_color=[0, 1],
            labels={"time_bucket": "Temps avant arrivée (min)", "accuracy": "Précision", "n": "Prédictions"},
            category_orders={"time_bucket": BUCKET_ORDER},
            hover_data=["n"],
        )
        fig.update_layout(coloraxis_showscale=False, yaxis_tickformat=".0%")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width="stretch", key="summary_bucket_bar")

    with col_right:
        st.subheader("Distribution des erreurs")
        fig2 = px.histogram(
            error_sample.to_pandas(),
            x="error_sec",
            color="time_bucket",
            barmode="overlay",
            opacity=0.65,
            nbins=80,
            labels={"error_sec": "Erreur (s)", "time_bucket": "Tranche"},
            category_orders={"time_bucket": BUCKET_ORDER},
            color_discrete_map=BUCKET_COLORS,
        )
        fig2.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="À l'heure")
        st.plotly_chart(fig2, width="stretch", key="summary_error_hist")

# ---------------------------------------------------------------------------
# Tab — By Route
# ---------------------------------------------------------------------------

with tab_routes:
    pivot = route_bucket.pivot(on="time_bucket", index="route_id", values="accuracy", aggregate_function="mean")
    available_buckets = [b for b in BUCKET_ORDER if b in pivot.columns]
    route_table = (
        route_overall.join(pivot.select(["route_id"] + available_buckets), on="route_id", how="left")
        .with_columns(pl.col("route_id").cast(pl.Int64, strict=False).alias("_sort_key"))
        .sort("_sort_key")
        .drop("_sort_key")
    )

    accuracy_cols = ["accuracy"] + available_buckets
    route_table_pct = route_table.with_columns([(pl.col(c) * 100).alias(c) for c in accuracy_cols])

    gtfs_zip = output_dir / agency / "GTFS.zip"
    if gtfs_zip.exists():
        route_shapes = _load_route_shapes(str(gtfs_zip))
        map_route_pl = route_table_pct.join(route_shapes, on="route_id", how="inner")

        if not map_route_pl.is_empty():
            st.subheader("Carte de précision par ligne")
            layer_options = ["Global"] + available_buckets
            layer_choice = st.radio(
                "Prédictions",
                options=layer_options,
                format_func=lambda x: x if x == "Global" else f"{x} min",
                horizontal=True,
                key="routes_map_layer",
            )
            active_col = "accuracy" if layer_choice == "Global" else layer_choice

            colormap = LinearColormap(["#e84b56", "#f5c518", "#3ecfad"], vmin=0, vmax=100, caption="Précision %")
            all_lats = route_shapes["lats"].explode().drop_nulls()
            all_lons = route_shapes["lons"].explode().drop_nulls()
            lat_min, lat_max = float(all_lats.min()), float(all_lats.max())
            lon_min, lon_max = float(all_lons.min()), float(all_lons.max())
            span = max(lat_max - lat_min, lon_max - lon_min, 0.001)
            fmap = folium.Map(
                location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2],
                zoom_start=max(1, min(15, int(math.log2(180.0 / span)))),
                tiles="OpenStreetMap",
            )
            colormap.add_to(fmap)
            for row in map_route_pl.iter_rows(named=True):
                acc = row.get(active_col)
                color = colormap(acc) if (acc is not None and acc == acc) else "#aaaaaa"
                bucket_lines = "".join(
                    f"<br>{b} min : {row[b]:.0f}%" if (row.get(b) is not None and row.get(b) == row.get(b)) else f"<br>{b} min : N/D"
                    for b in available_buckets
                )
                tooltip = (
                    f"<b>Ligne {row['route_id']}</b>" f"<br>Global : {row['accuracy']:.0f}%" f"{bucket_lines}" f"<br>Prédictions : {int(row['n'])}"
                )
                folium.PolyLine(
                    list(zip(row["lats"], row["lons"])),
                    color=color,
                    weight=4,
                    opacity=0.85,
                    tooltip=tooltip,
                ).add_to(fmap)
            st_folium(fmap, width="100%", height=520, returned_objects=[], key=f"routes_map_{layer_choice}")
            st.divider()

    bucket_col_config = {b: st.column_config.ProgressColumn(f"{b} min", format="%.0f%%", min_value=0, max_value=100) for b in available_buckets}
    _render_paginated_dataframe(
        route_table_pct,
        "route_table",
        column_config={
            "route_id": st.column_config.TextColumn("Ligne"),
            "n": st.column_config.NumberColumn("Prédictions", format="%d"),
            "accuracy": st.column_config.ProgressColumn("Global", format="%.0f%%", min_value=0, max_value=100),
            **bucket_col_config,
        },
        default_page_size=100,
    )

# ---------------------------------------------------------------------------
# Tab — By Stop
# ---------------------------------------------------------------------------

with tab_stops:
    stop_pivot = stop_bucket.pivot(on="time_bucket", index="stop_id", values="accuracy", aggregate_function="mean")
    available_stop_buckets = [b for b in BUCKET_ORDER if b in stop_pivot.columns]
    stop_table = (
        stop_overall.join(stop_pivot.select(["stop_id"] + available_stop_buckets), on="stop_id", how="left")
        .with_columns(pl.col("stop_id").cast(pl.Int64, strict=False).alias("_sort_key"))
        .sort("_sort_key")
        .drop("_sort_key")
    )

    stop_accuracy_cols = ["accuracy"] + available_stop_buckets
    stop_table_pct = stop_table.with_columns([(pl.col(c) * 100).alias(c) for c in stop_accuracy_cols])

    gtfs_zip = output_dir / agency / "GTFS.zip"
    if gtfs_zip.exists():
        stops_meta = _load_stop_names(str(gtfs_zip))
        stop_table_pct = stop_table_pct.join(stops_meta, on="stop_id", how="left").select(
            ["stop_id", "stop_name", "stop_lat", "stop_lon", "n", "accuracy"] + available_stop_buckets
        )

        map_df = stop_table_pct.drop_nulls(subset=["stop_lat", "stop_lon"]).to_pandas()
        if not map_df.empty:
            st.subheader("Carte de précision par arrêt")
            layer_options = ["Global"] + available_stop_buckets
            layer_choice = st.radio(
                "Prédictions",
                options=layer_options,
                format_func=lambda x: x if x == "Global" else f"{x} min",
                horizontal=True,
                key="stops_map_layer",
            )
            active_col = "accuracy" if layer_choice == "Global" else layer_choice

            colormap = LinearColormap(["#e84b56", "#f5c518", "#3ecfad"], vmin=0, vmax=100, caption="Précision %")
            fmap = folium.Map(tiles="OpenStreetMap")
            fmap.fit_bounds(
                [[map_df["stop_lat"].min(), map_df["stop_lon"].min()], [map_df["stop_lat"].max(), map_df["stop_lon"].max()]],
                padding=(20, 20),
            )
            colormap.add_to(fmap)

            for row in map_df.to_dict(orient="records"):
                acc = row.get(active_col)
                color = colormap(acc) if (acc is not None and acc == acc) else "#aaaaaa"
                bucket_lines = "".join(
                    f"<br>{b} min : {row[b]:.0f}%" if (row.get(b) is not None and row.get(b) == row.get(b)) else f"<br>{b} min : N/D"
                    for b in available_stop_buckets
                )
                tooltip = (
                    f"<b>{row['stop_name']}</b> ({row['stop_id']})"
                    f"<br>Global : {row['accuracy']:.0f}%"
                    f"{bucket_lines}"
                    f"<br>Prédictions : {int(row['n'])}"
                )
                folium.CircleMarker(
                    location=[row["stop_lat"], row["stop_lon"]],
                    radius=6,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.85,
                    tooltip=tooltip,
                ).add_to(fmap)

            st_folium(fmap, width="100%", height=520, returned_objects=[], key=f"stops_map_{layer_choice}")
            st.divider()

        stop_name_col = {"stop_name": st.column_config.TextColumn("Nom de l'arrêt")}
        table_df = stop_table_pct.drop(["stop_lat", "stop_lon"])
    else:
        stop_name_col = {}
        table_df = stop_table_pct

    stop_bucket_col_config = {
        b: st.column_config.ProgressColumn(f"{b} min", format="%.0f%%", min_value=0, max_value=100) for b in available_stop_buckets
    }
    _render_paginated_dataframe(
        table_df,
        "stop_table",
        column_config={
            "stop_id": st.column_config.TextColumn("ID arrêt"),
            **stop_name_col,
            "n": st.column_config.NumberColumn("Prédictions", format="%d"),
            "accuracy": st.column_config.ProgressColumn("Global", format="%.0f%%", min_value=0, max_value=100),
            **stop_bucket_col_config,
        },
        default_page_size=100,
    )

# ---------------------------------------------------------------------------
# Tab — Timeline
# ---------------------------------------------------------------------------

with tab_timeline:
    col_vol, col_acc = st.columns(2)

    with col_vol:
        st.subheader("Volume de prédictions par heure")
        fig = px.bar(
            hourly_bucket.with_columns(pl.col("hour").cast(pl.String)).to_pandas(),
            x="hour",
            y="n",
            color="time_bucket",
            barmode="group",
            labels={"hour": "Heure", "n": "Prédictions", "time_bucket": "Tranche"},
            category_orders={"time_bucket": BUCKET_ORDER, "hour": [str(h) for h in range(24)]},
            color_discrete_map=BUCKET_COLORS,
        )
        fig.update_xaxes(tickmode="array", tickvals=[str(h) for h in range(24)])
        st.plotly_chart(fig, width="stretch", key="timeline_volume")

    with col_acc:
        st.subheader("Précision par heure")
        fig2 = px.bar(
            hourly_bucket.with_columns(pl.col("hour").cast(pl.String)).to_pandas(),
            x="hour",
            y="accuracy",
            color="time_bucket",
            barmode="group",
            labels={"hour": "Heure", "accuracy": "Précision", "time_bucket": "Tranche"},
            category_orders={"time_bucket": BUCKET_ORDER, "hour": [str(h) for h in range(24)]},
            color_discrete_map=BUCKET_COLORS,
        )
        fig2.update_layout(yaxis_tickformat=".0%")
        fig2.update_xaxes(tickmode="array", tickvals=[str(h) for h in range(24)])
        st.plotly_chart(fig2, width="stretch", key="timeline_accuracy")

    st.subheader("Erreur au fil du temps")
    fig3 = px.scatter(
        error_sample.to_pandas(),
        x="pred_time",
        y="error_sec",
        color="time_bucket",
        opacity=0.4,
        labels={"pred_time": "Heure de prédiction", "error_sec": "Erreur (s)", "time_bucket": "Tranche"},
        category_orders={"time_bucket": BUCKET_ORDER},
        color_discrete_map=BUCKET_COLORS,
    )
    fig3.add_hline(y=0, line_dash="dash", line_color="black")
    st.plotly_chart(fig3, width="stretch", key="timeline_scatter")
