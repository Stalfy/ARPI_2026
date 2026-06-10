# Initial release by: Jeremy Ear, Méliza Foulem, Mohamed Lamine Gning, Anis Mehenni, Melody Nadeau, Zakaria Zair
# Copyright (c) 2025, CIMA+
# All rights reserved.

from dataclasses import dataclass
from typing import Any

import polars as pl

from apex_transit_arpi.models.time_bucket import TimeBuckets
from apex_transit_arpi.models.transit import TimePeriod

MIN_HORIZON = 900


@dataclass
class BenchmarkResult:
    df: pl.DataFrame
    statistics: dict[str, Any]


def build(predictions: pl.DataFrame, actuals: pl.DataFrame, transit_agency: str, time_period: TimePeriod) -> BenchmarkResult:
    """Merge predictions and actuals, compute ETA errors, and assign time buckets.

    Args:
        predictions (pl.DataFrame): Predictions DataFrame. Must include
            ``trip_id``, ``stop_id``, ``pred_time``, and ``pred_arrival`` columns.
        actuals (pl.DataFrame): Actuals DataFrame. Must include
            ``trip_id``, ``stop_id``, and ``actual_arrival`` columns.
        transit_agency (TransitAgency): The transit agency enumeration value
            to annotate benchmark rows.
        time_period (TimePeriod): The time period being processed. Predictions
            outside this date will be filtered out.

    Returns:
        tuple[pl.DataFrame, dict]: A tuple containing the benchmark DataFrame
        (first element) and a statistics dictionary (second element) with keys
        ``total_predictions``, ``merged_predictions``, ``merged_predictions_over_15_min``,
        and ``benchmark_predictions``.
    """
    # 0. Initial statistics
    stats = {
        "total_predictions": predictions.height,
        "merged_predictions": 0,
        "merged_predictions_over_15_min": 0,
        "benchmark_predictions": 0,
    }

    if predictions.is_empty() or actuals.is_empty():
        return BenchmarkResult(pl.DataFrame(), stats)

    # Remove predictions outside the target date
    target_date = time_period.start_time.date()
    predictions = predictions.filter(pl.col("pred_time").dt.date() == target_date)
    if predictions.is_empty():
        return BenchmarkResult(pl.DataFrame(), stats)

    # 1. Merge predictions and actuals
    merged = predictions.join(actuals, on=["trip_id", "stop_id"], how="inner")
    stats["merged_predictions"] = merged.height

    if merged.is_empty():
        return BenchmarkResult(pl.DataFrame(), stats)

    # 2. Compute error (predicted - actual) and time-to-arrival (when the prediction was made relative to actual)
    merged = merged.with_columns(
        [
            (pl.col("pred_arrival") - pl.col("actual_arrival")).dt.total_seconds().alias("raw_error_sec"),
            (pl.col("actual_arrival") - pl.col("pred_time")).dt.total_seconds().alias("time_to_arrival_sec"),
            (pl.col("pred_arrival") - pl.col("pred_time")).dt.total_seconds().alias("prediction_horizon_sec"),
        ]
    )

    # Remove rows were prediction was made after actual arrival
    merged = merged.filter(pl.col("time_to_arrival_sec") >= 0)

    # Remove prediction where "prediction" is basically "right now".
    merged = merged.filter(pl.col("prediction_horizon_sec") > 0)

    # If error is approximately 24 hours (82800-90000 sec), adjust by 24h
    # This handles cases where GTFS service day wasn't properly detected
    merged = merged.with_columns(
        pl.when((pl.col("raw_error_sec") >= 82800) & (pl.col("raw_error_sec") <= 90000))
        .then(pl.col("raw_error_sec") - 86400)
        .when((pl.col("raw_error_sec") <= -82800) & (pl.col("raw_error_sec") >= -90000))
        .then(pl.col("raw_error_sec") + 86400)
        .otherwise(pl.col("raw_error_sec"))
        .alias("error_sec")
    ).drop("raw_error_sec")

    # 3. Assign time buckets
    bins = TimeBuckets.bins()
    labels = TimeBuckets.labels()

    # Calculate predictions over 15 minutes before filtering
    stats["merged_predictions_over_15_min"] = merged.filter(pl.col("time_to_arrival_sec") > MIN_HORIZON).height

    merged = merged.with_columns(
        pl.when(pl.col("time_to_arrival_sec") < bins[1])
        .then(pl.lit(labels[0]))
        .when(pl.col("time_to_arrival_sec") < bins[2])
        .then(pl.lit(labels[1]))
        .when(pl.col("time_to_arrival_sec") < bins[3])
        .then(pl.lit(labels[2]))
        .when(pl.col("time_to_arrival_sec") <= bins[4])
        .then(pl.lit(labels[3]))
        .otherwise(pl.lit(None))
        .cast(pl.Utf8)
        .alias("time_bucket")
    )

    # Remove rows with no assigned time bucket
    merged = merged.filter(pl.col("time_bucket").is_not_null())
    stats["benchmark_predictions"] = merged.height

    if merged.is_empty():
        return BenchmarkResult(pl.DataFrame(), stats)

    # 4. Determine if each prediction is accurate based on thresholds for its bucket
    accuracy_condition = None
    for b in TimeBuckets.values():
        cond = (pl.col("time_bucket") == b.name) & (pl.col("error_sec") >= b.early) & (pl.col("error_sec") <= b.late)
        accuracy_condition = cond if accuracy_condition is None else accuracy_condition | cond

    merged = merged.with_columns(accuracy_condition.alias("is_accurate"))
    merged = merged.with_columns(pl.lit(transit_agency).alias("transit_agency"))

    benchmark_df = merged.select(
        [
            "trip_id",
            "stop_id",
            "pred_time",
            "error_sec",
            "time_to_arrival_sec",
            "time_bucket",
            "is_accurate",
            "transit_agency",
            "route_id",
        ]
    )

    return BenchmarkResult(benchmark_df, stats)
