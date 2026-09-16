from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd

from src.shared.core.version import PIPELINE_VERSION
from src.shared.features.extractor import prefixes_for_groups, groups_to_string


TEMPORAL_METADATA_COLUMNS = {
    "video_id",
    "video_name",
    "frame_id",
    "frame",
    "timestamp_s",
    "video_fps",
    "sample_fps",
    "frame_count",
    "duration_s",
    "original_frame_width",
    "original_frame_height",
    "standardized_frame_width",
    "standardized_frame_height",
    "standardized_max_size",
    "metadata_idx",
    "metadata_region_index",
    "region",
    "region_id",
    "region_label",
    "region_type",
    "track_id",
    "region_source",
    "region_confidence",
    "label",
    "feature_groups_used",
    "processed_at",
    "pipeline_version",
}

CANONICAL_TEMPORAL_SIGNAL_TOKENS = (
    "entropy_norm",
    "uniformity",
    "active_bin_ratio",
    "hist_js_distance",
    "magnitude_mean",
    "gradient_energy",
    "orientation_entropy_norm",
    "orientation_coherence",
    "strong_gradient_ratio",
    "signed_variance",
    "signed_energy",
    "abs_mean",
    "abs_p95",
    "rms",
    "mad",
    "channel_corr_mean",
    "horizontal_lag1_corr",
    "vertical_lag1_corr",
    "low_power_ratio",
    "mid_power_ratio",
    "high_power_ratio",
    "spectral_entropy_norm",
    "spectral_flatness",
    "angular_anisotropy",
    "spectral_slope",
    "l_mean",
    "l_contrast_p90_norm",
    "l_entropy_norm",
    "saturation_mean",
    "illumination_mean",
    "illumination_gradient_energy",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
    "candidate_ratio",
    "candidate_depth_mean",
    "boundary_density",
    "kp_density",
    "kp_coverage",
    "response_mean",
    "descriptor_self_similarity",
    "sim_mean",
    "sim_std",
)


def _safe_series(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def _finite_pair(values: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values) & np.isfinite(times)
    return values[valid], times[valid]


def _robust_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_mad": np.nan,
            f"{prefix}_iqr": np.nan,
            f"{prefix}_p95_abs": np.nan,
        }
    median = float(np.median(values))
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_median": median,
        f"{prefix}_mad": float(np.median(np.abs(values - median))),
        f"{prefix}_iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        f"{prefix}_p95_abs": float(np.percentile(np.abs(values), 95)),
    }


def _autocorrelation(values: np.ndarray, lag: int) -> float:
    values = values[np.isfinite(values)]
    if values.size <= lag + 1:
        return np.nan
    left = values[:-lag]
    right = values[lag:]
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _temporal_frequency_stats(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size < 4:
        return {
            "temporal_energy": np.nan,
            "temporal_high_energy_ratio": np.nan,
            "temporal_entropy_norm": np.nan,
        }
    centered = values - np.mean(values)
    spectrum = np.fft.rfft(centered)
    power = np.square(np.abs(spectrum))
    if power.size <= 1:
        return {
            "temporal_energy": np.nan,
            "temporal_high_energy_ratio": np.nan,
            "temporal_entropy_norm": np.nan,
        }
    power = power[1:]
    total = float(np.sum(power) + 1e-12)
    midpoint = max(1, power.size // 2)
    probabilities = power / total
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities)) / np.log2(power.size) if power.size > 1 else 0.0
    return {
        "temporal_energy": float(total),
        "temporal_high_energy_ratio": float(np.sum(power[midpoint:]) / total),
        "temporal_entropy_norm": float(entropy),
    }


def _first_difference(values: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    dt = np.diff(times)
    valid = np.isfinite(dt) & (np.abs(dt) > 1e-12)
    deltas = np.full(dt.shape, np.nan, dtype=float)
    deltas[valid] = np.diff(values)[valid] / dt[valid]
    mid_times = (times[1:] + times[:-1]) / 2.0
    return deltas, mid_times


def _second_difference(first_diff: np.ndarray, diff_times: np.ndarray) -> np.ndarray:
    if first_diff.size < 2:
        return np.array([], dtype=float)
    dt = np.diff(diff_times)
    valid = np.isfinite(dt) & (np.abs(dt) > 1e-12)
    second = np.full(dt.shape, np.nan, dtype=float)
    second[valid] = np.diff(first_diff)[valid] / dt[valid]
    return second


def temporal_metric_columns(frame_features: pd.DataFrame, groups: Iterable[str] | str = "abcde") -> list[str]:
    prefixes = prefixes_for_groups(groups)
    return [
        column
        for column in frame_features.columns
        if column not in TEMPORAL_METADATA_COLUMNS
        and not column.startswith("qc_")
        and pd.api.types.is_numeric_dtype(frame_features[column])
        and any(column.startswith(prefix) or f"_{prefix}_" in column for prefix in prefixes)
        and any(token in column for token in CANONICAL_TEMPORAL_SIGNAL_TOKENS)
    ]


def aggregate_region_temporal_features(
    frame_features: pd.DataFrame,
    groups: Iterable[str] | str = "abcde",
    min_points: int = 3,
) -> pd.DataFrame:
    if frame_features.empty:
        return pd.DataFrame()

    time_column = "timestamp_s" if "timestamp_s" in frame_features.columns else "frame_id"
    metric_cols = temporal_metric_columns(frame_features, groups=groups)
    if not metric_cols:
        return pd.DataFrame()

    keys = [column for column in ("video_id", "region", "region_type", "track_id") if column in frame_features.columns]
    rows = []
    for group_key, group in frame_features.groupby(keys, dropna=False):
        group = group.sort_values([time_column, "frame_id"] if "frame_id" in group.columns else [time_column])
        row = {}
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        for key, value in zip(keys, group_key):
            row[key] = value

        for column in ("label", "region_id", "region_label", "region_source"):
            if column in group.columns and not group[column].dropna().empty:
                row[column] = group[column].dropna().astype(str).iloc[0]

        row["n_temporal_frames"] = int(group["frame_id"].nunique()) if "frame_id" in group.columns else int(len(group))
        row["temporal_min_points"] = int(min_points)
        row["temporal_time_span_s"] = float(group[time_column].max() - group[time_column].min())
        row["feature_groups_used"] = groups_to_string(groups)
        row["aggregated_at"] = datetime.now(timezone.utc).isoformat()
        row["pipeline_version"] = PIPELINE_VERSION

        for metric in metric_cols:
            values, times = _finite_pair(_safe_series(group[metric]), _safe_series(group[time_column]))
            if values.size < min_points:
                continue
            d1, d1_times = _first_difference(values, times)
            d2 = _second_difference(d1, d1_times)
            base = f"temporal__{metric}"
            row.update(_robust_stats(d1, f"{base}__d1"))
            row.update(_robust_stats(d2, f"{base}__d2"))
            row[f"{base}__lag1_autocorr"] = _autocorrelation(values, 1)
            row[f"{base}__lag2_autocorr"] = _autocorrelation(values, 2)
            for key, value in _temporal_frequency_stats(values).items():
                row[f"{base}__{key}"] = value

        metadata_keys = {
            "video_id",
            "region",
            "region_type",
            "track_id",
            "label",
            "region_id",
            "region_label",
            "region_source",
            "n_temporal_frames",
            "temporal_min_points",
            "temporal_time_span_s",
            "feature_groups_used",
            "aggregated_at",
            "pipeline_version",
        }
        feature_values = {key: value for key, value in row.items() if key not in metadata_keys}
        row["temporal_missing_feature_ratio"] = (
            float(pd.Series(feature_values).isna().mean()) if feature_values else 1.0
        )
        rows.append(row)

    return pd.DataFrame(rows)
