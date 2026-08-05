from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.shared.video import (
    clip_bbox,
    create_face_regions,
    iter_sampled_frames,
    load_metadata,
    metadata_for_frame,
    standardize_frame,
)


FEATURE_MAX_FRAME_SIZE = 640


def masked_values(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if array.ndim == 3:
        values = array[mask == 1]
    else:
        values = array[mask == 1].reshape(-1)
    return values


def entropy_from_hist(hist: np.ndarray) -> float:
    hist = np.asarray(hist, dtype=float)
    total = hist.sum()
    if total <= 0:
        return np.nan
    prob = hist / total
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def numeric_features(row: dict) -> dict:
    return {k: v for k, v in row.items() if isinstance(v, (int, float, np.integer, np.floating))}


def region_contrasts(left: dict, right: dict, prefix: str, epsilon: float = 1e-6) -> dict:
    out = {}
    for key, left_value in left.items():
        right_value = right.get(key)
        if isinstance(left_value, (int, float, np.integer, np.floating)) and isinstance(
            right_value, (int, float, np.integer, np.floating)
        ):
            left_float = float(left_value)
            right_float = float(right_value)
            signed = left_float - right_float
            out[f"{prefix}_{key}_signed_diff"] = signed
            out[f"{prefix}_{key}_abs_diff"] = abs(signed)
            out[f"{prefix}_{key}_norm_diff"] = signed / (abs(left_float) + abs(right_float) + epsilon)
    return out


def circular_distance(left_angle: float, right_angle: float) -> float:
    if not np.isfinite(left_angle) or not np.isfinite(right_angle):
        return np.nan
    return float(abs(np.angle(np.exp(1j * (left_angle - right_angle)))))


def prepare_frame_regions(frame: np.ndarray, bbox, max_size: int = FEATURE_MAX_FRAME_SIZE):
    frame_std, scale = standardize_frame(frame, max_size=max_size)
    scaled_bbox = [float(value) * scale for value in bbox]
    clipped_bbox = clip_bbox(scaled_bbox, frame_std.shape[1], frame_std.shape[0])
    if clipped_bbox is None:
        return None, None
    return frame_std, create_face_regions(frame_std, clipped_bbox)


def aggregate_video_metrics(frame_metrics: pd.DataFrame, prefixes: tuple[str, ...]) -> dict:
    metric_cols = [
        col
        for col in frame_metrics.columns
        if not col.startswith("qc_")
        if any(col.startswith(prefix) or f"_{prefix}_" in col for prefix in prefixes)
    ]
    values = {}
    if not metric_cols:
        return values
    agg = frame_metrics[metric_cols].agg(["mean", "std", "median"])
    for metric in metric_cols:
        values[f"{metric}_mean"] = agg.loc["mean", metric]
        values[f"{metric}_std"] = agg.loc["std", metric]
        values[f"{metric}_median"] = agg.loc["median", metric]
    return values


def extract_frame_metrics(
    video_path: str | Path,
    metadata_path: str | Path,
    metric_functions,
    max_frames: int | None = None,
    label: str | None = None,
) -> pd.DataFrame:
    metadata = load_metadata(metadata_path)
    rows = []
    for frame_idx, frame, frame_count in iter_sampled_frames(video_path, max_frames=max_frames):
        meta, metadata_idx = metadata_for_frame(frame_idx, frame_count, metadata)
        if meta is None:
            continue
        frame_std, regions = prepare_frame_regions(frame, meta["bbox"])
        if regions is None:
            continue

        features = {
            "video_id": Path(video_path).stem,
            "video_name": Path(video_path).name,
            "frame_id": int(frame_idx),
            "frame": int(frame_idx),
            "metadata_idx": metadata_idx,
        }
        if label is not None:
            features["label"] = label

        for func in metric_functions:
            features.update(func(frame_std, regions))

        rows.append(features)

    return pd.DataFrame(rows)
