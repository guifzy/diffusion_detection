from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.shared.video import create_face_regions, iter_sampled_frames, load_metadata, metadata_for_frame


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


def region_diff(left: dict, right: dict, prefix: str) -> dict:
    out = {}
    for key, left_value in left.items():
        right_value = right.get(key)
        if isinstance(left_value, (int, float, np.integer, np.floating)) and isinstance(
            right_value, (int, float, np.integer, np.floating)
        ):
            out[f"{prefix}_{key}_diff"] = float(left_value) - float(right_value)
    return out


def aggregate_video_metrics(frame_metrics: pd.DataFrame, prefixes: tuple[str, ...]) -> dict:
    metric_cols = [
        col
        for col in frame_metrics.columns
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
        regions = create_face_regions(frame, meta["bbox"])
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
            features.update(func(frame, regions))

        rows.append(features)

    return pd.DataFrame(rows)
