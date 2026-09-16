from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.shared.video import (
    bbox_to_mask,
    clip_bbox,
    create_face_regions,
    create_region_context,
    get_video_properties,
    iter_sampled_frames,
    load_metadata,
    metadata_for_frame,
    polygon_to_mask,
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


def _scaled_bbox(bbox, scale: float, width: int, height: int):
    if not bbox:
        return None
    scaled = [float(value) * scale for value in bbox]
    return clip_bbox(scaled, width, height)


def _scaled_polygon(polygon, scale: float):
    if not polygon:
        return []
    return [[float(point[0]) * scale, float(point[1]) * scale] for point in polygon if len(point) >= 2]


def _annotation_mask(shape: tuple[int, int], region: dict, scale: float) -> np.ndarray:
    h, w = shape
    bbox = _scaled_bbox(region.get("bbox"), scale, w, h)
    polygon = _scaled_polygon(region.get("polygon"), scale)
    if polygon:
        mask = polygon_to_mask((h, w), polygon)
        if mask.sum() > 0:
            return mask
    if bbox is not None:
        return bbox_to_mask((h, w), bbox)
    return np.zeros((h, w), dtype=np.uint8)


def _background_mask_for_regions(shape: tuple[int, int], regions: list[dict], scale: float) -> np.ndarray:
    occupied = np.zeros(shape, dtype=np.uint8)
    for region in regions:
        if region.get("region_type") == "fundo":
            continue
        occupied = np.maximum(occupied, _annotation_mask(shape, region, scale))
    return (1 - occupied).astype(np.uint8)


def prepare_annotated_region_contexts(frame: np.ndarray, meta: dict, max_size: int = FEATURE_MAX_FRAME_SIZE):
    frame_std, scale = standardize_frame(frame, max_size=max_size)
    h, w = frame_std.shape[:2]
    annotations = list(meta.get("regions") or [])
    if not annotations:
        frame_std, legacy_regions = prepare_frame_regions(frame, meta.get("bbox"), max_size=max_size)
        if legacy_regions is None:
            return frame_std, []
        return frame_std, [
            {
                "region": "rosto_completo_1",
                "region_id": "rosto_completo_1",
                "region_label": "rosto completo 1",
                "region_type": "rosto_completo",
                "track_id": "face_1",
                "source": meta.get("source", "legacy"),
                "confidence": meta.get("detector_score"),
                "metadata_region_index": 0,
                "regions": legacy_regions,
            }
        ]

    background_mask = _background_mask_for_regions((h, w), annotations, scale)
    contexts = []
    for index, annotation in enumerate(annotations):
        region_type = annotation.get("region_type", "")
        if region_type == "fundo":
            target_mask = background_mask
            bbox = [0, 0, w, h]
            polygon = []
            context_background = (1 - background_mask).astype(np.uint8)
        else:
            target_mask = _annotation_mask((h, w), annotation, scale)
            bbox = _scaled_bbox(annotation.get("bbox"), scale, w, h)
            polygon = _scaled_polygon(annotation.get("polygon"), scale)
            context_background = background_mask
        if target_mask.sum() == 0 or bbox is None:
            continue
        region_context = create_region_context(
            frame_std.shape,
            bbox=bbox,
            polygon=polygon,
            target_mask=target_mask,
            background_mask=context_background,
        )
        if region_context is None:
            continue
        contexts.append(
            {
                "region": annotation.get("region", annotation.get("region_id", "")),
                "region_id": annotation.get("region_id", annotation.get("region", "")),
                "region_label": annotation.get("region_label", ""),
                "region_type": region_type,
                "track_id": annotation.get("track_id", ""),
                "source": annotation.get("source", meta.get("source", "")),
                "confidence": annotation.get("confidence", meta.get("detector_score")),
                "metadata_region_index": index,
                "regions": region_context,
            }
        )
    return frame_std, contexts


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
    sample_fps: float | None = None,
    label: str | None = None,
) -> pd.DataFrame:
    metadata = load_metadata(metadata_path)
    video_properties = get_video_properties(video_path)
    video_fps = float(video_properties["fps"])
    frame_count_total = int(video_properties["frame_count"])
    duration_s = float(video_properties["duration_s"])
    rows = []
    for frame_idx, frame, frame_count in iter_sampled_frames(video_path, max_frames=max_frames, sample_fps=sample_fps):
        meta, metadata_idx = metadata_for_frame(frame_idx, frame_count, metadata)
        if meta is None:
            continue
        frame_std, region_contexts = prepare_annotated_region_contexts(frame, meta)
        if not region_contexts:
            continue
        timestamp_s = float(frame_idx / video_fps) if video_fps > 0 else float(frame_idx)

        for context in region_contexts:
            features = {
                "video_id": Path(video_path).stem,
                "video_name": Path(video_path).name,
                "frame_id": int(frame_idx),
                "frame": int(frame_idx),
                "timestamp_s": timestamp_s,
                "video_fps": video_fps,
                "sample_fps": float(sample_fps) if sample_fps else video_fps,
                "frame_count": frame_count_total,
                "duration_s": duration_s,
                "original_frame_width": int(video_properties["width"]),
                "original_frame_height": int(video_properties["height"]),
                "standardized_frame_width": int(frame_std.shape[1]),
                "standardized_frame_height": int(frame_std.shape[0]),
                "standardized_max_size": FEATURE_MAX_FRAME_SIZE,
                "metadata_idx": metadata_idx,
                "metadata_region_index": context["metadata_region_index"],
                "region": context["region"],
                "region_id": context["region_id"],
                "region_label": context["region_label"],
                "region_type": context["region_type"],
                "track_id": context["track_id"],
                "region_source": context["source"],
                "region_confidence": context["confidence"],
            }
            if label is not None:
                features["label"] = label

            for func in metric_functions:
                features.update(func(frame_std, context["regions"]))

            rows.append(features)

    return pd.DataFrame(rows)
