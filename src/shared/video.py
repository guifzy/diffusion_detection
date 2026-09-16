from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def get_video_properties(video_path: str | Path) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "frame_count": 0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "duration_s": 0.0,
        }
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration_s = float(frame_count / fps) if fps > 0 else 0.0
    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_s": duration_s,
    }


def get_video_frame_count(video_path: str | Path) -> int:
    return int(get_video_properties(video_path)["frame_count"])


def sample_frame_indices(
    frame_count: int,
    max_frames: int | None = None,
    fps: float | None = None,
    sample_fps: float | None = None,
) -> np.ndarray:
    if frame_count <= 0:
        return np.array([], dtype=int)
    if sample_fps is not None and sample_fps > 0 and fps is not None and fps > 0:
        step = max(int(round(float(fps) / float(sample_fps))), 1)
        indices = np.arange(0, frame_count, step, dtype=int)
    else:
        indices = np.arange(frame_count, dtype=int)
    if max_frames is not None and len(indices) > max_frames:
        selected = np.linspace(0, len(indices) - 1, int(max_frames)).astype(int)
        indices = indices[selected]
    return np.unique(indices.astype(int))


def iter_sampled_frames(video_path: str | Path, max_frames: int | None = None, sample_fps: float | None = None):
    properties = get_video_properties(video_path)
    frame_count = int(properties["frame_count"])
    fps = float(properties["fps"])
    indices = sample_frame_indices(frame_count, max_frames=max_frames, fps=fps, sample_fps=sample_fps)
    wanted = set(int(i) for i in indices)

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in wanted:
            yield frame_idx, frame, frame_count
        frame_idx += 1
    cap.release()


def load_video_frames(
    video_path: str | Path,
    max_frames: int | None = None,
    return_indices: bool = False,
    sample_fps: float | None = None,
):
    rows = list(iter_sampled_frames(video_path, max_frames=max_frames, sample_fps=sample_fps))
    if rows:
        indices, frames, _ = zip(*rows)
        frame_count = rows[0][2]
    else:
        indices, frames, frame_count = [], [], 0
    frames_array = np.array(frames)
    if return_indices:
        return frames_array, np.array(indices, dtype=int), frame_count
    return frames_array


def standardize_frame(frame: np.ndarray, max_size: int = 640):
    h, w = frame.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    if scale == 1.0:
        return frame.copy(), scale
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def clip_bbox(bbox, width: int, height: int, min_size: int = 2):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        return None
    return [x1, y1, x2, y2]


def bbox_to_mask(shape: tuple[int, int], bbox) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    clipped = clip_bbox(bbox, w, h)
    if clipped is None:
        return mask
    x1, y1, x2, y2 = clipped
    mask[y1:y2, x1:x2] = 1
    return mask


def polygon_to_mask(shape: tuple[int, int], polygon) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if not polygon:
        return mask
    points = np.asarray(polygon, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return mask
    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, h - 1)
    hull = cv2.convexHull(points[:, :2].astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1)
    return mask


def bbox_from_mask(mask: np.ndarray, min_size: int = 2):
    ys, xs = np.where(mask == 1)
    if xs.size == 0:
        return None
    return clip_bbox([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], mask.shape[1], mask.shape[0], min_size=min_size)


def expand_bbox(bbox, width: int, height: int, padding: float = 0.2):
    clipped = clip_bbox(bbox, width, height)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * padding)
    py = int(bh * padding)
    return [
        max(0, x1 - px),
        max(0, y1 - py),
        min(width, x2 + px),
        min(height, y2 + py),
    ]


def scale_bbox(bbox, scale: float):
    if scale == 0:
        return None
    return [int(v / scale) for v in bbox]


def create_face_regions(frame: np.ndarray, bbox, padding: float = 0.2):
    h, w = frame.shape[:2]
    clipped = clip_bbox(bbox, w, h)
    if clipped is None:
        return None

    return create_region_context(frame.shape, bbox=clipped, padding=padding)


def create_region_context(
    frame_shape,
    bbox,
    polygon=None,
    target_mask: np.ndarray | None = None,
    background_mask: np.ndarray | None = None,
    padding: float = 0.2,
):
    h, w = frame_shape[:2]
    clipped = clip_bbox(bbox, w, h)
    if clipped is None:
        return None

    if target_mask is not None:
        target_mask = (target_mask > 0).astype(np.uint8)
        if target_mask.shape != (h, w):
            return None
        if target_mask.sum() == 0:
            return None
    elif polygon:
        target_mask = polygon_to_mask((h, w), polygon)
        if target_mask.sum() == 0:
            target_mask = bbox_to_mask((h, w), clipped)
    else:
        target_mask = bbox_to_mask((h, w), clipped)

    target_bbox = bbox_from_mask(target_mask) or clipped
    expanded = expand_bbox(target_bbox, w, h, padding=padding)
    if expanded is None:
        return None
    x1p, y1p, x2p, y2p = expanded

    expanded_mask = np.zeros((h, w), dtype=np.uint8)
    expanded_mask[y1p:y2p, x1p:x2p] = 1
    border_mask = np.clip(expanded_mask - target_mask, 0, 1).astype(np.uint8)

    if background_mask is None:
        context_mask = (1 - expanded_mask).astype(np.uint8)
    else:
        context_mask = (background_mask > 0).astype(np.uint8)
        context_mask[target_mask == 1] = 0

    x1, y1, x2, y2 = target_bbox
    return {
        "face": target_mask.astype(np.uint8),
        "border": border_mask,
        "background": context_mask.astype(np.uint8),
        "bbox": (x1, y1, x2, y2),
        "bbox_expanded": (x1p, y1p, x2p, y2p),
    }


def load_metadata(metadata_path: str | Path) -> list[dict[str, Any]]:
    with Path(metadata_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata(metadata: list[dict[str, Any]], metadata_path: str | Path) -> Path:
    path = Path(metadata_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return path


def metadata_index_for_frame(frame_idx: int, frame_count: int, metadata_len: int) -> int | None:
    if metadata_len <= 0:
        return None
    if metadata_len >= frame_count - 3:
        return min(int(frame_idx), metadata_len - 1)
    metadata_frame_indices = sample_frame_indices(frame_count, metadata_len)
    return int(np.argmin(np.abs(metadata_frame_indices - int(frame_idx))))


def metadata_for_frame(frame_idx: int, frame_count: int, metadata: list[dict[str, Any]]):
    indexed_frames = [
        (idx, item.get("frame_id"))
        for idx, item in enumerate(metadata)
        if isinstance(item.get("frame_id"), (int, float))
    ]
    if indexed_frames:
        exact = next((idx for idx, stored_frame in indexed_frames if int(stored_frame) == int(frame_idx)), None)
        if exact is not None:
            return metadata[exact], exact
        nearest = min(indexed_frames, key=lambda pair: abs(int(pair[1]) - int(frame_idx)))[0]
        return metadata[nearest], nearest

    idx = metadata_index_for_frame(frame_idx, frame_count, len(metadata))
    if idx is None:
        return None, None
    return metadata[idx], idx
