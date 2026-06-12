from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def get_video_frame_count(video_path: str | Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def sample_frame_indices(frame_count: int, max_frames: int | None = None) -> np.ndarray:
    if frame_count <= 0:
        return np.array([], dtype=int)
    if max_frames is None or frame_count <= max_frames:
        return np.arange(frame_count, dtype=int)
    return np.linspace(0, frame_count - 1, int(max_frames)).astype(int)


def iter_sampled_frames(video_path: str | Path, max_frames: int | None = None):
    frame_count = get_video_frame_count(video_path)
    indices = sample_frame_indices(frame_count, max_frames)
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


def load_video_frames(video_path: str | Path, max_frames: int | None = None, return_indices: bool = False):
    rows = list(iter_sampled_frames(video_path, max_frames=max_frames))
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


def scale_bbox(bbox, scale: float):
    if scale == 0:
        return None
    return [int(v / scale) for v in bbox]


def create_face_regions(frame: np.ndarray, bbox, padding: float = 0.2):
    h, w = frame.shape[:2]
    clipped = clip_bbox(bbox, w, h)
    if clipped is None:
        return None

    x1, y1, x2, y2 = clipped
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * padding)
    py = int(bh * padding)

    x1p = max(0, x1 - px)
    y1p = max(0, y1 - py)
    x2p = min(w, x2 + px)
    y2p = min(h, y2 + py)

    face_mask = np.zeros((h, w), dtype=np.uint8)
    face_mask[y1:y2, x1:x2] = 1

    expanded_mask = np.zeros((h, w), dtype=np.uint8)
    expanded_mask[y1p:y2p, x1p:x2p] = 1

    border_mask = expanded_mask - face_mask
    background_mask = 1 - expanded_mask

    return {
        "face": face_mask,
        "border": border_mask,
        "background": background_mask,
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
    idx = metadata_index_for_frame(frame_idx, frame_count, len(metadata))
    if idx is None:
        return None, None
    return metadata[idx], idx
