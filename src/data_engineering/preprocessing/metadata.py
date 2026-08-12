from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import cv2
import numpy as np

from src.shared.core.io_utils import write_dataframe
from src.shared.core.paths import (
    BRONZE_VIDEOS_DIR,
    METADATA_DIR,
    PROJECT_ROOT,
    metadata_path_for_video,
    silver_face_metadata_path,
)
from src.shared.core.version import PIPELINE_VERSION
from src.shared.video import (
    bbox_from_mask,
    clip_bbox,
    create_face_regions,
    iter_sampled_frames,
    save_metadata,
)

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

logger = logging.getLogger(__name__)


LEFT_EYE_INDICES = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
RIGHT_EYE_INDICES = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
OUTER_MOUTH_INDICES = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
INNER_MOUTH_INDICES = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191)
FACE_REGION_NAME = "rosto_completo"
EYES_REGION_NAME = "olhos"
MOUTH_REGION_NAME = "boca"
BODY_REGION_NAME = "corpo"
BACKGROUND_REGION_NAME = "fundo"


def _default_face_model_path() -> Path:
    env_path = os.environ.get("MEDIAPIPE_FACE_MODEL")
    if env_path:
        return Path(env_path)
    return PROJECT_ROOT / "experimentos" / "grupo_b" / "data" / "extracted" / "face_landmarker.task"


def _default_segmenter_model_path() -> Path:
    env_path = os.environ.get("MEDIAPIPE_SEGMENTER_MODEL")
    if env_path:
        return Path(env_path)
    return PROJECT_ROOT / "models" / "image_segmenter.task"


def _load_mediapipe_modules():
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is required for the current preprocessing pipeline. "
            "Install the dependencies from requirements.txt."
        ) from exc
    return mp, python, vision


def _create_face_landmarker(model_path: str | Path | None = None, max_faces: int = 10):
    mp, python, vision = _load_mediapipe_modules()
    model = Path(model_path) if model_path else _default_face_model_path()
    if not model.exists():
        raise FileNotFoundError(f"MediaPipe face model not found: {model}")
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=max_faces,
    )
    return mp, vision.FaceLandmarker.create_from_options(options)


def _create_image_segmenter(model_path: str | Path | None = None):
    mp, python, vision = _load_mediapipe_modules()
    model = Path(model_path) if model_path else _default_segmenter_model_path()
    if not model.exists():
        logger.warning("MediaPipe segmenter model not found: %s. Body masks will use geometric fallback.", model)
        return mp, None
    options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE,
        output_category_mask=True,
    )
    return mp, vision.ImageSegmenter.create_from_options(options)


def _mp_image(mp_module, frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb_frame)


def _landmark_rows(face_landmarks, width: int, height: int) -> list[dict[str, float]]:
    rows = []
    for idx, landmark in enumerate(face_landmarks):
        px = float(np.clip(landmark.x * (width - 1), 0, width - 1))
        py = float(np.clip(landmark.y * (height - 1), 0, height - 1))
        rows.append(
            {
                "point_id": int(idx),
                "x": float(landmark.x),
                "y": float(landmark.y),
                "z": float(landmark.z),
                "px": px,
                "py": py,
            }
        )
    return rows


def _points_for_indices(points: list[dict[str, float]], indices: tuple[int, ...]) -> list[dict[str, float]]:
    available = {point["point_id"]: point for point in points}
    return [available[index] for index in indices if index in available]


def _polygon_from_points(points: list[dict[str, float]]) -> list[list[float]]:
    if len(points) < 3:
        return []
    coords = np.asarray([[point["px"], point["py"]] for point in points], dtype=np.float32)
    hull = cv2.convexHull(coords.astype(np.int32)).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in hull]


def _bbox_from_points(points: list[dict[str, float]], width: int, height: int):
    if not points:
        return None
    xs = [point["px"] for point in points]
    ys = [point["py"] for point in points]
    return clip_bbox([min(xs), min(ys), max(xs) + 1, max(ys) + 1], width, height)


def _bbox_center(bbox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (float((x1 + x2) / 2), float((y1 + y2) / 2))


def _assign_track_id(bbox, tracks: dict[int, dict[str, Any]], frame_id: int) -> int:
    center = _bbox_center(bbox)
    x1, y1, x2, y2 = bbox
    scale = max(float(x2 - x1), float(y2 - y1), 1.0)
    best_track = None
    best_distance = float("inf")
    for track_id, track in tracks.items():
        previous = track["center"]
        distance = float(np.hypot(center[0] - previous[0], center[1] - previous[1]))
        if distance < best_distance and distance <= max(scale * 1.5, 48.0):
            best_track = track_id
            best_distance = distance
    if best_track is None:
        best_track = max(tracks.keys(), default=0) + 1
    tracks[best_track] = {"center": center, "bbox": bbox, "last_seen": frame_id}
    return best_track


def _segment_person_mask(mp_module, segmenter, frame) -> np.ndarray | None:
    if segmenter is None:
        return None
    result = segmenter.segment(_mp_image(mp_module, frame))
    if result.category_mask is None:
        return None
    mask = result.category_mask.numpy_view()
    if mask.dtype.kind in {"f", "c"}:
        return (mask > 0.5).astype(np.uint8)
    return (mask > 0).astype(np.uint8)


def _component_bbox_for_face(person_mask: np.ndarray | None, face_bbox):
    if person_mask is None or person_mask.sum() == 0:
        return None
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(person_mask.astype(np.uint8), connectivity=8)
    cx, cy = _bbox_center(face_bbox)
    best = None
    best_area = 0
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        contains_center = x <= cx <= x + w and y <= cy <= y + h
        if contains_center and area > best_area:
            best = [int(x), int(y), int(x + w), int(y + h)]
            best_area = int(area)
    if best is not None:
        return clip_bbox(best, person_mask.shape[1], person_mask.shape[0], min_size=4)
    return bbox_from_mask(person_mask, min_size=4)


def _fallback_body_bbox(face_bbox, width: int, height: int):
    x1, y1, x2, y2 = face_bbox
    bw = x2 - x1
    bh = y2 - y1
    return clip_bbox(
        [
            x1 - 0.75 * bw,
            y1 - 0.20 * bh,
            x2 + 0.75 * bw,
            y2 + 3.25 * bh,
        ],
        width,
        height,
        min_size=4,
    )


def _region_entry(
    region_type: str,
    track_id: int | str,
    bbox,
    source: str,
    polygon=None,
    landmarks=None,
    confidence: float | None = None,
) -> dict[str, Any]:
    suffix = str(track_id).replace("face_", "")
    if region_type == BACKGROUND_REGION_NAME:
        region_id = BACKGROUND_REGION_NAME
        region_label = "fundo"
        track_value = "global"
    else:
        region_id = f"{region_type}_{suffix}"
        region_label = f"{region_type.replace('_', ' ')} {suffix}"
        track_value = f"face_{suffix}"
    return {
        "region_id": region_id,
        "region": region_id,
        "region_label": region_label,
        "region_type": region_type,
        "track_id": track_value,
        "bbox": [int(v) for v in bbox],
        "polygon": polygon or [],
        "landmarks": landmarks or [],
        "source": source,
        "confidence": confidence,
    }


def create_csrt_tracker():
    tracker_factory = getattr(cv2, "TrackerCSRT_create", None)
    if callable(tracker_factory):
        return tracker_factory()

    legacy = getattr(cv2, "legacy", None)
    if legacy is not None:
        legacy_factory = getattr(legacy, "TrackerCSRT_create", None)
        if callable(legacy_factory):
            return legacy_factory()

    return None


def fallback_center_bbox(frame):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    size = max(2, min(h, w) // 4)
    return clip_bbox([cx - size, cy - size, cx + size, cy + size], w, h)


def extract_face_metadata(
    video_path: str | Path,
    output_path: str | Path | None = None,
    max_frames: int | None = None,
    detect_every: int = 1,
    allow_fallback: bool = True,
    save_silver: bool = True,
    face_model_path: str | Path | None = None,
    segmenter_model_path: str | Path | None = None,
    max_faces: int = 10,
) -> list[dict[str, Any]]:
    """Extract MediaPipe regions used by all feature groups.

    The output keeps one JSON item per sampled frame. Each item contains a
    `regions` list with full face, eyes, mouth, body and background annotations.
    Feature extraction later expands this list into one row per region.
    """

    video_path = Path(video_path)
    video_id = video_path.stem
    output_path = Path(output_path) if output_path else metadata_path_for_video(video_path, METADATA_DIR)
    processed_at = datetime.now(timezone.utc).isoformat()

    face_detector = None
    mp_face = None
    segmenter = None
    mp_segmenter = None
    try:
        mp_face, face_detector = _create_face_landmarker(face_model_path, max_faces=max_faces)
        mp_segmenter, segmenter = _create_image_segmenter(segmenter_model_path)
    except (RuntimeError, FileNotFoundError) as exc:
        if not allow_fallback:
            raise
        logger.warning("MediaPipe initialization failed: %s. Falling back to center regions.", exc)

    metadata: list[dict[str, Any]] = []
    tracks: dict[int, dict[str, Any]] = {}

    for sample_idx, (frame_id, frame, _frame_count) in enumerate(iter_sampled_frames(video_path, max_frames=max_frames)):
        h, w = frame.shape[:2]
        frame_regions: list[dict[str, Any]] = []
        person_mask = _segment_person_mask(mp_segmenter, segmenter, frame) if mp_segmenter is not None else None
        face_landmarks = []

        use_detection = sample_idx % max(1, detect_every) == 0
        if face_detector is not None and mp_face is not None and use_detection:
            result = face_detector.detect(_mp_image(mp_face, frame))
            face_landmarks = list(result.face_landmarks or [])

        for face_landmark in face_landmarks:
            points = _landmark_rows(face_landmark, w, h)
            face_bbox = _bbox_from_points(points, w, h)
            if face_bbox is None:
                continue
            track_id = _assign_track_id(face_bbox, tracks, int(frame_id))

            face_polygon = _polygon_from_points(points)
            eyes_points = _points_for_indices(points, LEFT_EYE_INDICES + RIGHT_EYE_INDICES)
            mouth_points = _points_for_indices(points, OUTER_MOUTH_INDICES + INNER_MOUTH_INDICES)
            eyes_bbox = _bbox_from_points(eyes_points, w, h)
            mouth_bbox = _bbox_from_points(mouth_points, w, h)

            frame_regions.append(
                _region_entry(
                    FACE_REGION_NAME,
                    track_id,
                    face_bbox,
                    "mediapipe_face_landmarker",
                    polygon=face_polygon,
                    landmarks=points,
                    confidence=1.0,
                )
            )
            if eyes_bbox is not None:
                frame_regions.append(
                    _region_entry(
                        EYES_REGION_NAME,
                        track_id,
                        eyes_bbox,
                        "mediapipe_face_landmarker",
                        polygon=_polygon_from_points(eyes_points),
                        landmarks=eyes_points,
                        confidence=1.0,
                    )
                )
            if mouth_bbox is not None:
                frame_regions.append(
                    _region_entry(
                        MOUTH_REGION_NAME,
                        track_id,
                        mouth_bbox,
                        "mediapipe_face_landmarker",
                        polygon=_polygon_from_points(mouth_points),
                        landmarks=mouth_points,
                        confidence=1.0,
                    )
                )

            body_bbox = _component_bbox_for_face(person_mask, face_bbox)
            body_source = "mediapipe_image_segmenter"
            if body_bbox is None:
                body_bbox = _fallback_body_bbox(face_bbox, w, h)
                body_source = "fallback_body_geometry"
            if body_bbox is not None:
                frame_regions.append(
                    _region_entry(
                        BODY_REGION_NAME,
                        track_id,
                        body_bbox,
                        body_source,
                        confidence=1.0 if person_mask is not None else None,
                    )
                )

        if not frame_regions and allow_fallback:
            fallback_bbox = fallback_center_bbox(frame)
            if fallback_bbox is not None:
                track_id = _assign_track_id(fallback_bbox, tracks, int(frame_id))
                frame_regions.append(
                    _region_entry(
                        FACE_REGION_NAME,
                        track_id,
                        fallback_bbox,
                        "fallback_center",
                        confidence=None,
                    )
                )
                body_bbox = _fallback_body_bbox(fallback_bbox, w, h)
                if body_bbox is not None:
                    frame_regions.append(
                        _region_entry(
                            BODY_REGION_NAME,
                            track_id,
                            body_bbox,
                            "fallback_body_geometry",
                            confidence=None,
                        )
                    )

        if not frame_regions:
            continue

        frame_regions.append(
            _region_entry(
                BACKGROUND_REGION_NAME,
                "global",
                [0, 0, w, h],
                "mediapipe_image_segmenter" if person_mask is not None else "computed_background",
                confidence=1.0 if person_mask is not None else None,
            )
        )
        primary = next((region for region in frame_regions if region["region_type"] == FACE_REGION_NAME), frame_regions[0])
        primary_bbox = primary["bbox"]
        legacy_regions = create_face_regions(frame, primary_bbox)
        expanded = legacy_regions["bbox_expanded"] if legacy_regions is not None else primary_bbox

        metadata.append(
            {
                "video_id": video_id,
                "frame_id": int(frame_id),
                "bbox": [int(v) for v in primary_bbox],
                "bbox_expanded": [int(v) for v in expanded],
                "source": primary["source"],
                "detector_score": primary.get("confidence"),
                "regions": frame_regions,
                "region_count": int(len(frame_regions)),
                "face_count": int(sum(1 for region in frame_regions if region["region_type"] == FACE_REGION_NAME)),
                "frame_width": int(w),
                "frame_height": int(h),
                "processed_at": processed_at,
                "pipeline_version": PIPELINE_VERSION,
                "detector": "mediapipe",
            }
        )

    for model in (face_detector, segmenter):
        close = getattr(model, "close", None)
        if callable(close):
            close()

    save_metadata(metadata, output_path)
    if save_silver:
        write_silver_face_metadata(metadata, silver_face_metadata_path(video_path))
    logger.info("Saved %s metadata rows to %s", len(metadata), output_path)
    return metadata


def metadata_to_frame_contract_rows(metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in metadata:
        frame_regions = item.get("regions") or [
            {
                "region": f"{FACE_REGION_NAME}_1",
                "region_id": f"{FACE_REGION_NAME}_1",
                "region_label": "rosto completo 1",
                "region_type": FACE_REGION_NAME,
                "track_id": "face_1",
                "bbox": item.get("bbox", [None, None, None, None]),
                "source": item.get("source", ""),
                "confidence": item.get("detector_score"),
            }
        ]
        bbox_expanded = item.get("bbox_expanded", [None, None, None, None])
        for region in frame_regions:
            bbox = region.get("bbox", [None, None, None, None])
            rows.append(
                {
                    "video_id": item.get("video_id", ""),
                    "frame_id": item.get("frame_id"),
                    "region": region.get("region", region.get("region_id", "")),
                    "region_id": region.get("region_id", region.get("region", "")),
                    "region_label": region.get("region_label", ""),
                    "region_type": region.get("region_type", ""),
                    "track_id": region.get("track_id", ""),
                    "bbox_x1": bbox[0],
                    "bbox_y1": bbox[1],
                    "bbox_x2": bbox[2],
                    "bbox_y2": bbox[3],
                    "bbox_expanded_x1": bbox_expanded[0],
                    "bbox_expanded_y1": bbox_expanded[1],
                    "bbox_expanded_x2": bbox_expanded[2],
                    "bbox_expanded_y2": bbox_expanded[3],
                    "source": region.get("source", item.get("source", "")),
                    "detector_score": region.get("confidence", item.get("detector_score")),
                    "frame_width": item.get("frame_width"),
                    "frame_height": item.get("frame_height"),
                    "processed_at": item.get("processed_at", ""),
                    "pipeline_version": item.get("pipeline_version", PIPELINE_VERSION),
                }
            )
    return rows


def write_silver_face_metadata(metadata: list[dict[str, Any]], output_path: str | Path) -> Path:
    import pandas as pd

    return write_dataframe(pd.DataFrame(metadata_to_frame_contract_rows(metadata)), output_path, index=False)


def metadata_is_current(metadata_path: str | Path) -> bool:
    path = Path(metadata_path)
    if not path.exists():
        return False
    try:
        from src.shared.video import load_metadata

        metadata = load_metadata(path)
    except (OSError, ValueError):
        return False
    if not metadata:
        return False
    for item in metadata:
        if str(item.get("pipeline_version", "")) != str(PIPELINE_VERSION):
            return False
        if not item.get("regions"):
            return False
    return True


def video_path_from_manifest_row(row: Any, videos_dir: str | Path) -> Path | None:
    storage_path = str(row.get("storage_path", "") or "").strip()
    if storage_path:
        return Path(storage_path)

    filename = str(row.get("filename", "") or "").strip()
    if filename:
        return Path(videos_dir) / filename

    return None


def is_processable_manifest_row(row: Any) -> bool:
    status = str(row.get("status", "") or "").strip()
    return status in {"", "downloaded", "skipped"}


def process_catalog(
    catalog_path: str | Path,
    videos_dir: str | Path,
    metadata_dir: str | Path = METADATA_DIR,
    max_frames: int | None = None,
    detect_every: int = 1,
    overwrite: bool = False,
    face_model_path: str | Path | None = None,
    segmenter_model_path: str | Path | None = None,
    max_faces: int = 10,
) -> list[Path]:
    import pandas as pd

    catalog = pd.read_csv(catalog_path)
    outputs: list[Path] = []

    for _, row in catalog.iterrows():
        if not is_processable_manifest_row(row):
            continue

        video_path = video_path_from_manifest_row(row, videos_dir)
        if video_path is None:
            logger.warning("Skipping manifest row without storage_path or filename.")
            continue

        output_path = metadata_path_for_video(video_path, metadata_dir)
        if output_path.exists() and not overwrite and metadata_is_current(output_path):
            outputs.append(output_path)
            continue
        if not video_path.exists():
            logger.warning("Video not found: %s", video_path)
            continue
        extract_face_metadata(
            video_path,
            output_path,
            max_frames=max_frames,
            detect_every=detect_every,
            face_model_path=face_model_path,
            segmenter_model_path=segmenter_model_path,
            max_faces=max_faces,
        )
        outputs.append(output_path)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract face metadata for one video or a catalog.")
    parser.add_argument("--video", type=Path, help="Path to a single video.")
    parser.add_argument("--manifest", "--catalog", dest="manifest", type=Path, help="Bronze manifest with storage_path/filename columns.")
    parser.add_argument("--videos-dir", type=Path, default=BRONZE_VIDEOS_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=METADATA_DIR)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--detect-every", type=int, default=1)
    parser.add_argument("--face-model", type=Path, default=None)
    parser.add_argument("--segmenter-model", type=Path, default=None)
    parser.add_argument("--max-faces", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.video:
        output = metadata_path_for_video(args.video, args.metadata_dir)
        if output.exists() and not args.overwrite:
            logger.info("Metadata already exists: %s", output)
            return
        extract_face_metadata(
            args.video,
            output,
            max_frames=args.max_frames,
            detect_every=args.detect_every,
            face_model_path=args.face_model,
            segmenter_model_path=args.segmenter_model,
            max_faces=args.max_faces,
        )
        return

    if args.manifest:
        process_catalog(
            args.manifest,
            args.videos_dir,
            metadata_dir=args.metadata_dir,
            max_frames=args.max_frames,
            detect_every=args.detect_every,
            overwrite=args.overwrite,
            face_model_path=args.face_model,
            segmenter_model_path=args.segmenter_model,
            max_faces=args.max_faces,
        )
        return

    parser.error("Use --video or --manifest.")


if __name__ == "__main__":
    main()
