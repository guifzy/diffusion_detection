from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import cv2

from src.shared.core.io_utils import write_dataframe
from src.shared.core.paths import BRONZE_VIDEOS_DIR, METADATA_DIR, metadata_path_for_video, silver_face_metadata_path
from src.shared.core.version import PIPELINE_VERSION
from src.shared.video import (
    clip_bbox,
    create_face_regions,
    iter_sampled_frames,
    save_metadata,
    scale_bbox,
    standardize_frame,
)

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

logger = logging.getLogger(__name__)


def _load_retinaface():
    from retinaface import RetinaFace

    return RetinaFace


def detect_face(frame, detector: Any | None = None):
    detector = detector or _load_retinaface()
    detections = detector.detect_faces(frame)
    if not detections:
        return None

    faces = list(detections.values())
    faces = [face for face in faces if isinstance(face, dict) and "facial_area" in face]
    if not faces:
        return None

    face = max(faces, key=lambda item: float(item.get("score", 0.0)))
    return {
        "bbox": face["facial_area"],
        "landmarks": face.get("landmarks", {}),
        "score": float(face.get("score", 0.0)),
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
    standard_max_size: int = 640,
    allow_fallback: bool = True,
    save_silver: bool = True,
) -> list[dict[str, Any]]:
    """Extract face boxes used by all feature groups.

    This function is intentionally production-oriented: it streams sampled frames,
    records original frame ids, and returns serializable metadata that can be used
    by both batch training and online inference.
    """

    video_path = Path(video_path)
    video_id = video_path.stem
    output_path = Path(output_path) if output_path else metadata_path_for_video(video_path, METADATA_DIR)
    detector = _load_retinaface()
    processed_at = datetime.now(timezone.utc).isoformat()

    metadata: list[dict[str, Any]] = []
    tracker: Any = None
    last_bbox = None

    for sample_idx, (frame_id, frame, _frame_count) in enumerate(iter_sampled_frames(video_path, max_frames=max_frames)):
        h, w = frame.shape[:2]
        use_detection = (sample_idx % max(1, detect_every) == 0) or tracker is None
        bbox_original = None
        detector_score = None
        source = "detector"

        if use_detection:
            frame_std, scale = standardize_frame(frame, max_size=standard_max_size)
            det = detect_face(frame_std, detector=detector)
            if det is not None:
                bbox_original = scale_bbox(det["bbox"], scale)
                bbox_original = clip_bbox(bbox_original, w, h) if bbox_original is not None else None
                detector_score = det.get("score")

                if bbox_original is not None:
                    x1, y1, x2, y2 = bbox_original
                    tracker = create_csrt_tracker()
                    if tracker is not None:
                        try:
                            tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                        except cv2.error:
                            tracker = None
                    last_bbox = bbox_original

        else:
            source = "tracker"
            if tracker is not None:
                success, tracked_box = tracker.update(frame)
                if success:
                    x, y, bw, bh = map(int, tracked_box)
                    bbox_original = clip_bbox([x, y, x + bw, y + bh], w, h)
                    if bbox_original is not None:
                        last_bbox = bbox_original

        if bbox_original is None and last_bbox is not None:
            bbox_original = last_bbox
            source = "last_bbox"

        if bbox_original is None and allow_fallback:
            bbox_original = fallback_center_bbox(frame)
            source = "fallback_center"

        if bbox_original is None:
            continue

        regions = create_face_regions(frame, bbox_original)
        if regions is None:
            continue

        metadata.append(
            {
                "video_id": video_id,
                "frame_id": int(frame_id),
                "bbox": [int(v) for v in bbox_original],
                "bbox_expanded": [int(v) for v in regions["bbox_expanded"]],
                "source": source,
                "detector_score": detector_score,
                "frame_width": int(w),
                "frame_height": int(h),
                "processed_at": processed_at,
                "pipeline_version": PIPELINE_VERSION,
            }
        )

    save_metadata(metadata, output_path)
    if save_silver:
        write_silver_face_metadata(metadata, silver_face_metadata_path(video_path))
    logger.info("Saved %s metadata rows to %s", len(metadata), output_path)
    return metadata


def metadata_to_frame_contract_rows(metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in metadata:
        bbox = item.get("bbox", [None, None, None, None])
        bbox_expanded = item.get("bbox_expanded", [None, None, None, None])
        rows.append(
            {
                "video_id": item.get("video_id", ""),
                "frame_id": item.get("frame_id"),
                "bbox_x1": bbox[0],
                "bbox_y1": bbox[1],
                "bbox_x2": bbox[2],
                "bbox_y2": bbox[3],
                "bbox_expanded_x1": bbox_expanded[0],
                "bbox_expanded_y1": bbox_expanded[1],
                "bbox_expanded_x2": bbox_expanded[2],
                "bbox_expanded_y2": bbox_expanded[3],
                "source": item.get("source", ""),
                "detector_score": item.get("detector_score"),
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


def process_catalog(
    catalog_path: str | Path,
    videos_dir: str | Path,
    metadata_dir: str | Path = METADATA_DIR,
    max_frames: int | None = None,
    detect_every: int = 1,
    overwrite: bool = False,
) -> list[Path]:
    import pandas as pd

    catalog = pd.read_csv(catalog_path)
    outputs: list[Path] = []

    for _, row in catalog.iterrows():
        video_path = Path(videos_dir) / row["Filename"]
        output_path = metadata_path_for_video(video_path, metadata_dir)
        if output_path.exists() and not overwrite:
            outputs.append(output_path)
            continue
        if not video_path.exists():
            logger.warning("Video not found: %s", video_path)
            continue
        extract_face_metadata(video_path, output_path, max_frames=max_frames, detect_every=detect_every)
        outputs.append(output_path)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract face metadata for one video or a catalog.")
    parser.add_argument("--video", type=Path, help="Path to a single video.")
    parser.add_argument("--catalog", type=Path, help="CSV catalog with a Filename column.")
    parser.add_argument("--videos-dir", type=Path, default=BRONZE_VIDEOS_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=METADATA_DIR)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--detect-every", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.video:
        output = metadata_path_for_video(args.video, args.metadata_dir)
        if output.exists() and not args.overwrite:
            logger.info("Metadata already exists: %s", output)
            return
        extract_face_metadata(args.video, output, max_frames=args.max_frames, detect_every=args.detect_every)
        return

    if args.catalog:
        process_catalog(
            args.catalog,
            args.videos_dir,
            metadata_dir=args.metadata_dir,
            max_frames=args.max_frames,
            detect_every=args.detect_every,
            overwrite=args.overwrite,
        )
        return

    parser.error("Use --video or --catalog.")


if __name__ == "__main__":
    main()
