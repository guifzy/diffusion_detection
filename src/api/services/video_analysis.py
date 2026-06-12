from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.shared.core.io_utils import sanitize_for_json, write_dataframe
from src.shared.core.paths import METADATA_DIR, SILVER_DIR, metadata_path_for_video, silver_frame_features_path

logger = logging.getLogger(__name__)


def process_video(
    video_path: str | Path,
    metadata_path: str | Path | None = None,
    groups: str = "abcde",
    max_frames: int | None = None,
    extract_metadata: bool = True,
    overwrite_metadata: bool = False,
    save_frame_features: bool = True,
) -> tuple[Any, dict]:
    from src.data_engineering.preprocessing import extract_face_metadata
    from src.shared.features.extractor import build_video_features

    video_path = Path(video_path)
    metadata_path = Path(metadata_path) if metadata_path else metadata_path_for_video(video_path, METADATA_DIR)

    if extract_metadata and (overwrite_metadata or not metadata_path.exists()):
        logger.info("Extracting metadata for %s", video_path)
        extract_face_metadata(video_path, metadata_path, max_frames=max_frames)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found for {video_path}: {metadata_path}")

    frame_features, video_features = build_video_features(
        video_path,
        metadata_path,
        groups=groups,
        max_frames=max_frames,
    )

    if save_frame_features:
        output_path = silver_frame_features_path(video_path, SILVER_DIR)
        saved_path = write_dataframe(frame_features, output_path, index=False)
        logger.info("Saved frame features to %s", saved_path)

    return frame_features, video_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Production-oriented feature pipeline for one video.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--groups", default="abcde")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-metadata-extraction", action="store_true")
    parser.add_argument("--overwrite-metadata", action="store_true")
    parser.add_argument("--no-save-frame-features", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print video-level features as JSON.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _, video_features = process_video(
        args.video,
        metadata_path=args.metadata,
        groups=args.groups,
        max_frames=args.max_frames,
        extract_metadata=not args.no_metadata_extraction,
        overwrite_metadata=args.overwrite_metadata,
        save_frame_features=not args.no_save_frame_features,
    )

    if args.json:
        print(json.dumps(sanitize_for_json(video_features), ensure_ascii=False, default=float, allow_nan=False))
    else:
        import pandas as pd

        print(pd.Series(video_features).to_string())


if __name__ == "__main__":
    main()
