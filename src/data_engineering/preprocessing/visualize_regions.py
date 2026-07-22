from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.shared.core.paths import METADATA_DIR, metadata_path_for_video
from src.shared.video import create_face_regions, load_metadata, metadata_for_frame


COLORS_BGR = {
    "face": (50, 205, 50),
    "border": (0, 215, 255),
    "background": (210, 90, 70),
}


def _read_frame(video_path: Path, frame_id: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = max(0, min(int(frame_id), max(0, frame_count - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_id} from {video_path}")
    return frame, frame_count, frame_id


def _blend_mask(overlay: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    color_layer = np.zeros_like(overlay)
    color_layer[:, :] = color
    selected = mask.astype(bool)
    overlay[selected] = cv2.addWeighted(overlay, 1.0 - alpha, color_layer, alpha, 0)[selected]


def render_region_overlay(
    video_path: str | Path,
    metadata_path: str | Path | None = None,
    output_path: str | Path | None = None,
    frame_id: int = 0,
    alpha: float = 0.32,
) -> Path:
    video_path = Path(video_path)
    metadata_path = Path(metadata_path) if metadata_path else metadata_path_for_video(video_path, METADATA_DIR)
    output_path = Path(output_path) if output_path else Path("docs/img") / f"{video_path.stem}_regions_frame_{frame_id}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame, frame_count, frame_id = _read_frame(video_path, frame_id)
    metadata = load_metadata(metadata_path)
    meta, metadata_idx = metadata_for_frame(frame_id, frame_count, metadata)
    if meta is None:
        raise RuntimeError(f"No compatible metadata found for frame {frame_id}: {metadata_path}")

    regions = create_face_regions(frame, meta["bbox"])
    if regions is None:
        raise RuntimeError(f"Could not create regions for frame {frame_id} using bbox={meta.get('bbox')}")

    overlay = frame.copy()
    _blend_mask(overlay, regions["background"], COLORS_BGR["background"], alpha * 0.45)
    _blend_mask(overlay, regions["border"], COLORS_BGR["border"], alpha)
    _blend_mask(overlay, regions["face"], COLORS_BGR["face"], alpha)

    x1, y1, x2, y2 = regions["bbox"]
    ex1, ey1, ex2, ey2 = regions["bbox_expanded"]
    cv2.rectangle(overlay, (ex1, ey1), (ex2, ey2), COLORS_BGR["border"], 2)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COLORS_BGR["face"], 2)

    labels = [
        ("face", COLORS_BGR["face"]),
        ("contorno", COLORS_BGR["border"]),
        ("fundo", COLORS_BGR["background"]),
        (f"frame={frame_id} metadata_idx={metadata_idx}", (245, 245, 245)),
    ]
    y = 28
    for label, color in labels:
        cv2.putText(overlay, label, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(overlay, label, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        y += 30

    if not cv2.imwrite(str(output_path), overlay):
        raise RuntimeError(f"Could not write output image: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a frame with face, border and background region overlays.")
    parser.add_argument("--video", required=True, type=Path, help="Path to the input video.")
    parser.add_argument("--metadata", type=Path, help="Path to the *_meta.json file. Defaults to the standard Silver path.")
    parser.add_argument("--output", type=Path, help="Output PNG path. Defaults to docs/img/<video>_regions_frame_<frame>.png.")
    parser.add_argument("--frame", type=int, default=0, help="Frame id to render.")
    parser.add_argument("--alpha", type=float, default=0.32, help="Overlay opacity.")
    args = parser.parse_args()

    output = render_region_overlay(
        video_path=args.video,
        metadata_path=args.metadata,
        output_path=args.output,
        frame_id=args.frame,
        alpha=args.alpha,
    )
    print(output)


if __name__ == "__main__":
    main()
