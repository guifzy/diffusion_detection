from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

BRONZE_DIR = DATA_DIR / "bronze"
BRONZE_VIDEOS_DIR = BRONZE_DIR / "videos"
BRONZE_MANIFESTS_DIR = BRONZE_DIR / "manifests"
SILVER_DIR = DATA_DIR / "silver"
METADATA_DIR = SILVER_DIR / "face_metadata_json"
SILVER_FACE_METADATA_DIR = SILVER_DIR / "face_metadata"
SILVER_FRAME_FEATURES_DIR = SILVER_DIR / "frame_features"
SILVER_VIDEO_FEATURES_DIR = SILVER_DIR / "video_features"
GOLD_DIR = DATA_DIR / "gold"
REPORTS_DIR = DATA_DIR / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

VIDEO_CATALOG_PATH = BRONZE_MANIFESTS_DIR / "video-metadata-publish-with-links.csv"
BRONZE_MANIFEST_PATH = BRONZE_MANIFESTS_DIR / "bronze_manifest.csv"


def ensure_data_dirs() -> None:
    for path in [
        BRONZE_VIDEOS_DIR,
        BRONZE_MANIFESTS_DIR,
        METADATA_DIR,
        SILVER_FACE_METADATA_DIR,
        SILVER_FRAME_FEATURES_DIR,
        SILVER_VIDEO_FEATURES_DIR,
        GOLD_DIR,
        REPORTS_DIR,
        MODELS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def video_id_from_path(video_path: str | Path) -> str:
    return Path(video_path).stem


def metadata_path_for_video(video_path: str | Path, metadata_dir: str | Path = METADATA_DIR) -> Path:
    return Path(metadata_dir) / f"{video_id_from_path(video_path)}_meta.json"


def silver_frame_features_path(video_path: str | Path, silver_dir: str | Path = SILVER_DIR) -> Path:
    return Path(silver_dir) / "frame_features" / f"{video_id_from_path(video_path)}.parquet"


def silver_face_metadata_path(video_path: str | Path, silver_dir: str | Path = SILVER_DIR) -> Path:
    return Path(silver_dir) / "face_metadata" / f"{video_id_from_path(video_path)}.parquet"


def silver_video_features_path(silver_dir: str | Path = SILVER_DIR) -> Path:
    return Path(silver_dir) / "video_features" / "video_features.parquet"


def gold_training_dataset_path(gold_dir: str | Path = GOLD_DIR) -> Path:
    return Path(gold_dir) / "gold_training_dataset.parquet"


def pipeline_report_path(reports_dir: str | Path = REPORTS_DIR, run_id: str | None = None) -> Path:
    from datetime import datetime, timezone

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(reports_dir) / f"pipeline_run_{run_id}.json"
