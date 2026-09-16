from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

EXTERNAL_DATA_DIR = DATA_DIR / "external"
DF26_DATASET_DIR = EXTERNAL_DATA_DIR / "df26"
BRONZE_DIR = DATA_DIR / "bronze"
BRONZE_VIDEOS_DIR = BRONZE_DIR / "videos"
BRONZE_MANIFESTS_DIR = BRONZE_DIR / "manifests"
SILVER_DIR = DATA_DIR / "silver"
METADATA_DIR = SILVER_DIR / "face_metadata_json"
SILVER_FACE_METADATA_DIR = SILVER_DIR / "face_metadata"
SILVER_FRAME_FEATURES_DIR = SILVER_DIR / "frame_features"
SILVER_VIDEO_FEATURES_DIR = SILVER_DIR / "video_features"
SILVER_TEMPORAL_FEATURES_DIR = SILVER_DIR / "temporal_features"
GOLD_DIR = DATA_DIR / "gold"
REPORTS_DIR = DATA_DIR / "reports"
REPORTS_LOG_DIR = REPORTS_DIR / "logs"
REPORTS_PLOTS_DIR = REPORTS_DIR / "plots"
MODELS_DIR = PROJECT_ROOT / "models"

VIDEO_CATALOG_PATH = BRONZE_MANIFESTS_DIR / "video-metadata-publish-with-links.csv"
BRONZE_MANIFEST_PATH = BRONZE_MANIFESTS_DIR / "bronze_manifest.csv"
DF26_MANIFEST_PATH = BRONZE_MANIFESTS_DIR / "bronze_manifest_df26.csv"
DF26_LOGOS_SPLITS_PATH = BRONZE_MANIFESTS_DIR / "df26_logos_splits.csv"


def ensure_data_dirs() -> None:
    for path in [
        EXTERNAL_DATA_DIR,
        DF26_DATASET_DIR,
        BRONZE_VIDEOS_DIR,
        BRONZE_MANIFESTS_DIR,
        METADATA_DIR,
        SILVER_FACE_METADATA_DIR,
        SILVER_FRAME_FEATURES_DIR,
        SILVER_VIDEO_FEATURES_DIR,
        SILVER_TEMPORAL_FEATURES_DIR,
        GOLD_DIR,
        REPORTS_DIR,
        REPORTS_LOG_DIR,
        REPORTS_PLOTS_DIR,
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


def silver_temporal_features_path(silver_dir: str | Path = SILVER_DIR) -> Path:
    return Path(silver_dir) / "temporal_features" / "temporal_features.parquet"


def gold_video_region_dataset_path(gold_dir: str | Path = GOLD_DIR) -> Path:
    return Path(gold_dir) / "gold_video_region_dataset.parquet"


def gold_training_dataset_path(gold_dir: str | Path = GOLD_DIR) -> Path:
    return Path(gold_dir) / "gold_training_dataset.parquet"


def pipeline_report_path(reports_dir: str | Path = REPORTS_DIR, run_id: str | None = None) -> Path:
    from datetime import datetime, timezone

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(reports_dir) / f"pipeline_run_{run_id}.json"


def pipeline_latest_report_path(reports_dir: str | Path = REPORTS_DIR) -> Path:
    return Path(reports_dir) / "pipeline_latest.json"


def pipeline_metrics_path(reports_dir: str | Path = REPORTS_DIR) -> Path:
    return Path(reports_dir) / "metrics.json"


def pipeline_log_path(reports_dir: str | Path = REPORTS_DIR, run_id: str | None = None) -> Path:
    from datetime import datetime, timezone

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d")
    return Path(reports_dir) / "logs" / f"pipeline_{run_id}.jsonl"


def pipeline_plot_path(name: str, reports_dir: str | Path = REPORTS_DIR) -> Path:
    return Path(reports_dir) / "plots" / name
