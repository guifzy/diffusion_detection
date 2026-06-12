from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.shared.contracts import summarize_validation_results, validate_dataframe_contract, validate_table_contract
from src.shared.core.io_utils import read_dataframe
from src.shared.core.paths import (
    BRONZE_MANIFEST_PATH,
    GOLD_DIR,
    METADATA_DIR,
    SILVER_DIR,
    gold_training_dataset_path,
    silver_video_features_path,
)


def _read_optional_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        return read_dataframe(path)
    except (FileNotFoundError, ValueError, ImportError):
        return pd.DataFrame()


def _read_many_tables(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frames.append(read_dataframe(path))
        except (FileNotFoundError, ValueError, ImportError):
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def bronze_quality(manifest_path: str | Path = BRONZE_MANIFEST_PATH) -> dict:
    manifest = _read_optional_table(manifest_path)
    if manifest.empty:
        return {
            "input_rows": 0,
            "downloaded": 0,
            "failed": 0,
            "skipped": 0,
            "unique_videos": 0,
            "manifest_path": str(manifest_path),
        }

    status_counts = manifest["status"].fillna("").value_counts().to_dict() if "status" in manifest else {}
    return {
        "input_rows": int(len(manifest)),
        "downloaded": int(status_counts.get("downloaded", 0)),
        "failed": int(status_counts.get("failed", 0)),
        "skipped": int(status_counts.get("skipped", 0)),
        "unique_videos": int(manifest["video_id"].dropna().astype(str).replace("", pd.NA).dropna().nunique())
        if "video_id" in manifest
        else 0,
        "manifest_path": str(manifest_path),
    }


def silver_metadata_quality(metadata_dir: str | Path = METADATA_DIR, silver_dir: str | Path = SILVER_DIR) -> dict:
    face_metadata_dir = Path(silver_dir) / "face_metadata"
    tables = _read_many_tables(sorted(face_metadata_dir.glob("*.parquet")) + sorted(face_metadata_dir.glob("*.csv")))
    json_count = len(list(Path(metadata_dir).glob("*_meta.json")))

    if tables.empty:
        return {
            "videos_processed": 0,
            "json_files": int(json_count),
            "frames_processados": 0,
            "frames_com_face": 0,
            "avg_face_coverage": 0.0,
            "coverage_ratio": 0.0,
            "percentual_detector": 0.0,
            "percentual_tracker": 0.0,
            "percentual_last_bbox": 0.0,
            "percentual_fallback_center": 0.0,
            "fallback_center_ratio": 0.0,
            "source_distribution": {},
        }

    total_frames = int(len(tables))
    videos_processed = int(tables["video_id"].nunique()) if "video_id" in tables else 0
    source_counts = tables["source"].fillna("").value_counts(normalize=True).to_dict() if "source" in tables else {}
    return {
        "videos_processed": videos_processed,
        "json_files": int(json_count),
        "frames_processados": total_frames,
        "frames_com_face": total_frames,
        "avg_face_coverage": 1.0 if total_frames else 0.0,
        "coverage_ratio": 1.0 if total_frames else 0.0,
        "percentual_detector": float(source_counts.get("detector", 0.0)),
        "percentual_tracker": float(source_counts.get("tracker", 0.0)),
        "percentual_last_bbox": float(source_counts.get("last_bbox", 0.0)),
        "percentual_fallback_center": float(source_counts.get("fallback_center", 0.0)),
        "fallback_center_ratio": float(source_counts.get("fallback_center", 0.0)),
        "source_distribution": {str(key): float(value) for key, value in source_counts.items()},
    }


def silver_features_quality(silver_dir: str | Path = SILVER_DIR) -> dict:
    silver_dir = Path(silver_dir)
    frame_tables = _read_many_tables(
        sorted((silver_dir / "frame_features").glob("*.parquet")) + sorted((silver_dir / "frame_features").glob("*.csv"))
    )
    video_features = _read_optional_table(silver_video_features_path(silver_dir))

    avg_missing = (
        float(video_features["missing_feature_ratio"].mean())
        if not video_features.empty and "missing_feature_ratio" in video_features
        else 0.0
    )
    return {
        "frame_feature_files": len(list((silver_dir / "frame_features").glob("*.parquet")))
        + len(list((silver_dir / "frame_features").glob("*.csv"))),
        "frame_rows": int(len(frame_tables)),
        "videos_with_frame_features": int(frame_tables["video_id"].nunique()) if "video_id" in frame_tables else 0,
        "videos_processed": int(len(video_features)),
        "avg_missing_feature_ratio": avg_missing,
    }


def gold_quality(gold_dir: str | Path = GOLD_DIR) -> dict:
    gold = _read_optional_table(gold_training_dataset_path(gold_dir))
    if gold.empty:
        return {"rows": 0, "trainable_rows": 0, "real_count": 0, "fake_count": 0, "split_distribution": {}}

    labels = gold["target_label"].fillna("") if "target_label" in gold else pd.Series([], dtype=str)
    splits = gold["dataset_split"].fillna("").value_counts().to_dict() if "dataset_split" in gold else {}
    return {
        "rows": int(len(gold)),
        "trainable_rows": int(gold["is_trainable"].fillna(False).astype(bool).sum()) if "is_trainable" in gold else 0,
        "real_count": int((labels == "Real").sum()),
        "fake_count": int((labels == "Fake").sum()),
        "split_distribution": {str(key): int(value) for key, value in splits.items()},
    }


def validate_pipeline_assets(
    manifest_path: str | Path = BRONZE_MANIFEST_PATH,
    silver_dir: str | Path = SILVER_DIR,
    gold_dir: str | Path = GOLD_DIR,
) -> dict:
    results = []
    results.append(validate_table_contract(manifest_path, "bronze_manifest"))

    face_tables = sorted((Path(silver_dir) / "face_metadata").glob("*.parquet")) + sorted(
        (Path(silver_dir) / "face_metadata").glob("*.csv")
    )
    if face_tables:
        face_metadata = _read_many_tables(face_tables)
        results.append(validate_dataframe_contract(face_metadata, "frame_metadata", Path(silver_dir) / "face_metadata"))

    frame_tables = sorted((Path(silver_dir) / "frame_features").glob("*.parquet")) + sorted(
        (Path(silver_dir) / "frame_features").glob("*.csv")
    )
    if frame_tables:
        frame_features = _read_many_tables(frame_tables)
        results.append(validate_dataframe_contract(frame_features, "frame_features", Path(silver_dir) / "frame_features"))

    results.append(validate_table_contract(silver_video_features_path(silver_dir), "video_features"))
    results.append(validate_table_contract(gold_training_dataset_path(gold_dir), "gold_training_dataset"))
    return summarize_validation_results(results)


def build_quality_report(
    manifest_path: str | Path = BRONZE_MANIFEST_PATH,
    metadata_dir: str | Path = METADATA_DIR,
    silver_dir: str | Path = SILVER_DIR,
    gold_dir: str | Path = GOLD_DIR,
) -> dict:
    return {
        "bronze": bronze_quality(manifest_path),
        "silver_metadata": silver_metadata_quality(metadata_dir, silver_dir),
        "silver_features": silver_features_quality(silver_dir),
        "gold": gold_quality(gold_dir),
        "contracts": validate_pipeline_assets(manifest_path, silver_dir, gold_dir),
    }
