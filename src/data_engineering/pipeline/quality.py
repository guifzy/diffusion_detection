from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.shared.contracts import summarize_validation_results, validate_dataframe_contract, validate_table_contract
from src.shared.core.io_utils import read_dataframe, write_dataframe, write_json
from src.shared.core.paths import (
    BRONZE_MANIFEST_PATH,
    GOLD_DIR,
    METADATA_DIR,
    REPORTS_DIR,
    SILVER_DIR,
    gold_training_dataset_path,
    pipeline_metrics_path,
    pipeline_plot_path,
    silver_video_features_path,
)
from src.shared.core.version import PIPELINE_VERSION


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


def _current_version_only(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "pipeline_version" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["pipeline_version"].astype(str) == str(PIPELINE_VERSION)].copy()


def collect_pipeline_tables(
    manifest_path: str | Path = BRONZE_MANIFEST_PATH,
    silver_dir: str | Path = SILVER_DIR,
    gold_dir: str | Path = GOLD_DIR,
) -> dict[str, tuple[pd.DataFrame, str]]:
    silver_dir = Path(silver_dir)
    frame_metadata = _read_many_tables(
        sorted((silver_dir / "face_metadata").glob("*.parquet")) + sorted((silver_dir / "face_metadata").glob("*.csv"))
    )
    frame_features = _read_many_tables(
        sorted((silver_dir / "frame_features").glob("*.parquet")) + sorted((silver_dir / "frame_features").glob("*.csv"))
    )
    frame_metadata = _current_version_only(frame_metadata)
    frame_features = _current_version_only(frame_features)
    return {
        "bronze_manifest": (_read_optional_table(manifest_path), str(manifest_path)),
        "frame_metadata": (frame_metadata, str(silver_dir / "face_metadata")),
        "frame_features": (frame_features, str(silver_dir / "frame_features")),
        "video_features": (_read_optional_table(silver_video_features_path(silver_dir)), str(silver_video_features_path(silver_dir))),
        "gold_training_dataset": (
            _read_optional_table(gold_training_dataset_path(gold_dir)),
            str(gold_training_dataset_path(gold_dir)),
        ),
    }


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
    frame_count = int(tables[["video_id", "frame_id"]].drop_duplicates().shape[0]) if {"video_id", "frame_id"} <= set(tables.columns) else total_frames
    videos_processed = int(tables["video_id"].nunique()) if "video_id" in tables else 0
    source_counts = tables["source"].fillna("").value_counts(normalize=True).to_dict() if "source" in tables else {}
    return {
        "videos_processed": videos_processed,
        "json_files": int(json_count),
        "frames_processados": frame_count,
        "frames_com_face": int(tables[tables.get("region_type", "") == "rosto_completo"][["video_id", "frame_id"]].drop_duplicates().shape[0])
        if {"video_id", "frame_id", "region_type"} <= set(tables.columns)
        else frame_count,
        "region_rows": total_frames,
        "avg_face_coverage": 1.0 if total_frames else 0.0,
        "coverage_ratio": 1.0 if total_frames else 0.0,
        "percentual_detector": float(source_counts.get("detector", 0.0)),
        "percentual_mediapipe_face_landmarker": float(source_counts.get("mediapipe_face_landmarker", 0.0)),
        "percentual_mediapipe_image_segmenter": float(source_counts.get("mediapipe_image_segmenter", 0.0)),
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
    frame_tables = _current_version_only(frame_tables)
    video_features = _read_optional_table(silver_video_features_path(silver_dir))
    video_features = _current_version_only(video_features)

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
        "videos_processed": int(video_features["video_id"].nunique()) if "video_id" in video_features else 0,
        "video_region_rows": int(len(video_features)),
        "avg_missing_feature_ratio": avg_missing,
    }


def gold_quality(gold_dir: str | Path = GOLD_DIR) -> dict:
    gold = _read_optional_table(gold_training_dataset_path(gold_dir))
    if gold.empty:
        return {
            "rows": 0,
            "trainable_rows": 0,
            "real_count": 0,
            "fake_count": 0,
            "quality_flag_distribution": {},
            "split_distribution": {},
        }

    labels = gold["target_label"].fillna("") if "target_label" in gold else pd.Series([], dtype=str)
    splits = gold["dataset_split"].fillna("").value_counts().to_dict() if "dataset_split" in gold else {}
    quality_flags = gold["quality_flag"].fillna("").value_counts().to_dict() if "quality_flag" in gold else {}
    return {
        "rows": int(len(gold)),
        "trainable_rows": int(gold["is_trainable"].fillna(False).astype(bool).sum()) if "is_trainable" in gold else 0,
        "real_count": int((labels == "Real").sum()),
        "fake_count": int((labels == "Fake").sum()),
        "quality_flag_distribution": {str(key): int(value) for key, value in quality_flags.items()},
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
        face_metadata = _current_version_only(face_metadata)
        results.append(validate_dataframe_contract(face_metadata, "frame_metadata", Path(silver_dir) / "face_metadata"))

    frame_tables = sorted((Path(silver_dir) / "frame_features").glob("*.parquet")) + sorted(
        (Path(silver_dir) / "frame_features").glob("*.csv")
    )
    if frame_tables:
        frame_features = _read_many_tables(frame_tables)
        frame_features = _current_version_only(frame_features)
        results.append(validate_dataframe_contract(frame_features, "frame_features", Path(silver_dir) / "frame_features"))

    results.append(validate_table_contract(silver_video_features_path(silver_dir), "video_features"))
    results.append(validate_table_contract(gold_training_dataset_path(gold_dir), "gold_training_dataset"))
    return summarize_validation_results(results)


def gx_validation_report(
    manifest_path: str | Path = BRONZE_MANIFEST_PATH,
    silver_dir: str | Path = SILVER_DIR,
    gold_dir: str | Path = GOLD_DIR,
) -> dict:
    from src.data_engineering.pipeline.gx_validation import validate_tables_with_gx

    tables = collect_pipeline_tables(manifest_path=manifest_path, silver_dir=silver_dir, gold_dir=gold_dir)
    return validate_tables_with_gx(tables)


def blocking_errors_for_report(
    report: dict,
    fail_on_empty_gold: bool = True,
    fail_on_contract_error: bool = True,
    fail_on_gx_error: bool = False,
    max_missing_feature_ratio: float = 0.5,
    max_fallback_center_ratio: float = 0.35,
) -> list[str]:
    errors: list[str] = []
    contracts = report.get("contracts", {})
    gx = report.get("great_expectations", {})
    bronze = report.get("bronze", {})
    silver_metadata = report.get("silver_metadata", {})
    silver_features = report.get("silver_features", {})
    gold = report.get("gold", {})

    if fail_on_contract_error and contracts.get("status") != "passed":
        errors.append("contract_validation_failed")
    if fail_on_gx_error and gx and gx.get("status") != "passed":
        errors.append("great_expectations_failed")
    if bronze.get("input_rows", 0) <= 0:
        errors.append("bronze_manifest_empty")
    if silver_metadata.get("videos_processed", 0) <= 0:
        errors.append("silver_metadata_empty")
    if silver_features.get("videos_processed", 0) <= 0:
        errors.append("silver_video_features_empty")
    if fail_on_empty_gold and gold.get("rows", 0) <= 0:
        errors.append("gold_dataset_empty")
    if fail_on_empty_gold and gold.get("trainable_rows", 0) <= 0:
        errors.append("gold_without_trainable_rows")
    if silver_features.get("avg_missing_feature_ratio", 0.0) > max_missing_feature_ratio:
        errors.append("missing_feature_ratio_above_threshold")
    if silver_metadata.get("fallback_center_ratio", 0.0) > max_fallback_center_ratio:
        errors.append("fallback_center_ratio_above_threshold")

    return errors


def build_metrics_summary(report: dict) -> dict:
    return {
        "pipeline_status": 1 if report.get("status") == "passed" else 0,
        "blocking_error_count": len(report.get("blocking_errors", [])),
        "bronze_input_rows": report.get("bronze", {}).get("input_rows", 0),
        "bronze_downloaded": report.get("bronze", {}).get("downloaded", 0),
        "bronze_failed": report.get("bronze", {}).get("failed", 0),
        "silver_metadata_videos": report.get("silver_metadata", {}).get("videos_processed", 0),
        "silver_metadata_fallback_center_ratio": report.get("silver_metadata", {}).get("fallback_center_ratio", 0.0),
        "silver_features_videos": report.get("silver_features", {}).get("videos_processed", 0),
        "silver_features_avg_missing_feature_ratio": report.get("silver_features", {}).get(
            "avg_missing_feature_ratio", 0.0
        ),
        "gold_rows": report.get("gold", {}).get("rows", 0),
        "gold_trainable_rows": report.get("gold", {}).get("trainable_rows", 0),
        "gold_real_count": report.get("gold", {}).get("real_count", 0),
        "gold_fake_count": report.get("gold", {}).get("fake_count", 0),
        "contract_status": 1 if report.get("contracts", {}).get("status") == "passed" else 0,
        "gx_status": 1 if report.get("great_expectations", {}).get("status") == "passed" else 0,
    }


def write_quality_artifacts(report: dict, reports_dir: str | Path = REPORTS_DIR) -> dict:
    reports_dir = Path(reports_dir)
    metrics_path = write_json(build_metrics_summary(report), pipeline_metrics_path(reports_dir))

    quality_rows = []
    for label, value in report.get("gold", {}).get("quality_flag_distribution", {}).items():
        quality_rows.append({"kind": "quality_flag", "label": label, "count": value})
    for label, value in report.get("gold", {}).get("split_distribution", {}).items():
        quality_rows.append({"kind": "dataset_split", "label": label, "count": value})
    quality_rows.extend(
        [
            {"kind": "target_label", "label": "Real", "count": report.get("gold", {}).get("real_count", 0)},
            {"kind": "target_label", "label": "Fake", "count": report.get("gold", {}).get("fake_count", 0)},
        ]
    )
    distribution_path = write_dataframe(
        pd.DataFrame(quality_rows),
        pipeline_plot_path("gold_distributions.csv", reports_dir),
        index=False,
    )

    stage_rows = [
        {"stage": "bronze", "metric": "input_rows", "value": report.get("bronze", {}).get("input_rows", 0)},
        {"stage": "bronze", "metric": "downloaded", "value": report.get("bronze", {}).get("downloaded", 0)},
        {
            "stage": "silver_metadata",
            "metric": "videos_processed",
            "value": report.get("silver_metadata", {}).get("videos_processed", 0),
        },
        {
            "stage": "silver_features",
            "metric": "videos_processed",
            "value": report.get("silver_features", {}).get("videos_processed", 0),
        },
        {"stage": "gold", "metric": "rows", "value": report.get("gold", {}).get("rows", 0)},
        {"stage": "gold", "metric": "trainable_rows", "value": report.get("gold", {}).get("trainable_rows", 0)},
    ]
    stage_path = write_dataframe(
        pd.DataFrame(stage_rows),
        pipeline_plot_path("stage_counts.csv", reports_dir),
        index=False,
    )
    return {"metrics_path": str(metrics_path), "distribution_plot_path": str(distribution_path), "stage_plot_path": str(stage_path)}


def build_quality_report(
    manifest_path: str | Path = BRONZE_MANIFEST_PATH,
    metadata_dir: str | Path = METADATA_DIR,
    silver_dir: str | Path = SILVER_DIR,
    gold_dir: str | Path = GOLD_DIR,
    include_gx: bool = False,
) -> dict:
    report = {
        "bronze": bronze_quality(manifest_path),
        "silver_metadata": silver_metadata_quality(metadata_dir, silver_dir),
        "silver_features": silver_features_quality(silver_dir),
        "gold": gold_quality(gold_dir),
        "contracts": validate_pipeline_assets(manifest_path, silver_dir, gold_dir),
    }
    if include_gx:
        report["great_expectations"] = gx_validation_report(manifest_path, silver_dir, gold_dir)
    return report
