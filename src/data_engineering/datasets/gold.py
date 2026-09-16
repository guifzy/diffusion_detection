from __future__ import annotations

import argparse
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.shared.core.io_utils import write_dataframe
from src.shared.core.paths import (
    BRONZE_MANIFEST_PATH,
    BRONZE_VIDEOS_DIR,
    GOLD_DIR,
    METADATA_DIR,
    ensure_data_dirs,
    gold_video_region_dataset_path,
    gold_training_dataset_path,
    metadata_path_for_video,
    silver_frame_features_path,
    silver_temporal_features_path,
    silver_video_features_path,
)
from src.shared.core.version import PIPELINE_VERSION
from src.shared.features.extractor import build_video_features
from src.shared.features.temporal import aggregate_region_temporal_features
from src.data_engineering.preprocessing import extract_face_metadata, metadata_is_current

logger = logging.getLogger(__name__)


GOLD_GOVERNANCE_DEFAULTS = {
    "video_id": "",
    "label": "",
    "n_frames": 0,
    "metadata_rows_used": 0,
    "feature_groups_used": "",
    "aggregated_at": "",
    "pipeline_version": PIPELINE_VERSION,
    "missing_feature_ratio": 1.0,
}

REGION_METADATA_COLUMNS = {
    "video_id",
    "label",
    "target_label",
    "dataset_split",
    "is_trainable",
    "quality_flag",
    "region",
    "region_id",
    "region_label",
    "region_type",
    "track_id",
    "region_source",
    "filename",
    "source_url",
    "storage_path",
    "ingestion_status",
    "n_frames",
    "metadata_rows_used",
    "n_temporal_frames",
    "temporal_min_points",
    "temporal_time_span_s",
    "feature_groups_used",
    "aggregated_at",
    "pipeline_version",
    "missing_feature_ratio",
    "temporal_missing_feature_ratio",
}

GOLD_CANONICAL_SIGNAL_TOKENS = (
    "entropy_norm",
    "uniformity",
    "active_bin_ratio",
    "hist_js_distance",
    "magnitude_mean",
    "gradient_energy",
    "orientation_entropy_norm",
    "orientation_coherence",
    "strong_gradient_ratio",
    "signed_variance",
    "signed_energy",
    "abs_mean",
    "abs_p95",
    "rms",
    "mad",
    "channel_corr_mean",
    "horizontal_lag1_corr",
    "vertical_lag1_corr",
    "low_power_ratio",
    "mid_power_ratio",
    "high_power_ratio",
    "spectral_entropy_norm",
    "spectral_flatness",
    "angular_anisotropy",
    "spectral_slope",
    "l_mean",
    "l_contrast_p90_norm",
    "l_entropy_norm",
    "saturation_mean",
    "illumination_mean",
    "illumination_gradient_energy",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
    "candidate_ratio",
    "candidate_depth_mean",
    "boundary_density",
    "kp_density",
    "kp_coverage",
    "response_mean",
    "descriptor_self_similarity",
    "sim_mean",
    "sim_std",
)

GOLD_TEMPORAL_SUFFIXES = (
    "__d1_std",
    "__d1_mad",
    "__d1_p95_abs",
    "__d2_std",
    "__d2_mad",
    "__d2_p95_abs",
    "__lag1_autocorr",
    "__temporal_high_energy_ratio",
)


def _manifest_rows(manifest_path: str | Path, videos_dir: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    if "storage_path" not in manifest.columns and "filename" not in manifest.columns:
        raise ValueError("Bronze manifest must contain storage_path or filename.")

    if "status" in manifest.columns:
        manifest = manifest[manifest["status"].fillna("").isin(["", "downloaded", "skipped"])].copy()

    def resolve_video_path(row: pd.Series) -> str:
        storage_path = str(row.get("storage_path", "") or "").strip()
        if storage_path:
            return storage_path
        filename = str(row.get("filename", "") or "").strip()
        return str(Path(videos_dir) / filename) if filename else ""

    manifest["video_path"] = manifest.apply(resolve_video_path, axis=1)
    manifest = manifest[manifest["video_path"].astype(str).str.len() > 0].copy()
    return manifest


def build_gold_dataset(
    catalog_path: str | Path = BRONZE_MANIFEST_PATH,
    videos_dir: str | Path = BRONZE_VIDEOS_DIR,
    metadata_dir: str | Path = METADATA_DIR,
    output_path: str | Path | None = None,
    silver_output_path: str | Path | None = None,
    groups: str = "abcde",
    max_frames: int | None = None,
    sample_fps: float | None = None,
    temporal_min_points: int = 3,
    generate_missing_metadata: bool = False,
    overwrite_metadata: bool = False,
    face_detector_model_path: str | Path | None = None,
    face_model_path: str | Path | None = None,
    segmenter_model_path: str | Path | None = None,
    max_faces: int = 10,
    face_detection_confidence: float = 0.3,
    face_landmark_confidence: float = 0.3,
    limit: int | None = None,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    missing_feature_threshold: float = 0.5,
) -> pd.DataFrame:
    ensure_data_dirs()
    output_path = Path(output_path) if output_path else gold_training_dataset_path(GOLD_DIR)
    silver_output_path = Path(silver_output_path) if silver_output_path else silver_video_features_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    silver_output_path.parent.mkdir(parents=True, exist_ok=True)

    catalog = _manifest_rows(catalog_path, videos_dir)
    if limit is not None:
        catalog = catalog.head(limit)

    rows = []
    temporal_rows = []
    for _, row in catalog.iterrows():
        video_path = Path(row["video_path"])
        if not video_path.exists():
            logger.warning("Skipping missing video: %s", video_path)
            continue

        metadata_path = metadata_path_for_video(video_path, metadata_dir)
        if (overwrite_metadata or not metadata_is_current(metadata_path)) and generate_missing_metadata:
            extract_face_metadata(
                video_path,
                metadata_path,
                max_frames=max_frames,
                sample_fps=sample_fps,
                face_detector_model_path=face_detector_model_path,
                face_model_path=face_model_path,
                segmenter_model_path=segmenter_model_path,
                max_faces=max_faces,
                face_detection_confidence=face_detection_confidence,
                face_landmark_confidence=face_landmark_confidence,
            )

        if not metadata_path.exists():
            logger.warning("Skipping video without metadata: %s", video_path)
            continue
        if not metadata_is_current(metadata_path):
            logger.warning("Skipping video with stale metadata: %s", metadata_path)
            continue

        label = row.get("label")
        try:
            frame_features, video_features = build_video_features(
                video_path,
                metadata_path,
                groups=groups,
                max_frames=max_frames,
                sample_fps=sample_fps,
                label=label,
            )
        except Exception as exc:
            logger.exception("Failed to extract features for %s: %s", video_path, exc)
            continue

        frame_saved_path = write_dataframe(frame_features, silver_frame_features_path(video_path), index=False)
        logger.info("Saved Silver frame features for %s to %s", video_path.name, frame_saved_path)

        if isinstance(video_features, pd.DataFrame):
            video_features = video_features.copy()
            video_features["filename"] = video_path.name
            video_features["source_url"] = row.get("source_url", "")
            video_features["storage_path"] = str(video_path)
            video_features["ingestion_status"] = row.get("status", "")
            rows.extend(video_features.to_dict(orient="records"))
        else:
            video_features["filename"] = video_path.name
            video_features["source_url"] = row.get("source_url", "")
            video_features["storage_path"] = str(video_path)
            video_features["ingestion_status"] = row.get("status", "")
            rows.append(video_features)

        temporal_features = aggregate_region_temporal_features(
            frame_features,
            groups=groups,
            min_points=temporal_min_points,
        )
        if not temporal_features.empty:
            temporal_features = temporal_features.copy()
            temporal_features["filename"] = video_path.name
            temporal_features["source_url"] = row.get("source_url", "")
            temporal_features["storage_path"] = str(video_path)
            temporal_features["ingestion_status"] = row.get("status", "")
            temporal_rows.extend(temporal_features.to_dict(orient="records"))

    silver_video_features = pd.DataFrame(rows)
    silver_saved_path = write_dataframe(silver_video_features, silver_output_path, index=False)
    logger.info("Saved Silver video features with %s rows to %s", len(silver_video_features), silver_saved_path)

    silver_temporal_features = pd.DataFrame(temporal_rows)
    temporal_saved_path = write_dataframe(silver_temporal_features, silver_temporal_features_path(), index=False)
    logger.info("Saved Silver temporal features with %s rows to %s", len(silver_temporal_features), temporal_saved_path)

    gold_region_dataset = merge_static_temporal_region_features(silver_video_features, silver_temporal_features)
    gold_region_dataset = ensure_gold_governance_columns(gold_region_dataset)
    if not gold_region_dataset.empty:
        gold_region_dataset["target_label"] = gold_region_dataset["label"]
        gold_region_dataset["quality_flag"] = gold_region_dataset.apply(
            lambda row: _quality_flag(row, missing_feature_threshold=missing_feature_threshold), axis=1
        )
        gold_region_dataset["is_trainable"] = (
            gold_region_dataset["target_label"].isin(["Real", "Fake"])
            & (gold_region_dataset["n_frames"] > 0)
            & (gold_region_dataset["metadata_rows_used"] > 0)
            & (gold_region_dataset["missing_feature_ratio"] <= missing_feature_threshold)
            & (gold_region_dataset["quality_flag"] == "ok")
        )
        gold_region_dataset["dataset_split"] = assign_dataset_splits(
            gold_region_dataset,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )
        gold_region_dataset["pipeline_version"] = gold_region_dataset.get("pipeline_version", PIPELINE_VERSION)
    regional_saved_path = write_dataframe(gold_region_dataset, gold_video_region_dataset_path(), index=False)
    logger.info("Saved Gold video-region dataset with %s rows to %s", len(gold_region_dataset), regional_saved_path)

    gold_dataset = build_video_level_gold_dataset(
        gold_region_dataset,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        missing_feature_threshold=missing_feature_threshold,
    )
    saved_path = write_dataframe(gold_dataset, output_path, index=False)
    logger.info("Saved Gold training dataset with %s rows to %s", len(gold_dataset), saved_path)
    return gold_dataset


def merge_static_temporal_region_features(
    static_features: pd.DataFrame,
    temporal_features: pd.DataFrame,
) -> pd.DataFrame:
    if static_features.empty:
        return static_features.copy()
    if temporal_features.empty:
        return static_features.copy()

    merge_keys = [
        column
        for column in ("video_id", "region", "region_type", "track_id")
        if column in static_features.columns and column in temporal_features.columns
    ]
    temporal = temporal_features.copy()
    duplicate_columns = [
        column
        for column in temporal.columns
        if column not in merge_keys and column in static_features.columns
    ]
    temporal = temporal.drop(columns=duplicate_columns)
    return static_features.merge(temporal, on=merge_keys, how="left")


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in REGION_METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(df[column])
        and any(token in column for token in GOLD_CANONICAL_SIGNAL_TOKENS)
        and (not column.startswith("temporal__") or column.endswith(GOLD_TEMPORAL_SUFFIXES))
    ]


def _safe_region_prefix(region_type: str) -> str:
    return str(region_type or "unknown").strip().lower().replace(" ", "_")


def _max_int_or_zero(values: pd.Series) -> int:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return int(numeric.max()) if not numeric.empty else 0


def build_video_level_gold_dataset(
    region_dataset: pd.DataFrame,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    missing_feature_threshold: float = 0.5,
) -> pd.DataFrame:
    if region_dataset.empty:
        return ensure_gold_governance_columns(pd.DataFrame())

    feature_cols = _feature_columns(region_dataset)
    rows = []
    for video_id, group in region_dataset.groupby("video_id", dropna=False):
        row = {
            "video_id": video_id,
            "label": group["label"].dropna().astype(str).iloc[0] if "label" in group and not group["label"].dropna().empty else "",
            "target_label": group["target_label"].dropna().astype(str).iloc[0]
            if "target_label" in group and not group["target_label"].dropna().empty
            else "",
            "filename": group["filename"].dropna().astype(str).iloc[0]
            if "filename" in group and not group["filename"].dropna().empty
            else "",
            "source_url": group["source_url"].dropna().astype(str).iloc[0]
            if "source_url" in group and not group["source_url"].dropna().empty
            else "",
            "storage_path": group["storage_path"].dropna().astype(str).iloc[0]
            if "storage_path" in group and not group["storage_path"].dropna().empty
            else "",
            "n_regions": int(len(group)),
            "n_frames": _max_int_or_zero(group.get("n_frames", pd.Series(dtype=float))),
            "metadata_rows_used": _max_int_or_zero(group.get("metadata_rows_used", pd.Series(dtype=float))),
            "feature_groups_used": group["feature_groups_used"].dropna().astype(str).iloc[0]
            if "feature_groups_used" in group and not group["feature_groups_used"].dropna().empty
            else "",
            "aggregated_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": PIPELINE_VERSION,
        }

        for region_type, region_group in group.groupby("region_type", dropna=False):
            prefix = _safe_region_prefix(region_type)
            row[f"{prefix}__region_count"] = int(len(region_group))
            row[f"{prefix}__track_count"] = (
                int(region_group["track_id"].dropna().astype(str).nunique()) if "track_id" in region_group else 0
            )
            for feature in feature_cols:
                values = pd.to_numeric(region_group[feature], errors="coerce")
                row[f"{prefix}__{feature}"] = float(values.mean()) if values.notna().any() else float("nan")

        numeric_features = {
            key: value
            for key, value in row.items()
            if key not in REGION_METADATA_COLUMNS
            and isinstance(value, (int, float))
        }
        row["missing_feature_ratio"] = float(pd.Series(numeric_features).isna().mean()) if numeric_features else 1.0
        row["quality_flag"] = _quality_flag(pd.Series(row), missing_feature_threshold=missing_feature_threshold)
        row["is_trainable"] = (
            row["target_label"] in {"Real", "Fake"}
            and row["n_frames"] > 0
            and row["metadata_rows_used"] > 0
            and row["missing_feature_ratio"] <= missing_feature_threshold
            and row["quality_flag"] == "ok"
        )
        rows.append(row)

    gold = pd.DataFrame(rows)
    gold["dataset_split"] = assign_dataset_splits(
        gold,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )
    return gold


def ensure_gold_governance_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column, default in GOLD_GOVERNANCE_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default
    return df


def _quality_flag(row: pd.Series, missing_feature_threshold: float = 0.5) -> str:
    if row.get("target_label") not in {"Real", "Fake"}:
        return "missing_label"
    if row.get("n_frames", 0) <= 0 or row.get("metadata_rows_used", 0) <= 0:
        return "insufficient_metadata"
    if row.get("missing_feature_ratio", 1.0) >= 1.0:
        return "feature_failure"
    if row.get("missing_feature_ratio", 1.0) > missing_feature_threshold:
        return "review"
    return "ok"


def _stable_hash_fraction(value: str, seed: int = 42) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def assign_dataset_splits(
    gold_dataset: pd.DataFrame,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> pd.Series:
    splits = pd.Series("unassigned", index=gold_dataset.index, dtype="object")
    if gold_dataset.empty:
        return splits

    trainable = gold_dataset[gold_dataset["is_trainable"]].copy()
    if trainable.empty:
        return splits

    test_cutoff = train_ratio + validation_ratio
    for label, group in trainable.groupby("target_label"):
        for idx, row in group.iterrows():
            key = str(row.get("video_id") or row.get("filename") or idx)
            fraction = _stable_hash_fraction(f"{label}:{key}", seed=seed)
            if fraction < train_ratio:
                splits.loc[idx] = "train"
            elif fraction < test_cutoff:
                splits.loc[idx] = "validation"
            else:
                splits.loc[idx] = "test"
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local Gold dataset for model training.")
    parser.add_argument("--manifest", "--catalog", dest="manifest", type=Path, default=BRONZE_MANIFEST_PATH)
    parser.add_argument("--videos-dir", type=Path, default=BRONZE_VIDEOS_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=METADATA_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--silver-output", type=Path, default=None)
    parser.add_argument("--groups", default="abcde")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--sample-fps", type=float)
    parser.add_argument("--temporal-min-points", type=int, default=3)
    parser.add_argument("--generate-missing-metadata", action="store_true")
    parser.add_argument("--overwrite-metadata", action="store_true")
    parser.add_argument("--face-detector-model", type=Path, default=None)
    parser.add_argument("--face-model", type=Path, default=None)
    parser.add_argument("--segmenter-model", type=Path, default=None)
    parser.add_argument("--max-faces", type=int, default=10)
    parser.add_argument("--face-detection-confidence", type=float, default=0.3)
    parser.add_argument("--face-landmark-confidence", type=float, default=0.3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--missing-feature-threshold", type=float, default=0.5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_gold_dataset(
        catalog_path=args.manifest,
        videos_dir=args.videos_dir,
        metadata_dir=args.metadata_dir,
        output_path=args.output,
        silver_output_path=args.silver_output,
        groups=args.groups,
        max_frames=args.max_frames,
        sample_fps=args.sample_fps,
        temporal_min_points=args.temporal_min_points,
        generate_missing_metadata=args.generate_missing_metadata,
        overwrite_metadata=args.overwrite_metadata,
        face_detector_model_path=args.face_detector_model,
        face_model_path=args.face_model,
        segmenter_model_path=args.segmenter_model,
        max_faces=args.max_faces,
        face_detection_confidence=args.face_detection_confidence,
        face_landmark_confidence=args.face_landmark_confidence,
        limit=args.limit,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        missing_feature_threshold=args.missing_feature_threshold,
    )


if __name__ == "__main__":
    main()
