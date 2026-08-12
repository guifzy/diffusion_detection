from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import pandas as pd

from src.shared.core.io_utils import write_dataframe
from src.shared.core.paths import (
    BRONZE_MANIFEST_PATH,
    BRONZE_VIDEOS_DIR,
    GOLD_DIR,
    METADATA_DIR,
    ensure_data_dirs,
    gold_training_dataset_path,
    metadata_path_for_video,
    silver_frame_features_path,
    silver_video_features_path,
)
from src.shared.core.version import PIPELINE_VERSION
from src.shared.features.extractor import build_video_features
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
    generate_missing_metadata: bool = False,
    overwrite_metadata: bool = False,
    face_model_path: str | Path | None = None,
    segmenter_model_path: str | Path | None = None,
    max_faces: int = 10,
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
                face_model_path=face_model_path,
                segmenter_model_path=segmenter_model_path,
                max_faces=max_faces,
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

    silver_video_features = pd.DataFrame(rows)
    silver_saved_path = write_dataframe(silver_video_features, silver_output_path, index=False)
    logger.info("Saved Silver video features with %s rows to %s", len(silver_video_features), silver_saved_path)

    gold_dataset = ensure_gold_governance_columns(silver_video_features.copy())
    if not gold_dataset.empty:
        gold_dataset["target_label"] = gold_dataset["label"]
        gold_dataset["quality_flag"] = gold_dataset.apply(
            lambda row: _quality_flag(row, missing_feature_threshold=missing_feature_threshold), axis=1
        )
        gold_dataset["is_trainable"] = (
            gold_dataset["target_label"].isin(["Real", "Fake"])
            & (gold_dataset["n_frames"] > 0)
            & (gold_dataset["metadata_rows_used"] > 0)
            & (gold_dataset["missing_feature_ratio"] <= missing_feature_threshold)
            & (gold_dataset["quality_flag"] == "ok")
        )
        gold_dataset["dataset_split"] = assign_dataset_splits(
            gold_dataset,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )
        gold_dataset["pipeline_version"] = gold_dataset.get("pipeline_version", PIPELINE_VERSION)
    saved_path = write_dataframe(gold_dataset, output_path, index=False)
    logger.info("Saved Gold training dataset with %s rows to %s", len(gold_dataset), saved_path)
    return gold_dataset


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
    parser.add_argument("--generate-missing-metadata", action="store_true")
    parser.add_argument("--overwrite-metadata", action="store_true")
    parser.add_argument("--face-model", type=Path, default=None)
    parser.add_argument("--segmenter-model", type=Path, default=None)
    parser.add_argument("--max-faces", type=int, default=10)
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
        generate_missing_metadata=args.generate_missing_metadata,
        overwrite_metadata=args.overwrite_metadata,
        face_model_path=args.face_model,
        segmenter_model_path=args.segmenter_model,
        max_faces=args.max_faces,
        limit=args.limit,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        missing_feature_threshold=args.missing_feature_threshold,
    )


if __name__ == "__main__":
    main()
