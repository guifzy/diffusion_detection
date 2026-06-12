from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataContract:
    name: str
    layer: str
    grain: str
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...] = ()
    accepted_values: dict[str, tuple[str, ...]] | None = None
    description: str = ""


BRONZE_MANIFEST_COLUMNS = (
    "video_id",
    "source_url",
    "filename",
    "storage_path",
    "sha256",
    "downloaded_at",
    "label",
    "status",
    "error_message",
    "source_type",
)

FRAME_METADATA_COLUMNS = (
    "video_id",
    "frame_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_expanded_x1",
    "bbox_expanded_y1",
    "bbox_expanded_x2",
    "bbox_expanded_y2",
    "source",
    "detector_score",
    "frame_width",
    "frame_height",
    "processed_at",
    "pipeline_version",
)

FRAME_FEATURES_COLUMNS = (
    "video_id",
    "frame_id",
    "metadata_idx",
    "label",
    "feature_groups_used",
    "processed_at",
    "pipeline_version",
)

VIDEO_FEATURES_EXTRA_COLUMNS = (
    "video_id",
    "label",
    "n_frames",
    "metadata_rows_used",
    "feature_groups_used",
    "aggregated_at",
    "pipeline_version",
    "missing_feature_ratio",
)

GOLD_TRAINING_EXTRA_COLUMNS = (
    "video_id",
    "target_label",
    "dataset_split",
    "is_trainable",
    "quality_flag",
    "missing_feature_ratio",
    "pipeline_version",
)

PREDICTION_PAYLOAD_FIELDS = (
    "video_id",
    "prediction",
    "score_fake",
    "score_real",
    "model_version",
    "feature_pipeline_version",
    "processed_at",
    "top_signals",
    "processing_status",
    "error_message",
)


CONTRACTS = {
    "bronze_manifest": DataContract(
        name="bronze_manifest",
        layer="bronze",
        grain="one row per source video",
        required_columns=BRONZE_MANIFEST_COLUMNS,
        accepted_values={
            "label": ("Real", "Fake", ""),
            "status": ("pending", "downloaded", "failed", "skipped"),
            "source_type": ("youtube", "manual_upload", "dataset_local"),
        },
        description="Catalog of raw videos and ingestion status.",
    ),
    "frame_metadata": DataContract(
        name="frame_metadata",
        layer="silver",
        grain="one row per processed frame",
        required_columns=FRAME_METADATA_COLUMNS,
        accepted_values={
            "source": ("detector", "tracker", "last_bbox", "fallback_center"),
        },
        description="Face-region metadata extracted from each processed frame.",
    ),
    "frame_features": DataContract(
        name="frame_features",
        layer="silver",
        grain="one row per processed frame",
        required_columns=FRAME_FEATURES_COLUMNS,
        description="Frame-level forensic signals extracted by feature groups A-E.",
    ),
    "video_features": DataContract(
        name="video_features",
        layer="silver",
        grain="one row per video",
        required_columns=VIDEO_FEATURES_EXTRA_COLUMNS,
        description="Video-level aggregation of frame features before ML curation.",
    ),
    "gold_training_dataset": DataContract(
        name="gold_training_dataset",
        layer="gold",
        grain="one row per trainable video",
        required_columns=GOLD_TRAINING_EXTRA_COLUMNS,
        accepted_values={
            "dataset_split": ("train", "validation", "test", "unassigned"),
            "quality_flag": ("ok", "review", "insufficient_metadata", "missing_label", "feature_failure"),
        },
        description="ML-ready dataset consumed by model training.",
    ),
    "prediction_payload": DataContract(
        name="prediction_payload",
        layer="serving",
        grain="one response per analyzed video",
        required_columns=PREDICTION_PAYLOAD_FIELDS,
        accepted_values={
            "prediction": ("Real", "Fake", "Unknown"),
            "processing_status": ("success", "failed", "partial"),
        },
        description="Inference response contract for the future SaaS backend.",
    ),
}


def missing_required_columns(contract_name: str, columns: set[str]) -> list[str]:
    contract = CONTRACTS[contract_name]
    return [column for column in contract.required_columns if column not in columns]

