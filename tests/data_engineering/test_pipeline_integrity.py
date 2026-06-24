from __future__ import annotations

from pathlib import Path

from src.data_engineering.pipeline.run import parse_bool
from src.data_engineering.pipeline.quality import blocking_errors_for_report, build_metrics_summary


def test_dvc_pipeline_uses_source_csv_then_bronze_manifest() -> None:
    dvc_yaml = Path("dvc.yaml").read_text(encoding="utf-8")

    assert "--source-csv ${pipeline.source_csv}" in dvc_yaml
    assert "--manifest ${pipeline.manifest}" in dvc_yaml
    assert "${pipeline.source_csv}" in dvc_yaml
    assert "${pipeline.manifest}" in dvc_yaml
    assert "--generate-missing-metadata ${pipeline.generate_missing_metadata}" in dvc_yaml


def test_pipeline_bool_parser_accepts_dvc_values() -> None:
    assert parse_bool("true") is True
    assert parse_bool("false") is False
    assert parse_bool(True) is True


def test_quality_report_blocks_empty_downstream_layers() -> None:
    report = {
        "status": "failed",
        "contracts": {"status": "passed"},
        "bronze": {"input_rows": 2},
        "silver_metadata": {"videos_processed": 0, "fallback_center_ratio": 0.0},
        "silver_features": {"videos_processed": 0, "avg_missing_feature_ratio": 0.0},
        "gold": {"rows": 0, "trainable_rows": 0},
    }

    errors = blocking_errors_for_report(report)
    metrics = build_metrics_summary({**report, "blocking_errors": errors})

    assert "silver_metadata_empty" in errors
    assert "gold_dataset_empty" in errors
    assert metrics["blocking_error_count"] == len(errors)
