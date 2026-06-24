from __future__ import annotations

from src.data_engineering.infra.minio_dvc import lake_object_name


def test_lake_object_name_keeps_layer_relative_path(tmp_path) -> None:
    data_dir = tmp_path / "data"
    local_file = data_dir / "gold" / "gold_training_dataset.parquet"
    local_file.parent.mkdir(parents=True)
    local_file.write_text("placeholder", encoding="utf-8")

    assert lake_object_name(local_file, data_dir=data_dir, prefix="lake") == "lake/gold/gold_training_dataset.parquet"
