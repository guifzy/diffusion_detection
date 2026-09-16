from __future__ import annotations

import pandas as pd

from src.data_engineering.ingestion.df26 import build_df26_manifest


def test_build_df26_manifest_preserves_training_roles_and_logos_splits(tmp_path) -> None:
    dataset_dir = tmp_path / "df26"
    metadata_dir = dataset_dir / "metadata"
    metadata_dir.mkdir(parents=True)

    rows = [
        {"path": "real/Studio_Interview/clip001.mp4"},
        {"path": "fake/Studio_Interview/HunyuanVideo 1.5 I2V/clip001.mp4"},
        {"path": "fake/Studio_Interview/LTX_2.3_distilled_i2v/clip001.mp4"},
        {"path": "fake/Studio_Interview/Wan_2.2_14b_i2v/clip001.mp4"},
        {"path": "fake/Studio_Interview/Kling_3.0/clip001.mp4"},
    ]
    pd.DataFrame(rows).to_csv(metadata_dir / "videos.csv", index=False)
    for row in rows:
        path = dataset_dir / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not-a-real-video")

    manifest_path = tmp_path / "bronze_manifest_df26.csv"
    splits_path = tmp_path / "df26_logos_splits.csv"
    manifest, splits = build_df26_manifest(dataset_dir, manifest_path, splits_path)

    assert manifest_path.exists()
    assert splits_path.exists()
    assert set(manifest["label"]) == {"Real", "Fake"}
    assert set(manifest["source_type"]) == {"dataset_local"}

    commercial = manifest[manifest["df26_generator"].eq("Kling_3.0")].iloc[0]
    assert commercial["df26_training_role"] == "commercial_evaluation_only"
    assert commercial["df26_allowed_for_training"] == False  # noqa: E712

    open_weight = manifest[manifest["df26_generator"].eq("HunyuanVideo 1.5 I2V")].iloc[0]
    assert open_weight["df26_training_role"] == "open_weight_trainable"
    assert open_weight["df26_allowed_for_training"] == True  # noqa: E712
    assert open_weight["df26_generator_canonical"] == "HunyuanVideo_1.5_14b_i2v"

    held_out = splits[
        splits["df26_logos_fold"].eq("logos_leave_HunyuanVideo_1.5")
        & splits["df26_generator"].eq("HunyuanVideo 1.5 I2V")
    ].iloc[0]
    assert held_out["dataset_split"] == "test"

    commercial_split = splits[
        splits["df26_logos_fold"].eq("logos_leave_HunyuanVideo_1.5")
        & splits["df26_generator"].eq("Kling_3.0")
    ].iloc[0]
    assert commercial_split["dataset_split"] == "holdout_commercial"
