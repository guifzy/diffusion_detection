from __future__ import annotations

import pandas as pd

from src.data_engineering.datasets.gold import assign_dataset_splits, build_video_level_gold_dataset


def test_assign_dataset_splits_is_reproducible() -> None:
    df = pd.DataFrame(
        {
            "video_id": [f"v{i}" for i in range(20)],
            "target_label": ["Real", "Fake"] * 10,
            "is_trainable": [True] * 20,
        }
    )

    first = assign_dataset_splits(df)
    second = assign_dataset_splits(df)

    assert first.tolist() == second.tolist()
    assert set(first.unique()) <= {"train", "validation", "test", "unassigned"}


def test_assign_dataset_splits_keeps_non_trainable_unassigned() -> None:
    df = pd.DataFrame(
        {
            "video_id": ["a", "b"],
            "target_label": ["Real", "Fake"],
            "is_trainable": [True, False],
        }
    )

    splits = assign_dataset_splits(df)

    assert splits.loc[1] == "unassigned"


def test_build_video_level_gold_dataset_collapses_regions_to_video_rows() -> None:
    region_dataset = pd.DataFrame(
        {
            "video_id": ["v1", "v1", "v2"],
            "region": ["rosto_completo_1", "fundo", "rosto_completo_1"],
            "region_type": ["rosto_completo", "fundo", "rosto_completo"],
            "track_id": ["face_1", "global", "face_1"],
            "label": ["Fake", "Fake", "Real"],
            "target_label": ["Fake", "Fake", "Real"],
            "n_frames": [5, 5, 5],
            "metadata_rows_used": [5, 5, 5],
            "feature_groups_used": ["a", "a", "a"],
            "pipeline_version": ["0.3.0", "0.3.0", "0.3.0"],
            "missing_feature_ratio": [0.0, 0.0, 0.0],
            "quality_flag": ["ok", "ok", "ok"],
            "is_trainable": [True, True, True],
            "lbp_face_entropy_norm_mean": [0.4, 0.1, 0.8],
            "temporal__lbp_face_entropy_norm__d1_std": [0.05, 0.02, 0.01],
        }
    )

    gold = build_video_level_gold_dataset(region_dataset)

    assert len(gold) == 2
    assert "rosto_completo__lbp_face_entropy_norm_mean" in gold.columns
    assert "fundo__temporal__lbp_face_entropy_norm__d1_std" in gold.columns
    assert set(gold["target_label"]) == {"Fake", "Real"}
