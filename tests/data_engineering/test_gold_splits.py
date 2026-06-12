from __future__ import annotations

import pandas as pd

from src.data_engineering.datasets.gold import assign_dataset_splits


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

