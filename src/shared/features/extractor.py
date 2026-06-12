from __future__ import annotations

from pathlib import Path
from typing import Iterable
from datetime import datetime, timezone

import pandas as pd

from src.shared.core.version import PIPELINE_VERSION
from src.shared.features.common import aggregate_video_metrics, extract_frame_metrics
from src.shared.features.group_a import GROUP_A_FUNCTIONS, GROUP_A_PREFIXES
from src.shared.features.group_b import GROUP_B_FUNCTIONS, GROUP_B_PREFIXES
from src.shared.features.group_c import GROUP_C_FUNCTIONS, GROUP_C_PREFIXES
from src.shared.features.group_d import GROUP_D_FUNCTIONS, GROUP_D_PREFIXES
from src.shared.features.group_e import GROUP_E_FUNCTIONS, GROUP_E_PREFIXES


GROUPS = {
    "a": (GROUP_A_FUNCTIONS, GROUP_A_PREFIXES),
    "b": (GROUP_B_FUNCTIONS, GROUP_B_PREFIXES),
    "c": (GROUP_C_FUNCTIONS, GROUP_C_PREFIXES),
    "d": (GROUP_D_FUNCTIONS, GROUP_D_PREFIXES),
    "e": (GROUP_E_FUNCTIONS, GROUP_E_PREFIXES),
}


def normalize_groups(groups: Iterable[str] | str = "abcde") -> list[str]:
    if isinstance(groups, str):
        groups = list(groups.lower())
    normalized = [group.lower() for group in groups]
    unknown = sorted(set(normalized) - set(GROUPS))
    if unknown:
        raise ValueError(f"Unknown feature groups: {unknown}")
    return normalized


def groups_to_string(groups: Iterable[str] | str = "abcde") -> str:
    return "".join(normalize_groups(groups))


def metric_functions_for_groups(groups: Iterable[str] | str = "abcde"):
    functions = []
    for group in normalize_groups(groups):
        functions.extend(GROUPS[group][0])
    return functions


def prefixes_for_groups(groups: Iterable[str] | str = "abcde") -> tuple[str, ...]:
    prefixes = []
    for group in normalize_groups(groups):
        prefixes.extend(GROUPS[group][1])
    return tuple(prefixes)


def extract_video_frame_features(
    video_path: str | Path,
    metadata_path: str | Path,
    groups: Iterable[str] | str = "abcde",
    max_frames: int | None = None,
    label: str | None = None,
) -> pd.DataFrame:
    frame_features = extract_frame_metrics(
        video_path,
        metadata_path,
        metric_functions_for_groups(groups),
        max_frames=max_frames,
        label=label,
    )
    if not frame_features.empty:
        frame_features["feature_groups_used"] = groups_to_string(groups)
        frame_features["processed_at"] = datetime.now(timezone.utc).isoformat()
        frame_features["pipeline_version"] = PIPELINE_VERSION
    return frame_features


def aggregate_video_features(
    frame_features: pd.DataFrame,
    groups: Iterable[str] | str = "abcde",
    video_id: str | None = None,
    label: str | None = None,
) -> dict:
    values = aggregate_video_metrics(frame_features, prefixes_for_groups(groups))
    if video_id is not None:
        values["video_id"] = video_id
    if label is not None:
        values["label"] = label
    values["n_frames"] = int(len(frame_features))
    if "metadata_idx" in frame_features.columns:
        values["metadata_rows_used"] = int(frame_features["metadata_idx"].nunique())
    values["feature_groups_used"] = groups_to_string(groups)
    values["aggregated_at"] = datetime.now(timezone.utc).isoformat()
    values["pipeline_version"] = PIPELINE_VERSION
    feature_values = {
        key: value
        for key, value in values.items()
        if key not in {"video_id", "label", "feature_groups_used", "aggregated_at", "pipeline_version"}
    }
    if feature_values:
        values["missing_feature_ratio"] = float(pd.Series(feature_values).isna().mean())
    else:
        values["missing_feature_ratio"] = 1.0
    return values


def build_video_features(
    video_path: str | Path,
    metadata_path: str | Path,
    groups: Iterable[str] | str = "abcde",
    max_frames: int | None = None,
    label: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    frame_features = extract_video_frame_features(
        video_path,
        metadata_path,
        groups=groups,
        max_frames=max_frames,
        label=label,
    )
    video_features = aggregate_video_features(
        frame_features,
        groups=groups,
        video_id=Path(video_path).stem,
        label=label,
    )
    return frame_features, video_features
