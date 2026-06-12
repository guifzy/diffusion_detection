from __future__ import annotations

import pandas as pd

from src.shared.contracts import validate_dataframe_contract
from src.shared.contracts.schemas import BRONZE_MANIFEST_COLUMNS


def test_bronze_manifest_contract_accepts_valid_rows() -> None:
    df = pd.DataFrame(
        [
            {
                "video_id": "abc",
                "source_url": "https://youtube.com/watch?v=abc",
                "filename": "abc.mp4",
                "storage_path": "data/bronze/videos/abc.mp4",
                "sha256": "hash",
                "downloaded_at": "2026-01-01T00:00:00+00:00",
                "label": "Real",
                "status": "downloaded",
                "error_message": "",
                "source_type": "youtube",
            }
        ],
        columns=BRONZE_MANIFEST_COLUMNS,
    )

    result = validate_dataframe_contract(df, "bronze_manifest")

    assert result.status == "passed"
    assert result.missing_columns == ()
    assert result.invalid_values == {}


def test_bronze_manifest_contract_rejects_invalid_status() -> None:
    df = pd.DataFrame(
        [
            {
                "video_id": "abc",
                "source_url": "https://youtube.com/watch?v=abc",
                "filename": "abc.mp4",
                "storage_path": "data/bronze/videos/abc.mp4",
                "sha256": "hash",
                "downloaded_at": "2026-01-01T00:00:00+00:00",
                "label": "Real",
                "status": "ok",
                "error_message": "",
                "source_type": "youtube",
            }
        ],
        columns=BRONZE_MANIFEST_COLUMNS,
    )

    result = validate_dataframe_contract(df, "bronze_manifest")

    assert result.status == "failed"
    assert result.invalid_values["status"] == ("ok",)

