from __future__ import annotations

import pytest

from src.data_engineering.ingestion.youtube import merge_manifest_rows, read_source_csv


def test_source_csv_accepts_only_link_and_boolean_label(tmp_path) -> None:
    source_csv = tmp_path / "source.csv"
    source_csv.write_text(
        "link,label\n"
        "https://www.youtube.com/watch?v=abc,true\n"
        "https://www.youtube.com/watch?v=def,false\n",
        encoding="utf-8",
    )

    records = read_source_csv(source_csv)

    assert records == [
        {"source_url": "https://www.youtube.com/watch?v=abc", "label": "Real", "source_type": "youtube"},
        {"source_url": "https://www.youtube.com/watch?v=def", "label": "Fake", "source_type": "youtube"},
    ]


def test_source_csv_rejects_missing_label_column(tmp_path) -> None:
    source_csv = tmp_path / "source.csv"
    source_csv.write_text("link\nhttps://www.youtube.com/watch?v=abc\n", encoding="utf-8")

    with pytest.raises(ValueError, match="label"):
        read_source_csv(source_csv)


def test_source_csv_rejects_non_boolean_label(tmp_path) -> None:
    source_csv = tmp_path / "source.csv"
    source_csv.write_text("link,label\nhttps://www.youtube.com/watch?v=abc,Real\n", encoding="utf-8")

    with pytest.raises(ValueError, match="true or false"):
        read_source_csv(source_csv)


def test_manifest_merge_is_idempotent_for_existing_download() -> None:
    existing = [
        {
            "video_id": "abc",
            "source_url": "https://www.youtube.com/watch?v=abc",
            "filename": "abc.mp4",
            "status": "downloaded",
            "downloaded_at": "first",
        }
    ]
    skipped = [
        {
            "video_id": "abc",
            "source_url": "https://www.youtube.com/watch?v=abc",
            "filename": "abc.mp4",
            "status": "skipped",
            "downloaded_at": "second",
        }
    ]

    merged = merge_manifest_rows(existing, skipped)

    assert merged == existing
