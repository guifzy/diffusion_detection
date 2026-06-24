from __future__ import annotations

from pathlib import Path

from src.data_engineering.datasets.gold import _manifest_rows


def test_manifest_rows_uses_bronze_manifest_storage_path(tmp_path) -> None:
    videos_dir = tmp_path / "videos"
    manifest = tmp_path / "bronze_manifest.csv"
    manifest.write_text(
        "video_id,source_url,filename,storage_path,sha256,downloaded_at,label,status,error_message,source_type\n"
        f"abc,https://youtube.com/watch?v=abc,abc.mp4,{videos_dir / 'abc.mp4'},hash,2026-01-01T00:00:00+00:00,Real,downloaded,,youtube\n"
        f"bad,https://youtube.com/watch?v=bad,,,,2026-01-01T00:00:00+00:00,Fake,failed,error,youtube\n",
        encoding="utf-8",
    )

    rows = _manifest_rows(manifest, videos_dir)

    assert rows["video_path"].tolist() == [str(videos_dir / "abc.mp4")]
    assert rows["label"].tolist() == ["Real"]
