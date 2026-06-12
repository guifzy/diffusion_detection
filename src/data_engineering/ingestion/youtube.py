from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.shared.contracts import BRONZE_MANIFEST_COLUMNS
from src.shared.core.paths import BRONZE_MANIFEST_PATH, BRONZE_VIDEOS_DIR, ensure_data_dirs

logger = logging.getLogger(__name__)


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_links(args_links: list[str], links_file: Path | None) -> list[str]:
    links = list(args_links)
    if links_file:
        links.extend(line.strip() for line in links_file.read_text(encoding="utf-8").splitlines() if line.strip())
    return list(dict.fromkeys(links))


def read_source_csv(csv_path: str | Path, url_column: str = "Media", label_column: str = "Video Ground Truth") -> list[dict]:
    with Path(csv_path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    records = []
    for row in rows:
        source_url = (row.get(url_column) or "").strip()
        if not source_url:
            continue
        records.append(
            {
                "source_url": source_url,
                "label": (row.get(label_column) or "").strip(),
                "source_type": "youtube",
            }
        )
    return records


def normalize_label(label: str | None) -> str:
    value = (label or "").strip()
    lowered = value.lower()
    if lowered in {"real", "true", "verdadeiro"}:
        return "Real"
    if lowered in {"fake", "false", "falso", "ia", "ai"}:
        return "Fake"
    return value


def infer_youtube_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname and "youtu.be" in parsed.hostname:
        return parsed.path.strip("/").split("/")[0]
    query_id = parse_qs(parsed.query).get("v", [""])[0]
    if query_id:
        return query_id
    parts = [part for part in parsed.path.split("/") if part]
    if "shorts" in parts:
        idx = parts.index("shorts")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return ""


def read_existing_manifest(manifest_path: str | Path) -> list[dict]:
    path = Path(manifest_path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def existing_success_indexes(rows: list[dict]) -> tuple[set[str], set[str], set[str]]:
    video_ids = set()
    sha256_values = set()
    source_urls = set()
    for row in rows:
        if row.get("status") not in {"downloaded", "skipped"}:
            continue
        if row.get("video_id"):
            video_ids.add(row["video_id"])
        if row.get("sha256"):
            sha256_values.add(row["sha256"])
        if row.get("source_url"):
            source_urls.add(row["source_url"])
    return video_ids, sha256_values, source_urls


def skipped_row(record: dict, reason: str, existing_file: Path | None = None) -> dict:
    filename = existing_file.name if existing_file else ""
    sha = file_sha256(existing_file) if existing_file and existing_file.exists() else ""
    return {
        "video_id": record.get("video_id") or infer_youtube_id(record.get("source_url", "")),
        "source_url": record.get("source_url", ""),
        "filename": filename,
        "storage_path": str(existing_file) if existing_file else "",
        "sha256": sha,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "label": normalize_label(record.get("label", "")),
        "status": "skipped",
        "error_message": reason,
        "source_type": record.get("source_type", "youtube"),
    }


def find_existing_video(record: dict, output_dir: str | Path) -> Path | None:
    output_dir = Path(output_dir)
    candidates = []
    if record.get("filename"):
        candidates.append(output_dir / record["filename"])
    video_id = record.get("video_id") or infer_youtube_id(record.get("source_url", ""))
    if video_id:
        candidates.extend(output_dir.glob(f"{video_id}.*"))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def download_youtube_video(url: str, output_dir: str | Path = BRONZE_VIDEOS_DIR) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(id)s.%(ext)s")

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--restrict-filenames",
        "-f",
        "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "--print-json",
        "-o",
        output_template,
        url,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    info = json.loads(result.stdout.strip().splitlines()[-1])
    filename = info.get("_filename") or str(output_dir / f"{info['id']}.mp4")
    path = Path(filename)
    if path.suffix.lower() != ".mp4":
        path = path.with_suffix(".mp4")

    return {
        "video_id": info.get("id"),
        "title": info.get("title"),
        "source_url": url,
        "filename": path.name,
        "storage_path": str(path),
        "duration": info.get("duration"),
        "uploader": info.get("uploader"),
        "webpage_url": info.get("webpage_url", url),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "sha256": file_sha256(path) if path.exists() else None,
        "status": "downloaded" if path.exists() else "failed",
        "error_message": "" if path.exists() else "Downloaded file was not found after yt-dlp finished.",
        "source_type": "youtube",
    }


def append_manifest(rows: list[dict], manifest_path: str | Path) -> Path:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(BRONZE_MANIFEST_COLUMNS) + ["title", "duration", "uploader", "webpage_url"]
    exists = manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return manifest_path


def ingest_links(
    links: list[str],
    output_dir: str | Path = BRONZE_VIDEOS_DIR,
    manifest_path: str | Path = BRONZE_MANIFEST_PATH,
    label: str | None = None,
) -> list[dict]:
    ensure_data_dirs()
    records = [{"source_url": url, "label": label or "", "source_type": "youtube"} for url in links]
    return ingest_records(records, output_dir=output_dir, manifest_path=manifest_path)


def ingest_records(
    records: list[dict],
    output_dir: str | Path = BRONZE_VIDEOS_DIR,
    manifest_path: str | Path = BRONZE_MANIFEST_PATH,
) -> list[dict]:
    ensure_data_dirs()
    existing_manifest = read_existing_manifest(manifest_path)
    existing_video_ids, existing_sha256, existing_source_urls = existing_success_indexes(existing_manifest)
    rows = []
    for record in records:
        url = record["source_url"]
        record = {**record, "label": normalize_label(record.get("label", ""))}
        inferred_video_id = record.get("video_id") or infer_youtube_id(url)
        existing_file = find_existing_video({**record, "video_id": inferred_video_id}, output_dir)
        if url in existing_source_urls:
            rows.append(skipped_row({**record, "video_id": inferred_video_id}, "source_url already present in manifest", existing_file))
            continue
        if inferred_video_id and inferred_video_id in existing_video_ids:
            rows.append(skipped_row({**record, "video_id": inferred_video_id}, "video_id already present in manifest", existing_file))
            continue
        if existing_file is not None:
            rows.append(skipped_row({**record, "video_id": inferred_video_id}, "video file already exists in Bronze", existing_file))
            continue

        logger.info("Downloading %s", url)
        try:
            row = download_youtube_video(url, output_dir=output_dir)
        except Exception as exc:
            row = {
                "video_id": inferred_video_id,
                "source_url": url,
                "filename": "",
                "storage_path": "",
                "sha256": "",
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
                "error_message": str(exc),
                "source_type": record.get("source_type", "youtube"),
            }
        if row.get("status") == "downloaded" and row.get("sha256") in existing_sha256:
            row["status"] = "skipped"
            row["error_message"] = "sha256 already present in manifest"
        row["label"] = record.get("label", "")
        row["source_type"] = record.get("source_type", row.get("source_type", "youtube"))
        rows.append(row)
    append_manifest(rows, manifest_path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest YouTube videos into the local Bronze layer.")
    parser.add_argument("links", nargs="*", help="YouTube URLs.")
    parser.add_argument("--links-file", type=Path, help="Text file with one URL per line.")
    parser.add_argument("--source-csv", type=Path, help="CSV dataset with a YouTube URL column.")
    parser.add_argument("--url-column", default="Media", help="Column containing YouTube links in --source-csv.")
    parser.add_argument("--label-column", default="Video Ground Truth", help="Column containing labels in --source-csv.")
    parser.add_argument("--output-dir", type=Path, default=BRONZE_VIDEOS_DIR)
    parser.add_argument("--manifest", type=Path, default=BRONZE_MANIFEST_PATH)
    parser.add_argument("--label", choices=["Real", "Fake"], help="Optional label for supervised ingestion.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    records = read_source_csv(args.source_csv, args.url_column, args.label_column) if args.source_csv else []
    links = read_links(args.links, args.links_file)
    records.extend({"source_url": link, "label": args.label or "", "source_type": "youtube"} for link in links)
    if not records:
        parser.error("Provide at least one link, --links-file, or --source-csv.")
    ingest_records(records, output_dir=args.output_dir, manifest_path=args.manifest)


if __name__ == "__main__":
    main()
