from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.shared.core.io_utils import write_json
from src.shared.core.paths import (
    BRONZE_MANIFEST_PATH,
    BRONZE_VIDEOS_DIR,
    GOLD_DIR,
    METADATA_DIR,
    REPORTS_DIR,
    SILVER_DIR,
    VIDEO_CATALOG_PATH,
    ensure_data_dirs,
    pipeline_report_path,
)
from src.shared.core.version import PIPELINE_VERSION

logger = logging.getLogger(__name__)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_ingest(
    source_csv: str | Path = VIDEO_CATALOG_PATH,
    url_column: str = "Media",
    label_column: str = "Video Ground Truth",
    output_dir: str | Path = BRONZE_VIDEOS_DIR,
    manifest_path: str | Path = BRONZE_MANIFEST_PATH,
    limit: int | None = None,
) -> list[dict]:
    from src.data_engineering.ingestion.youtube import ingest_records, read_source_csv

    records = read_source_csv(source_csv, url_column=url_column, label_column=label_column)
    if limit is not None:
        records = records[:limit]
    if not records:
        logger.warning("No downloadable YouTube rows found in %s using column %s", source_csv, url_column)
        return []
    return ingest_records(records, output_dir=output_dir, manifest_path=manifest_path)


def run_preprocess(
    catalog_path: str | Path = VIDEO_CATALOG_PATH,
    videos_dir: str | Path = BRONZE_VIDEOS_DIR,
    metadata_dir: str | Path = METADATA_DIR,
    max_frames: int | None = None,
    detect_every: int = 1,
    overwrite: bool = False,
    limit: int | None = None,
) -> list[Path]:
    import pandas as pd

    from src.data_engineering.preprocessing.metadata import process_catalog

    if limit is None:
        return process_catalog(
            catalog_path,
            videos_dir,
            metadata_dir=metadata_dir,
            max_frames=max_frames,
            detect_every=detect_every,
            overwrite=overwrite,
        )

    catalog = pd.read_csv(catalog_path).head(limit)
    limited_catalog = Path(REPORTS_DIR) / "_limited_preprocess_catalog.csv"
    limited_catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(limited_catalog, index=False)
    return process_catalog(
        limited_catalog,
        videos_dir,
        metadata_dir=metadata_dir,
        max_frames=max_frames,
        detect_every=detect_every,
        overwrite=overwrite,
    )


def run_gold(
    catalog_path: str | Path = VIDEO_CATALOG_PATH,
    videos_dir: str | Path = BRONZE_VIDEOS_DIR,
    metadata_dir: str | Path = METADATA_DIR,
    groups: str = "abcde",
    max_frames: int | None = None,
    generate_missing_metadata: bool = False,
    overwrite_metadata: bool = False,
    limit: int | None = None,
) -> object:
    from src.data_engineering.datasets.gold import build_gold_dataset

    return build_gold_dataset(
        catalog_path=catalog_path,
        videos_dir=videos_dir,
        metadata_dir=metadata_dir,
        groups=groups,
        max_frames=max_frames,
        generate_missing_metadata=generate_missing_metadata,
        overwrite_metadata=overwrite_metadata,
        limit=limit,
    )


def run_validate(
    report_path: str | Path | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> dict:
    from src.data_engineering.pipeline.quality import build_quality_report

    started_at = started_at or now_iso()
    quality = build_quality_report(
        manifest_path=BRONZE_MANIFEST_PATH,
        metadata_dir=METADATA_DIR,
        silver_dir=SILVER_DIR,
        gold_dir=GOLD_DIR,
    )
    report = {
        "started_at": started_at,
        "finished_at": finished_at or now_iso(),
        "pipeline_version": PIPELINE_VERSION,
        **quality,
    }
    output_path = Path(report_path) if report_path else pipeline_report_path()
    write_json(report, output_path)
    logger.info("Saved pipeline quality report to %s", output_path)
    return report


def run_build(args: argparse.Namespace) -> dict:
    started_at = now_iso()
    ensure_data_dirs()

    if not args.skip_ingest:
        run_ingest(
            source_csv=args.catalog,
            url_column=args.url_column,
            label_column=args.label_column,
            limit=args.limit,
        )
    if not args.skip_preprocess:
        run_preprocess(
            catalog_path=args.catalog,
            videos_dir=args.videos_dir,
            metadata_dir=args.metadata_dir,
            max_frames=args.max_frames,
            detect_every=args.detect_every,
            overwrite=args.overwrite_metadata,
            limit=args.limit,
        )
    if not args.skip_gold:
        run_gold(
            catalog_path=args.catalog,
            videos_dir=args.videos_dir,
            metadata_dir=args.metadata_dir,
            groups=args.groups,
            max_frames=args.max_frames,
            generate_missing_metadata=args.generate_missing_metadata,
            overwrite_metadata=args.overwrite_metadata,
            limit=args.limit,
        )
    return run_validate(report_path=args.report, started_at=started_at, finished_at=now_iso())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Data engineering pipeline entrypoint for local, DVC and Prefect runs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_catalog = argparse.ArgumentParser(add_help=False)
    common_catalog.add_argument("--catalog", type=Path, default=VIDEO_CATALOG_PATH)
    common_catalog.add_argument("--videos-dir", type=Path, default=BRONZE_VIDEOS_DIR)
    common_catalog.add_argument("--metadata-dir", type=Path, default=METADATA_DIR)
    common_catalog.add_argument("--limit", type=int)

    ingest = subparsers.add_parser("ingest", parents=[common_catalog], help="Build/update Bronze videos and manifest.")
    ingest.add_argument("--url-column", default="Media")
    ingest.add_argument("--label-column", default="Video Ground Truth")

    preprocess = subparsers.add_parser("preprocess", parents=[common_catalog], help="Build Silver face metadata.")
    preprocess.add_argument("--max-frames", type=int)
    preprocess.add_argument("--detect-every", type=int, default=1)
    preprocess.add_argument("--overwrite", action="store_true")

    gold = subparsers.add_parser("gold", parents=[common_catalog], help="Build Silver features and Gold dataset.")
    gold.add_argument("--groups", default="abcde")
    gold.add_argument("--max-frames", type=int)
    gold.add_argument("--generate-missing-metadata", action="store_true")
    gold.add_argument("--overwrite-metadata", action="store_true")

    validate = subparsers.add_parser("validate", help="Validate contracts and write a quality report.")
    validate.add_argument("--report", type=Path)

    build = subparsers.add_parser("build", parents=[common_catalog], help="Run ingestion, preprocessing, Gold and validation.")
    build.add_argument("--url-column", default="Media")
    build.add_argument("--label-column", default="Video Ground Truth")
    build.add_argument("--groups", default="abcde")
    build.add_argument("--max-frames", type=int)
    build.add_argument("--detect-every", type=int, default=1)
    build.add_argument("--generate-missing-metadata", action="store_true")
    build.add_argument("--overwrite-metadata", action="store_true")
    build.add_argument("--skip-ingest", action="store_true")
    build.add_argument("--skip-preprocess", action="store_true")
    build.add_argument("--skip-gold", action="store_true")
    build.add_argument("--report", type=Path)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ensure_data_dirs()

    if args.command == "ingest":
        rows = run_ingest(
            args.catalog,
            args.url_column,
            args.label_column,
            output_dir=args.videos_dir,
            limit=args.limit,
        )
        logger.info("Ingestion finished with %s manifest rows appended.", len(rows))
    elif args.command == "preprocess":
        outputs = run_preprocess(
            args.catalog,
            args.videos_dir,
            args.metadata_dir,
            max_frames=args.max_frames,
            detect_every=args.detect_every,
            overwrite=args.overwrite,
            limit=args.limit,
        )
        logger.info("Preprocessing finished with %s metadata files.", len(outputs))
    elif args.command == "gold":
        dataset = run_gold(
            args.catalog,
            args.videos_dir,
            args.metadata_dir,
            groups=args.groups,
            max_frames=args.max_frames,
            generate_missing_metadata=args.generate_missing_metadata,
            overwrite_metadata=args.overwrite_metadata,
            limit=args.limit,
        )
        logger.info("Gold build finished with %s rows.", len(dataset))
    elif args.command == "validate":
        run_validate(report_path=args.report)
    elif args.command == "build":
        run_build(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
