from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.shared.core.io_utils import write_json
from src.shared.core.logging_utils import log_pipeline_event
from src.shared.core.paths import (
    BRONZE_MANIFEST_PATH,
    BRONZE_VIDEOS_DIR,
    GOLD_DIR,
    METADATA_DIR,
    REPORTS_DIR,
    SILVER_DIR,
    VIDEO_CATALOG_PATH,
    ensure_data_dirs,
    pipeline_latest_report_path,
    pipeline_report_path,
)
from src.shared.core.version import PIPELINE_VERSION

logger = logging.getLogger(__name__)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "sim"}:
        return True
    if normalized in {"0", "false", "no", "n", "nao", "não"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def run_ingest(
    source_csv: str | Path = VIDEO_CATALOG_PATH,
    url_column: str = "link",
    label_column: str = "label",
    output_dir: str | Path = BRONZE_VIDEOS_DIR,
    manifest_path: str | Path = BRONZE_MANIFEST_PATH,
    limit: int | None = None,
    run_id: str | None = None,
) -> list[dict]:
    from src.data_engineering.ingestion.youtube import ingest_records, read_source_csv

    log_pipeline_event("stage_start", "ingest_bronze", "running", run_id)
    records = read_source_csv(source_csv, url_column=url_column, label_column=label_column)
    if limit is not None:
        records = records[:limit]
    if not records:
        logger.warning("No downloadable video rows found in %s using column %s", source_csv, url_column)
        log_pipeline_event("stage_end", "ingest_bronze", "skipped", run_id, {"reason": "no_records"})
        return []
    rows = ingest_records(records, output_dir=output_dir, manifest_path=manifest_path)
    log_pipeline_event("stage_end", "ingest_bronze", "success", run_id, {"rows_appended": len(rows)})
    return rows


def run_preprocess(
    catalog_path: str | Path = BRONZE_MANIFEST_PATH,
    videos_dir: str | Path = BRONZE_VIDEOS_DIR,
    metadata_dir: str | Path = METADATA_DIR,
    max_frames: int | None = None,
    detect_every: int = 1,
    overwrite: bool = False,
    limit: int | None = None,
    face_detector_model: str | Path | None = None,
    face_model: str | Path | None = None,
    segmenter_model: str | Path | None = None,
    max_faces: int = 10,
    face_detection_confidence: float = 0.3,
    face_landmark_confidence: float = 0.3,
    run_id: str | None = None,
) -> list[Path]:
    import pandas as pd

    from src.data_engineering.preprocessing.metadata import process_catalog

    log_pipeline_event("stage_start", "build_silver_metadata", "running", run_id)
    if limit is None:
        outputs = process_catalog(
            catalog_path,
            videos_dir,
            metadata_dir=metadata_dir,
            max_frames=max_frames,
            detect_every=detect_every,
            overwrite=overwrite,
            face_detector_model_path=face_detector_model,
            face_model_path=face_model,
            segmenter_model_path=segmenter_model,
            max_faces=max_faces,
            face_detection_confidence=face_detection_confidence,
            face_landmark_confidence=face_landmark_confidence,
        )
        log_pipeline_event("stage_end", "build_silver_metadata", "success", run_id, {"metadata_files": len(outputs)})
        return outputs

    catalog = pd.read_csv(catalog_path).head(limit)
    limited_catalog = Path(REPORTS_DIR) / "_limited_preprocess_catalog.csv"
    limited_catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(limited_catalog, index=False)
    outputs = process_catalog(
        limited_catalog,
        videos_dir,
        metadata_dir=metadata_dir,
        max_frames=max_frames,
        detect_every=detect_every,
        overwrite=overwrite,
        face_detector_model_path=face_detector_model,
        face_model_path=face_model,
        segmenter_model_path=segmenter_model,
        max_faces=max_faces,
        face_detection_confidence=face_detection_confidence,
        face_landmark_confidence=face_landmark_confidence,
    )
    log_pipeline_event("stage_end", "build_silver_metadata", "success", run_id, {"metadata_files": len(outputs)})
    return outputs


def run_gold(
    catalog_path: str | Path = BRONZE_MANIFEST_PATH,
    videos_dir: str | Path = BRONZE_VIDEOS_DIR,
    metadata_dir: str | Path = METADATA_DIR,
    groups: str = "abcde",
    max_frames: int | None = None,
    generate_missing_metadata: bool = False,
    overwrite_metadata: bool = False,
    limit: int | None = None,
    face_detector_model: str | Path | None = None,
    face_model: str | Path | None = None,
    segmenter_model: str | Path | None = None,
    max_faces: int = 10,
    face_detection_confidence: float = 0.3,
    face_landmark_confidence: float = 0.3,
    run_id: str | None = None,
) -> object:
    from src.data_engineering.datasets.gold import build_gold_dataset

    log_pipeline_event("stage_start", "build_gold_dataset", "running", run_id)
    dataset = build_gold_dataset(
        catalog_path=catalog_path,
        videos_dir=videos_dir,
        metadata_dir=metadata_dir,
        groups=groups,
        max_frames=max_frames,
        generate_missing_metadata=generate_missing_metadata,
        overwrite_metadata=overwrite_metadata,
        limit=limit,
        face_detector_model_path=face_detector_model,
        face_model_path=face_model,
        segmenter_model_path=segmenter_model,
        max_faces=max_faces,
        face_detection_confidence=face_detection_confidence,
        face_landmark_confidence=face_landmark_confidence,
    )
    log_pipeline_event("stage_end", "build_gold_dataset", "success", run_id, {"rows": len(dataset)})
    return dataset


def run_validate(
    report_path: str | Path | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    include_gx: bool = False,
    fail_on_error: bool = False,
    fail_on_gx_error: bool = False,
    max_missing_feature_ratio: float = 0.5,
    max_fallback_center_ratio: float = 0.35,
    write_artifacts: bool = True,
    run_id: str | None = None,
) -> dict:
    from src.data_engineering.pipeline.quality import (
        blocking_errors_for_report,
        build_quality_report,
        write_quality_artifacts,
    )

    started_at = started_at or now_iso()
    log_pipeline_event("stage_start", "validate_data_contracts", "running", run_id)
    quality = build_quality_report(
        manifest_path=BRONZE_MANIFEST_PATH,
        metadata_dir=METADATA_DIR,
        silver_dir=SILVER_DIR,
        gold_dir=GOLD_DIR,
        include_gx=include_gx,
    )
    report = {
        "started_at": started_at,
        "finished_at": finished_at or now_iso(),
        "pipeline_version": PIPELINE_VERSION,
        **quality,
    }
    report["blocking_errors"] = blocking_errors_for_report(
        report,
        fail_on_gx_error=fail_on_gx_error,
        max_missing_feature_ratio=max_missing_feature_ratio,
        max_fallback_center_ratio=max_fallback_center_ratio,
    )
    report["status"] = "passed" if not report["blocking_errors"] else "failed"
    if write_artifacts:
        report["artifacts"] = write_quality_artifacts(report)
    output_path = Path(report_path) if report_path else pipeline_latest_report_path()
    write_json(report, output_path)
    if report_path is None:
        write_json(report, pipeline_report_path(run_id=run_id))
    logger.info("Saved pipeline quality report to %s", output_path)
    log_pipeline_event(
        "stage_end",
        "validate_data_contracts",
        report["status"],
        run_id,
        {"blocking_errors": report["blocking_errors"], "report_path": str(output_path)},
    )
    if fail_on_error and report["blocking_errors"]:
        raise SystemExit(f"Pipeline validation failed: {', '.join(report['blocking_errors'])}")
    return report


def run_build(args: argparse.Namespace) -> dict:
    started_at = now_iso()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ensure_data_dirs()

    if not args.skip_ingest:
        run_ingest(
            source_csv=args.source_csv,
            url_column=args.url_column,
            label_column=args.label_column,
            manifest_path=args.manifest,
            limit=args.limit,
            run_id=run_id,
        )
    if not args.skip_preprocess:
        run_preprocess(
            catalog_path=args.manifest,
            videos_dir=args.videos_dir,
            metadata_dir=args.metadata_dir,
            max_frames=args.max_frames,
            detect_every=args.detect_every,
            overwrite=args.overwrite_metadata,
            limit=args.limit,
            face_detector_model=args.face_detector_model,
            face_model=args.face_model,
            segmenter_model=args.segmenter_model,
            max_faces=args.max_faces,
            face_detection_confidence=args.face_detection_confidence,
            face_landmark_confidence=args.face_landmark_confidence,
            run_id=run_id,
        )
    if not args.skip_gold:
        gold_overwrite_metadata = args.overwrite_metadata if args.skip_preprocess else False
        run_gold(
            catalog_path=args.manifest,
            videos_dir=args.videos_dir,
            metadata_dir=args.metadata_dir,
            groups=args.groups,
            max_frames=args.max_frames,
            generate_missing_metadata=args.generate_missing_metadata,
            overwrite_metadata=gold_overwrite_metadata,
            limit=args.limit,
            face_detector_model=args.face_detector_model,
            face_model=args.face_model,
            segmenter_model=args.segmenter_model,
            max_faces=args.max_faces,
            face_detection_confidence=args.face_detection_confidence,
            face_landmark_confidence=args.face_landmark_confidence,
            run_id=run_id,
        )
    return run_validate(
        report_path=args.report,
        started_at=started_at,
        finished_at=now_iso(),
        include_gx=args.with_gx,
        fail_on_error=args.fail_on_error,
        fail_on_gx_error=args.fail_on_gx_error,
        max_missing_feature_ratio=args.max_missing_feature_ratio,
        max_fallback_center_ratio=args.max_fallback_center_ratio,
        run_id=run_id,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Data engineering pipeline entrypoint for local, DVC and Prefect runs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    runtime_paths = argparse.ArgumentParser(add_help=False)
    runtime_paths.add_argument("--videos-dir", type=Path, default=BRONZE_VIDEOS_DIR)
    runtime_paths.add_argument("--metadata-dir", type=Path, default=METADATA_DIR)
    runtime_paths.add_argument("--limit", type=int)

    ingest = subparsers.add_parser("ingest", parents=[runtime_paths], help="Build/update Bronze videos and manifest.")
    ingest.add_argument("--source-csv", "--catalog", dest="source_csv", type=Path, default=VIDEO_CATALOG_PATH)
    ingest.add_argument("--manifest", type=Path, default=BRONZE_MANIFEST_PATH)
    ingest.add_argument("--url-column", default="link")
    ingest.add_argument("--label-column", default="label")
    ingest.add_argument("--run-id")

    preprocess = subparsers.add_parser("preprocess", parents=[runtime_paths], help="Build Silver face metadata.")
    preprocess.add_argument("--manifest", "--catalog", dest="manifest", type=Path, default=BRONZE_MANIFEST_PATH)
    preprocess.add_argument("--max-frames", type=int)
    preprocess.add_argument("--detect-every", type=int, default=1)
    preprocess.add_argument("--face-detector-model", type=Path)
    preprocess.add_argument("--face-model", type=Path)
    preprocess.add_argument("--segmenter-model", type=Path)
    preprocess.add_argument("--max-faces", type=int, default=10)
    preprocess.add_argument("--face-detection-confidence", type=float, default=0.3)
    preprocess.add_argument("--face-landmark-confidence", type=float, default=0.3)
    preprocess.add_argument("--overwrite", action="store_true")
    preprocess.add_argument("--run-id")

    gold = subparsers.add_parser("gold", parents=[runtime_paths], help="Build Silver features and Gold dataset.")
    gold.add_argument("--manifest", "--catalog", dest="manifest", type=Path, default=BRONZE_MANIFEST_PATH)
    gold.add_argument("--groups", default="abcde")
    gold.add_argument("--max-frames", type=int)
    gold.add_argument("--generate-missing-metadata", nargs="?", const=True, default=False, type=parse_bool)
    gold.add_argument("--overwrite-metadata", nargs="?", const=True, default=False, type=parse_bool)
    gold.add_argument("--face-detector-model", type=Path)
    gold.add_argument("--face-model", type=Path)
    gold.add_argument("--segmenter-model", type=Path)
    gold.add_argument("--max-faces", type=int, default=10)
    gold.add_argument("--face-detection-confidence", type=float, default=0.3)
    gold.add_argument("--face-landmark-confidence", type=float, default=0.3)
    gold.add_argument("--run-id")

    validate = subparsers.add_parser("validate", help="Validate contracts and write a quality report.")
    validate.add_argument("--report", type=Path, default=pipeline_latest_report_path())
    validate.add_argument("--with-gx", action="store_true", help="Run optional Great Expectations validation.")
    validate.add_argument("--fail-on-error", action="store_true", help="Exit with code 1 for blocking quality errors.")
    validate.add_argument("--fail-on-gx-error", action="store_true", help="Treat GX failures as blocking errors.")
    validate.add_argument("--max-missing-feature-ratio", type=float, default=0.5)
    validate.add_argument("--max-fallback-center-ratio", type=float, default=0.35)
    validate.add_argument("--run-id")

    build = subparsers.add_parser("build", parents=[runtime_paths], help="Run ingestion, preprocessing, Gold and validation.")
    build.add_argument("--source-csv", "--catalog", dest="source_csv", type=Path, default=VIDEO_CATALOG_PATH)
    build.add_argument("--manifest", type=Path, default=BRONZE_MANIFEST_PATH)
    build.add_argument("--url-column", default="link")
    build.add_argument("--label-column", default="label")
    build.add_argument("--groups", default="abcde")
    build.add_argument("--max-frames", type=int)
    build.add_argument("--detect-every", type=int, default=1)
    build.add_argument("--face-detector-model", type=Path)
    build.add_argument("--face-model", type=Path)
    build.add_argument("--segmenter-model", type=Path)
    build.add_argument("--max-faces", type=int, default=10)
    build.add_argument("--face-detection-confidence", type=float, default=0.3)
    build.add_argument("--face-landmark-confidence", type=float, default=0.3)
    build.add_argument("--generate-missing-metadata", nargs="?", const=True, default=False, type=parse_bool)
    build.add_argument("--overwrite-metadata", nargs="?", const=True, default=False, type=parse_bool)
    build.add_argument("--skip-ingest", action="store_true")
    build.add_argument("--skip-preprocess", action="store_true")
    build.add_argument("--skip-gold", action="store_true")
    build.add_argument("--report", type=Path)
    build.add_argument("--with-gx", action="store_true")
    build.add_argument("--fail-on-error", action="store_true")
    build.add_argument("--fail-on-gx-error", action="store_true")
    build.add_argument("--max-missing-feature-ratio", type=float, default=0.5)
    build.add_argument("--max-fallback-center-ratio", type=float, default=0.35)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ensure_data_dirs()

    if args.command == "ingest":
        rows = run_ingest(
            args.source_csv,
            args.url_column,
            args.label_column,
            output_dir=args.videos_dir,
            manifest_path=args.manifest,
            limit=args.limit,
            run_id=args.run_id,
        )
        logger.info("Ingestion finished with %s source rows processed.", len(rows))
    elif args.command == "preprocess":
        outputs = run_preprocess(
            args.manifest,
            args.videos_dir,
            args.metadata_dir,
            max_frames=args.max_frames,
            detect_every=args.detect_every,
            overwrite=args.overwrite,
            limit=args.limit,
            face_detector_model=args.face_detector_model,
            face_model=args.face_model,
            segmenter_model=args.segmenter_model,
            max_faces=args.max_faces,
            face_detection_confidence=args.face_detection_confidence,
            face_landmark_confidence=args.face_landmark_confidence,
            run_id=args.run_id,
        )
        logger.info("Preprocessing finished with %s metadata files.", len(outputs))
    elif args.command == "gold":
        dataset = run_gold(
            args.manifest,
            args.videos_dir,
            args.metadata_dir,
            groups=args.groups,
            max_frames=args.max_frames,
            generate_missing_metadata=args.generate_missing_metadata,
            overwrite_metadata=args.overwrite_metadata,
            limit=args.limit,
            face_detector_model=args.face_detector_model,
            face_model=args.face_model,
            segmenter_model=args.segmenter_model,
            max_faces=args.max_faces,
            face_detection_confidence=args.face_detection_confidence,
            face_landmark_confidence=args.face_landmark_confidence,
            run_id=args.run_id,
        )
        logger.info("Gold build finished with %s rows.", len(dataset))
    elif args.command == "validate":
        run_validate(
            report_path=args.report,
            include_gx=args.with_gx,
            fail_on_error=args.fail_on_error,
            fail_on_gx_error=args.fail_on_gx_error,
            max_missing_feature_ratio=args.max_missing_feature_ratio,
            max_fallback_center_ratio=args.max_fallback_center_ratio,
            run_id=args.run_id,
        )
    elif args.command == "build":
        run_build(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
