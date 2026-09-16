from __future__ import annotations

import argparse
import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data_engineering.ingestion.youtube import file_sha256
from src.shared.core.io_utils import write_dataframe
from src.shared.core.paths import DF26_DATASET_DIR, DF26_LOGOS_SPLITS_PATH, DF26_MANIFEST_PATH, ensure_data_dirs

logger = logging.getLogger(__name__)


OPEN_WEIGHT_GENERATORS = {
    "HunyuanVideo_1.5_14b_i2v",
    "HunyuanVideo_1.5_14b_t2v",
    "LTX_2.3_distilled_i2v",
    "LTX_2.3_distilled_t2v",
    "Wan_2.2_14b_i2v",
    "Wan_2.2_14b_t2v",
}

COMMERCIAL_GENERATORS = {
    "Grok_Imagine",
    "Kling_3.0",
    "Veo_3.1",
    "Wan_2.6",
}

GENERATOR_FAMILIES = {
    "HunyuanVideo_1.5_14b_i2v": "HunyuanVideo_1.5",
    "HunyuanVideo_1.5_14b_t2v": "HunyuanVideo_1.5",
    "LTX_2.3_distilled_i2v": "LTX_2.3",
    "LTX_2.3_distilled_t2v": "LTX_2.3",
    "Wan_2.2_14b_i2v": "Wan_2.2",
    "Wan_2.2_14b_t2v": "Wan_2.2",
    "Grok_Imagine": "Grok_Imagine",
    "Kling_3.0": "Kling_3.0",
    "Veo_3.1": "Veo_3.1",
    "Wan_2.6": "Wan_2.6",
}

GENERATOR_ALIASES = {
    "hunyuanvideo1514bi2v": "HunyuanVideo_1.5_14b_i2v",
    "hunyuanvideo15i2v": "HunyuanVideo_1.5_14b_i2v",
    "hunyuanvideo1514bt2v": "HunyuanVideo_1.5_14b_t2v",
    "hunyuanvideo15t2v": "HunyuanVideo_1.5_14b_t2v",
    "ltx23distilledi2v": "LTX_2.3_distilled_i2v",
    "ltx23i2v": "LTX_2.3_distilled_i2v",
    "ltx23distilledt2v": "LTX_2.3_distilled_t2v",
    "ltx23t2v": "LTX_2.3_distilled_t2v",
    "wan2214bi2v": "Wan_2.2_14b_i2v",
    "wan22i2v": "Wan_2.2_14b_i2v",
    "wan2214bt2v": "Wan_2.2_14b_t2v",
    "wan22t2v": "Wan_2.2_14b_t2v",
    "grokimagine": "Grok_Imagine",
    "kling30": "Kling_3.0",
    "veo31": "Veo_3.1",
    "wan26": "Wan_2.6",
}

DEFAULT_ALLOW_PATTERNS = (
    "metadata/*",
    "prompts.csv",
    "real/**",
    "fake/**",
)

PATH_COLUMNS = (
    "path",
    "video_path",
    "relative_path",
    "rel_path",
    "filepath",
    "file_path",
    "filename",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return normalized.strip("_") or "unknown"


def _canonical_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _canonical_generator(generator: str) -> str:
    if not generator:
        return ""
    if generator in GENERATOR_FAMILIES:
        return generator
    return GENERATOR_ALIASES.get(_canonical_key(generator), generator)


def _clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _first_present(row: pd.Series, columns: tuple[str, ...]) -> str:
    for column in columns:
        if column in row.index:
            value = _clean(row[column])
            if value:
                return value
    return ""


def _metadata_path(dataset_dir: str | Path) -> Path:
    return Path(dataset_dir) / "metadata" / "videos.csv"


def _infer_relative_path(row: pd.Series) -> str:
    value = _first_present(row, PATH_COLUMNS)
    if value:
        return value.replace("\\", "/").lstrip("./")

    label = _infer_label(row, "")
    scenario = _first_present(row, ("scenario", "scene", "category"))
    clip_id = _first_present(row, ("clip_id", "clip", "id", "video_id"))
    generator = _first_present(row, ("generator", "model", "generator_name"))
    filename = f"{clip_id}.mp4" if clip_id and not clip_id.endswith(".mp4") else clip_id

    if label == "Real" and scenario and filename:
        return f"real/{scenario}/{filename}"
    if label == "Fake" and scenario and generator and filename:
        return f"fake/{scenario}/{generator}/{filename}"
    return ""


def _parse_path(relative_path: str) -> dict[str, str]:
    parts = [part for part in relative_path.replace("\\", "/").split("/") if part]
    parsed = {
        "df26_label_from_path": "",
        "df26_scenario": "",
        "df26_generator": "",
        "df26_clip_id": "",
    }
    if len(parts) >= 3 and parts[0] == "real":
        parsed["df26_label_from_path"] = "Real"
        parsed["df26_scenario"] = parts[1]
        parsed["df26_clip_id"] = Path(parts[-1]).stem
    elif len(parts) >= 4 and parts[0] == "fake":
        parsed["df26_label_from_path"] = "Fake"
        parsed["df26_scenario"] = parts[1]
        parsed["df26_generator"] = parts[2]
        parsed["df26_clip_id"] = Path(parts[-1]).stem
    return parsed


def _infer_label(row: pd.Series, relative_path: str) -> str:
    path_label = _parse_path(relative_path).get("df26_label_from_path", "") if relative_path else ""
    if path_label:
        return path_label

    if "is_fake" in row.index:
        is_fake = _clean(row["is_fake"]).lower()
        if is_fake in {"1", "true", "yes", "y"}:
            return "Fake"
        if is_fake in {"0", "false", "no", "n"}:
            return "Real"

    raw = _first_present(row, ("label", "target", "class", "is_fake", "split"))
    lowered = raw.lower()
    if lowered in {"real", "0", "false"}:
        return "Real"
    if lowered in {"fake", "synthetic", "generated", "ai", "1", "true", "true_fake"}:
        return "Fake"
    return ""


def _generator_availability(generator: str) -> str:
    generator = _canonical_generator(generator)
    if not generator:
        return "real"
    if generator in OPEN_WEIGHT_GENERATORS:
        return "open_weight"
    if generator in COMMERCIAL_GENERATORS:
        return "commercial"
    return "unknown"


def _training_role(label: str, availability: str) -> str:
    if label == "Real":
        return "real_reference"
    if label == "Fake" and availability == "open_weight":
        return "open_weight_trainable"
    if label == "Fake" and availability == "commercial":
        return "commercial_evaluation_only"
    return "review"


def _video_id(label: str, scenario: str, generator: str, clip_id: str, relative_path: str) -> str:
    if label == "Real":
        return "__".join(["df26", "real", _safe_id(scenario), _safe_id(clip_id)])
    if label == "Fake":
        return "__".join(["df26", "fake", _safe_id(scenario), _safe_id(generator), _safe_id(clip_id)])
    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]
    return f"df26__unknown__{digest}"


def read_df26_metadata(dataset_dir: str | Path) -> pd.DataFrame:
    path = _metadata_path(dataset_dir)
    if path.exists():
        return pd.read_csv(path)

    rows = []
    for video_path in sorted(Path(dataset_dir).glob("real/**/*.mp4")) + sorted(Path(dataset_dir).glob("fake/**/*.mp4")):
        rows.append({"path": video_path.relative_to(dataset_dir).as_posix()})
    if rows:
        logger.warning("metadata/videos.csv not found; built DF26 inventory by scanning local video files.")
        return pd.DataFrame(rows)
    raise FileNotFoundError(f"DF26 metadata not found at {path} and no local videos were found.")


def load_generator_inventory(dataset_dir: str | Path) -> dict[str, str]:
    path = Path(dataset_dir) / "metadata" / "generators.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    inventory: dict[str, str] = {}
    for _, row in df.iterrows():
        generator = _first_present(row, ("generator", "name", "generator_name", "model"))
        if not generator:
            continue
        availability = _first_present(row, ("availability", "source", "type", "license_role")).lower()
        if "open" in availability:
            inventory[generator] = "open_weight"
        elif "commercial" in availability or "closed" in availability:
            inventory[generator] = "commercial"
    return inventory


def build_df26_manifest(
    dataset_dir: str | Path = DF26_DATASET_DIR,
    output_path: str | Path = DF26_MANIFEST_PATH,
    splits_path: str | Path = DF26_LOGOS_SPLITS_PATH,
    repo_id: str = "DF26/DF26",
    compute_sha256: bool = False,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_data_dirs()
    dataset_dir = Path(dataset_dir)
    metadata = read_df26_metadata(dataset_dir)
    generator_inventory = load_generator_inventory(dataset_dir)
    processed_at = now_iso()

    rows = []
    for _, row in metadata.iterrows():
        relative_path = _infer_relative_path(row)
        parsed = _parse_path(relative_path)
        label = _infer_label(row, relative_path)
        scenario = _first_present(row, ("scenario", "scene", "category")) or parsed["df26_scenario"]
        generator = _first_present(row, ("generator", "model", "generator_name")) or parsed["df26_generator"]
        canonical_generator = _canonical_generator(generator)
        clip_id = _first_present(row, ("clip_id", "clip", "id")) or parsed["df26_clip_id"]
        generator_family = GENERATOR_FAMILIES.get(canonical_generator, canonical_generator or "real")
        availability = generator_inventory.get(generator) or generator_inventory.get(canonical_generator)
        availability = availability or _generator_availability(canonical_generator)
        training_role = _training_role(label, availability)
        storage_path = dataset_dir / relative_path if relative_path else Path("")
        exists = storage_path.exists() if relative_path else False
        status = "downloaded" if exists else "failed"
        error = "" if exists else "DF26 video file was not found locally."
        video_id = _video_id(label, scenario, generator, clip_id, relative_path)

        rows.append(
            {
                "video_id": video_id,
                "source_url": f"hf://datasets/{repo_id}/{relative_path}" if relative_path else f"hf://datasets/{repo_id}",
                "filename": Path(relative_path).name,
                "storage_path": str(storage_path) if relative_path else "",
                "sha256": file_sha256(storage_path) if compute_sha256 and exists else "",
                "downloaded_at": processed_at,
                "label": label,
                "status": status,
                "error_message": error,
                "source_type": "dataset_local",
                "benchmark_dataset": "DF26",
                "df26_relative_path": relative_path,
                "df26_clip_id": clip_id,
                "df26_scenario": scenario,
                "df26_generator": generator,
                "df26_generator_canonical": canonical_generator,
                "df26_generator_family": generator_family,
                "df26_generator_availability": availability,
                "df26_training_role": training_role,
                "df26_allowed_for_training": label == "Real" or training_role == "open_weight_trainable",
                "df26_allowed_for_evaluation": status == "downloaded",
                "df26_license_note": (
                    "Open-weight fakes may be used for detector training only under LOGOS; "
                    "commercial fakes are evaluation-only."
                ),
            }
        )

    manifest = pd.DataFrame(rows)
    if limit and limit > 0:
        manifest = _balanced_limit(manifest, limit)
    manifest = manifest.sort_values(["label", "df26_scenario", "df26_generator", "df26_clip_id"]).reset_index(drop=True)
    splits = build_logos_splits(manifest)

    write_dataframe(manifest, output_path, index=False)
    write_dataframe(splits, splits_path, index=False)
    logger.info("Saved DF26 Bronze manifest with %s rows to %s", len(manifest), output_path)
    logger.info("Saved DF26 LOGOS split table with %s rows to %s", len(splits), splits_path)
    return manifest, splits


def _balanced_limit(manifest: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(manifest) <= limit:
        return manifest
    groups = ["label", "df26_generator_family"]
    chunks = []
    per_group = max(1, limit // max(1, manifest.groupby(groups, dropna=False).ngroups))
    for _, group in manifest.groupby(groups, dropna=False):
        chunks.append(group.sort_values("video_id").head(per_group))
    limited = pd.concat(chunks, ignore_index=True)
    if len(limited) < limit:
        remaining = manifest[~manifest["video_id"].isin(limited["video_id"])]
        limited = pd.concat([limited, remaining.sort_values("video_id").head(limit - len(limited))], ignore_index=True)
    return limited.head(limit)


def build_logos_splits(manifest: pd.DataFrame) -> pd.DataFrame:
    open_families = sorted(
        family
        for family in manifest.loc[
            manifest["df26_generator_availability"].eq("open_weight"), "df26_generator_family"
        ].dropna().unique()
        if family
    )
    rows = []
    for family in open_families:
        fold = f"logos_leave_{family}"
        for _, row in manifest.iterrows():
            availability = row["df26_generator_availability"]
            label = row["label"]
            generator_family = row["df26_generator_family"]
            if label == "Real":
                split = "train"
            elif availability == "open_weight" and generator_family == family:
                split = "test"
            elif availability == "open_weight":
                split = "train"
            elif availability == "commercial":
                split = "holdout_commercial"
            else:
                split = "review"
            rows.append(
                {
                    "video_id": row["video_id"],
                    "df26_logos_fold": fold,
                    "dataset_split": split,
                    "target_label": row["label"],
                    "df26_clip_id": row["df26_clip_id"],
                    "df26_scenario": row["df26_scenario"],
                    "df26_generator": row["df26_generator"],
                    "df26_generator_canonical": row["df26_generator_canonical"],
                    "df26_generator_family": generator_family,
                    "df26_generator_availability": availability,
                    "df26_training_role": row["df26_training_role"],
                }
            )

    for _, row in manifest.iterrows():
        availability = row["df26_generator_availability"]
        if row["label"] == "Real":
            split = "reference_real"
        elif availability == "commercial":
            split = "test"
        elif availability == "open_weight":
            split = "excluded_open_weight"
        else:
            split = "review"
        rows.append(
            {
                "video_id": row["video_id"],
                "df26_logos_fold": "commercial_final",
                "dataset_split": split,
                "target_label": row["label"],
                "df26_clip_id": row["df26_clip_id"],
                "df26_scenario": row["df26_scenario"],
                "df26_generator": row["df26_generator"],
                "df26_generator_canonical": row["df26_generator_canonical"],
                "df26_generator_family": row["df26_generator_family"],
                "df26_generator_availability": availability,
                "df26_training_role": row["df26_training_role"],
            }
        )
    return pd.DataFrame(rows)


def download_df26_snapshot(
    repo_id: str = "DF26/DF26",
    output_dir: str | Path = DF26_DATASET_DIR,
    revision: str = "main",
    token: str | None = None,
    allow_patterns: tuple[str, ...] = DEFAULT_ALLOW_PATTERNS,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub or run `pip install -r requirements.txt`.") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=output_dir,
        token=token,
        allow_patterns=list(allow_patterns),
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and prepare DF26 for the local Bronze/Silver/Gold pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download DF26 from Hugging Face after access approval.")
    download.add_argument("--repo-id", default="DF26/DF26")
    download.add_argument("--output-dir", type=Path, default=DF26_DATASET_DIR)
    download.add_argument("--revision", default="main")
    download.add_argument("--token", default=None, help="Optional Hugging Face token. Prefer `huggingface-cli login`.")
    download.add_argument("--allow-pattern", action="append", dest="allow_patterns", default=None)

    manifest = subparsers.add_parser("manifest", help="Build the DF26 Bronze manifest and LOGOS split table.")
    manifest.add_argument("--dataset-dir", type=Path, default=DF26_DATASET_DIR)
    manifest.add_argument("--output", type=Path, default=DF26_MANIFEST_PATH)
    manifest.add_argument("--splits-output", type=Path, default=DF26_LOGOS_SPLITS_PATH)
    manifest.add_argument("--repo-id", default="DF26/DF26")
    manifest.add_argument("--compute-sha256", action="store_true")
    manifest.add_argument("--limit", type=int, default=0, help="Optional balanced pilot limit. Use 0 for full DF26.")

    prepare = subparsers.add_parser("prepare", help="Download DF26 and build the Bronze manifest.")
    prepare.add_argument("--repo-id", default="DF26/DF26")
    prepare.add_argument("--dataset-dir", type=Path, default=DF26_DATASET_DIR)
    prepare.add_argument("--manifest-output", type=Path, default=DF26_MANIFEST_PATH)
    prepare.add_argument("--splits-output", type=Path, default=DF26_LOGOS_SPLITS_PATH)
    prepare.add_argument("--revision", default="main")
    prepare.add_argument("--token", default=None)
    prepare.add_argument("--allow-pattern", action="append", dest="allow_patterns", default=None)
    prepare.add_argument("--compute-sha256", action="store_true")
    prepare.add_argument("--limit", type=int, default=0, help="Optional balanced pilot limit. Use 0 for full DF26.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "download":
        output = download_df26_snapshot(
            repo_id=args.repo_id,
            output_dir=args.output_dir,
            revision=args.revision,
            token=args.token,
            allow_patterns=tuple(args.allow_patterns or DEFAULT_ALLOW_PATTERNS),
        )
        logger.info("DF26 downloaded to %s", output)
    elif args.command == "manifest":
        build_df26_manifest(
            dataset_dir=args.dataset_dir,
            output_path=args.output,
            splits_path=args.splits_output,
            repo_id=args.repo_id,
            compute_sha256=args.compute_sha256,
            limit=args.limit or None,
        )
    elif args.command == "prepare":
        download_df26_snapshot(
            repo_id=args.repo_id,
            output_dir=args.dataset_dir,
            revision=args.revision,
            token=args.token,
            allow_patterns=tuple(args.allow_patterns or DEFAULT_ALLOW_PATTERNS),
        )
        build_df26_manifest(
            dataset_dir=args.dataset_dir,
            output_path=args.manifest_output,
            splits_path=args.splits_output,
            repo_id=args.repo_id,
            compute_sha256=args.compute_sha256,
            limit=args.limit or None,
        )


if __name__ == "__main__":
    main()
