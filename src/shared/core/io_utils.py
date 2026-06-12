from __future__ import annotations

import logging
import math
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def write_dataframe(df, output_path: str | Path, index: bool = False) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=index)
            return path
        except ImportError:
            fallback = path.with_suffix(".csv")
            logger.warning("Parquet engine not installed; writing CSV fallback to %s", fallback)
            df.to_csv(fallback, index=index)
            return fallback

    if path.suffix == ".csv":
        df.to_csv(path, index=index)
        return path

    df.to_csv(path.with_suffix(".csv"), index=index)
    return path.with_suffix(".csv")


def read_dataframe(path: str | Path) -> pd.DataFrame:
    import pandas as pd

    path = Path(path)
    if not path.exists() and path.suffix == ".parquet" and path.with_suffix(".csv").exists():
        path = path.with_suffix(".csv")
    if not path.exists() and path.suffix == ".csv" and path.with_suffix(".parquet").exists():
        path = path.with_suffix(".parquet")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataframe format: {path}")


def write_json(data: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_for_json(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        import numpy as np

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return None if not math.isfinite(float(value)) else float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
    except ImportError:
        pass
    return value
