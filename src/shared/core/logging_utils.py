from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.shared.core.io_utils import sanitize_for_json, write_json
from src.shared.core.paths import pipeline_log_path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(sanitize_for_json(payload), ensure_ascii=False, allow_nan=False) + "\n")
    return path


def log_pipeline_event(
    event: str,
    stage: str,
    status: str,
    run_id: str | None = None,
    details: dict[str, Any] | None = None,
    log_path: str | Path | None = None,
) -> Path:
    payload = {
        "event": event,
        "stage": stage,
        "status": status,
        "run_id": run_id,
        "timestamp": utc_now_iso(),
        "details": details or {},
    }
    return append_jsonl(log_path or pipeline_log_path(run_id=run_id), payload)


def write_stage_snapshot(path: str | Path, payload: dict[str, Any]) -> Path:
    return write_json(payload, path)

