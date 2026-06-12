from __future__ import annotations

from pathlib import Path
from typing import Protocol


class StorageClient(Protocol):
    """Minimal storage boundary used by data pipeline code.

    Local files remain the MVP implementation. MinIO/S3 can implement this
    protocol later without changing contracts, validation or pipeline stages.
    """

    def exists(self, path: str | Path) -> bool:
        ...

    def list(self, prefix: str | Path) -> list[Path]:
        ...

    def uri(self, path: str | Path) -> str:
        ...

