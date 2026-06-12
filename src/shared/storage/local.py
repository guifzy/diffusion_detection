from __future__ import annotations

from pathlib import Path


class LocalStorageClient:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).resolve() if root else None

    def _resolve(self, path: str | Path) -> Path:
        path = Path(path)
        if path.is_absolute() or self.root is None:
            return path
        return self.root / path

    def exists(self, path: str | Path) -> bool:
        return self._resolve(path).exists()

    def list(self, prefix: str | Path) -> list[Path]:
        root = self._resolve(prefix)
        if not root.exists():
            return []
        if root.is_file():
            return [root]
        return sorted(path for path in root.rglob("*") if path.is_file())

    def uri(self, path: str | Path) -> str:
        return str(self._resolve(path))

