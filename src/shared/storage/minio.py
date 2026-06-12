from __future__ import annotations

from pathlib import Path


class MinIOStorageClient:


    def __init__(self, endpoint: str, bucket: str, access_key: str, secret_key: str, secure: bool = False) -> None:
        try:
            from minio import Minio
        except ImportError as exc:
            raise ImportError("Install minio to use MinIOStorageClient.") from exc

        self.bucket = bucket
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

    def exists(self, path: str | Path) -> bool:
        try:
            self.client.stat_object(self.bucket, str(path))
            return True
        except Exception:
            return False

    def list(self, prefix: str | Path) -> list[Path]:
        objects = self.client.list_objects(self.bucket, prefix=str(prefix), recursive=True)
        return [Path(item.object_name) for item in objects]

    def uri(self, path: str | Path) -> str:
        return f"s3://{self.bucket}/{path}"

