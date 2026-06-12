"""Storage adapters for local development and future MinIO/S3 execution."""

from .base import StorageClient
from .local import LocalStorageClient

__all__ = ["StorageClient", "LocalStorageClient"]

