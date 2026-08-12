"""Video preprocessing and metadata extraction."""

from .metadata import extract_face_metadata, metadata_is_current, process_catalog

__all__ = ["extract_face_metadata", "metadata_is_current", "process_catalog"]
