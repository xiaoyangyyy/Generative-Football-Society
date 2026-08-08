"""Cross-cutting infrastructure with no domain or product dependencies."""

from .locking import FileLease, LeaseUnavailable
from .integrity import (
    file_sha256, portable_text_hash_matches, verify_artifact_manifest,
)

__all__ = [
    "FileLease", "LeaseUnavailable", "file_sha256",
    "portable_text_hash_matches",
    "verify_artifact_manifest",
]
