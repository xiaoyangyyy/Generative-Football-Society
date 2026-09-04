"""Cross-cutting infrastructure with no domain or product dependencies."""

from .locking import FileLease, LeaseUnavailable
from .integrity import (
    file_sha256, portable_text_hash_matches, verify_artifact_manifest,
)
from .code_identity import (
    EXPLICIT_FILES_V1,
    TRANSITIVE_LOCAL_IMPORTS_V1,
    code_identity_manifest,
)
from .atomic_io import fsync_directory

__all__ = [
    "FileLease", "LeaseUnavailable", "file_sha256",
    "portable_text_hash_matches",
    "verify_artifact_manifest",
    "EXPLICIT_FILES_V1",
    "TRANSITIVE_LOCAL_IMPORTS_V1",
    "code_identity_manifest",
    "fsync_directory",
]
