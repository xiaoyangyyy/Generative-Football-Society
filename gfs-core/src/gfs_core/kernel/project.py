from __future__ import annotations

import hashlib
from pathlib import Path

from .identity import content_identity


def source_identity(root: str | Path) -> str:
    root = Path(root).resolve()
    if root.name in {"gfs-core", "gfs-studio"}:
        root = root.parent
    files = []
    for project in ("gfs-core", "gfs-studio"):
        files.append(root / project / "pyproject.toml")
        files.append(root / project / "architecture.toml")
        files.extend((root / project / "src").rglob("*.py"))
        files.extend((root / project / "profiles").glob("*.toml"))
        files.append(root / project / "Dockerfile")
    files.extend((root / "scripts").glob("*.py"))
    files.extend((root / "deployment").rglob("*"))
    identities = {}
    for path in sorted(files, key=lambda item: str(item)):
        if path.is_file():
            identities[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return content_identity(identities)


def data_identity(root: str | Path) -> str:
    root = Path(root).resolve()
    if root.name == "gfs-studio":
        root = root.parent / "gfs-core"
    elif (root / "gfs-core").is_dir():
        root = root / "gfs-core"
    allowed_suffixes = {".csv", ".json", ".jsonl", ".toml"}
    excluded = {"persistence", "training", "backups"}
    identities = {}
    data_root = root / "data"
    if not data_root.is_dir():
        return content_identity({})
    for path in sorted(data_root.rglob("*"), key=lambda item: str(item)):
        if (
            path.is_file()
            and path.suffix.lower() in allowed_suffixes
            and not excluded.intersection(path.relative_to(data_root).parts)
        ):
            identities[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return content_identity(identities)
