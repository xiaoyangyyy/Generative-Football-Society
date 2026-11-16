from __future__ import annotations

import hashlib
from pathlib import Path

from .identity import content_identity

_VERIFICATION_SCRIPTS = (
    "build_plugin_supply_chain.py",
    "validate_supply_chain_locks.py",
    "verify_deployment.py",
    "verify_plugin_containers.py",
    "verify_release.py",
)


def _workspace(root: str | Path) -> Path:
    root = Path(root).resolve()
    return root.parent if root.name in {"gfs-core", "gfs-studio"} else root


def _project_files(root: Path, project: str) -> list[Path]:
    base = root / project
    files = [
        base / "pyproject.toml",
        base / "architecture.toml",
        base / "requirements.txt",
        base / "requirements-linux-py312.in",
        base / "requirements-linux-py312.lock",
        base / "supply-chain" / "sbom.cdx.json",
    ]
    for directory, pattern in (
        ("src", "*.py"), ("src", "*.toml"), ("profiles", "*.toml"),
    ):
        files.extend((base / directory).rglob(pattern))
    return files


def _identity(root: Path, files: list[Path]) -> str:
    identities = {}
    for path in sorted(set(files), key=lambda item: str(item)):
        if path.is_file():
            identities[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return content_identity(identities)


def source_identities(root: str | Path) -> dict[str, str]:
    root = _workspace(root)
    core = _project_files(root, "gfs-core")
    studio = _project_files(root, "gfs-studio")
    verification = [root / "scripts" / name for name in _VERIFICATION_SCRIPTS]
    verification.extend((root / "src" / "infrastructure").glob("*.py"))
    for project in ("gfs-core", "gfs-studio"):
        verification.extend((root / project / "tests").rglob("*.py"))
    deployment = [
        root / "gfs-core" / "Dockerfile",
        root / "gfs-studio" / "Dockerfile",
    ]
    deployment.extend((root / "deployment").rglob("*"))
    identities = {
        "core": _identity(root, core),
        "studio": _identity(root, studio),
        "verification": _identity(root, verification),
        "deployment": _identity(root, deployment),
    }
    identities["simulation"] = content_identity({
        "core": identities["core"], "studio": identities["studio"],
    })
    identities["build"] = content_identity({
        key: identities[key]
        for key in ("core", "studio", "verification", "deployment")
    })
    identities["all"] = content_identity(identities)
    return identities


def source_identity(root: str | Path) -> str:
    return source_identities(root)["all"]


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
