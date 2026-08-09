"""Deterministic parsing and CycloneDX projection for hashed Python locks."""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import Any


PACKAGE = re.compile(r"(?m)^([A-Za-z0-9_.-]+)==([^ \\\r\n]+)")
HASH = re.compile(r"--hash=sha256:([0-9a-f]{64})")
SOURCE = re.compile(r"# from (https?://[^\s]+)")


def canonical_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).casefold()


def parse_hashed_lock(text: str) -> list[dict[str, Any]]:
    """Return exact packages with their complete recorded distribution hashes."""
    matches = list(PACKAGE.finditer(text))
    packages: list[dict[str, Any]] = []
    observed: set[str] = set()
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start():end]
        name = canonical_package_name(match.group(1))
        if name in observed:
            raise ValueError(f"duplicate locked package: {name}")
        observed.add(name)
        hashes = sorted(set(HASH.findall(block)))
        if not hashes:
            raise ValueError(f"locked package has no SHA-256: {name}")
        packages.append({
            "name": name,
            "version": match.group(2),
            "sha256": hashes,
            "sources": sorted(set(SOURCE.findall(block))),
        })
    if not packages:
        raise ValueError("lock contains no exact packages")
    return packages


def lock_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_cyclonedx_sbom(
    lock_path: Path, *, project_name: str, project_version: str,
    resolution_target: str = "CPython 3.12 / x86_64 manylinux_2_28 / CPU",
) -> dict[str, Any]:
    """Build a stable CycloneDX 1.6 runtime-closure SBOM from a hashed lock."""
    lock_text = lock_path.read_text(encoding="utf-8")
    packages = parse_hashed_lock(lock_text)
    lock_digest = lock_sha256(lock_path)
    root_ref = f"pkg:pypi/{project_name}@{project_version}"
    components: list[dict[str, Any]] = []
    component_refs: list[str] = []
    for package in packages:
        ref = f"pkg:pypi/{package['name']}@{package['version']}"
        component_refs.append(ref)
        component: dict[str, Any] = {
            "type": "library",
            "bom-ref": ref,
            "name": package["name"],
            "version": package["version"],
            "purl": ref,
            "properties": [
                {
                    "name": "gfs:locked-distribution-sha256-count",
                    "value": str(len(package["sha256"])),
                },
                {
                    "name": "gfs:resolution-target",
                    "value": resolution_target,
                },
            ],
        }
        if package["sources"]:
            component["externalReferences"] = [
                {"type": "distribution", "url": source}
                for source in package["sources"]
            ]
        components.append(component)
    serial = uuid.uuid5(uuid.NAMESPACE_URL, f"gfs-sbom:{lock_digest}")
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": f"urn:uuid:{serial}",
        "version": 1,
        "metadata": {
            "tools": {
                "components": [
                    {"type": "application", "name": "uv", "version": "0.11.13"},
                    {
                        "type": "application",
                        "name": "gfs-supply-chain-builder",
                        "version": "1",
                    },
                ]
            },
            "component": {
                "type": "application",
                "bom-ref": root_ref,
                "name": project_name,
                "version": project_version,
                "purl": root_ref,
                "licenses": [{"license": {"id": "MIT"}}],
                "properties": [
                    {"name": "gfs:lock-file", "value": lock_path.name},
                    {"name": "gfs:lock-sha256", "value": lock_digest},
                    {
                        "name": "gfs:dependency-edge-scope",
                        "value": "root-to-resolved-runtime-closure",
                    },
                ],
            },
        },
        "components": components,
        "dependencies": [
            {"ref": root_ref, "dependsOn": component_refs},
            *({"ref": ref, "dependsOn": []} for ref in component_refs),
        ],
    }
