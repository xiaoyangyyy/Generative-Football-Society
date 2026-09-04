"""Deterministic, root-confined identities for local Python dependency closures."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

from .integrity import file_sha256


EXPLICIT_FILES_V1 = "explicit_files_v1"
TRANSITIVE_LOCAL_IMPORTS_V1 = "transitive_local_imports_v1"
_SUPPORTED_MODES = {EXPLICIT_FILES_V1, TRANSITIVE_LOCAL_IMPORTS_V1}
_LOCAL_NAMESPACES = frozenset({"src", "scripts"})
_MAX_CLOSURE_FILES = 4096


def _confined_file(root: Path, raw_relative: str | Path) -> tuple[str, Path]:
    relative = Path(str(raw_relative))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"code identity path must be project-relative: {raw_relative}")
    path = (root / relative).resolve()
    try:
        canonical = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"code identity path escapes root: {raw_relative}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"code identity file missing: {raw_relative}")
    return canonical, path


def _module_name(root: Path, path: Path) -> str | None:
    relative = path.relative_to(root)
    if path.suffix != ".py" or not relative.parts:
        return None
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    if not parts or parts[0] not in _LOCAL_NAMESPACES:
        return None
    return ".".join(parts)


def _absolute_import_name(
    node: ast.ImportFrom,
    *,
    current_module: str,
    is_package: bool,
) -> str | None:
    if node.level == 0:
        return node.module
    package = current_module.split(".")
    if not is_package:
        package = package[:-1]
    remove = node.level - 1
    if remove > len(package):
        return None
    if remove:
        package = package[:-remove]
    if node.module:
        package.extend(node.module.split("."))
    return ".".join(package) if package else None


def _imported_modules(root: Path, path: Path) -> set[str]:
    current_module = _module_name(root, path)
    if current_module is None:
        return set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise ValueError(f"cannot parse code identity file: {path}") from exc

    modules: set[str] = set()
    is_package = path.name == "__init__.py"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
            continue
        if isinstance(node, ast.ImportFrom):
            base = _absolute_import_name(
                node, current_module=current_module, is_package=is_package,
            )
            if base:
                modules.add(base)
                modules.update(
                    f"{base}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                )
            continue
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        is_dynamic_import = (
            isinstance(function, ast.Name)
            and function.id in {"__import__", "import_module"}
        ) or (
            isinstance(function, ast.Attribute)
            and function.attr == "import_module"
        )
        first = node.args[0]
        if (
            is_dynamic_import
            and isinstance(first, ast.Constant)
            and isinstance(first.value, str)
            and not first.value.startswith(".")
        ):
            modules.add(first.value)
    return {
        module for module in modules
        if module.split(".", 1)[0] in _LOCAL_NAMESPACES
    }


def _module_files(root: Path, module: str) -> list[Path]:
    relative = Path(*module.split("."))
    candidates = [
        root / relative.with_suffix(".py"),
        root / relative / "__init__.py",
    ]
    return [path.resolve() for path in candidates if path.is_file()]


def _package_initializers(root: Path, path: Path) -> list[Path]:
    relative = path.relative_to(root)
    parents = list(relative.parents)[:-1]
    initializers = []
    for parent in reversed(parents):
        if not parent.parts or parent.parts[0] not in _LOCAL_NAMESPACES:
            continue
        candidate = (root / parent / "__init__.py").resolve()
        if candidate.is_file():
            initializers.append(candidate)
    return initializers


def code_identity_manifest(
    root: str | Path,
    declared_files: Iterable[str | Path],
    *,
    mode: str = EXPLICIT_FILES_V1,
) -> dict[str, str]:
    """Hash declared files, optionally expanding their local import closure.

    Legacy sealed protocols use ``explicit_files_v1``. New protocols may opt in
    to ``transitive_local_imports_v1`` without changing how historical evidence
    is replayed.
    """
    if mode not in _SUPPORTED_MODES:
        raise ValueError(f"unsupported code identity mode: {mode}")
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise ValueError(f"code identity root is not a directory: {root}")

    pending: list[Path] = []
    resolved: dict[str, Path] = {}
    for raw_relative in declared_files:
        canonical, path = _confined_file(root_path, raw_relative)
        resolved[canonical] = path
        if path.suffix == ".py":
            pending.append(path)
    if not resolved:
        raise ValueError("code identity requires at least one declared file")

    if mode == TRANSITIVE_LOCAL_IMPORTS_V1:
        visited: set[str] = set()
        while pending:
            path = pending.pop()
            canonical = path.relative_to(root_path).as_posix()
            if canonical in visited:
                continue
            visited.add(canonical)
            discovered = _package_initializers(root_path, path)
            for module in _imported_modules(root_path, path):
                discovered.extend(_module_files(root_path, module))
            for candidate in discovered:
                try:
                    relative = candidate.relative_to(root_path).as_posix()
                except ValueError as exc:
                    raise ValueError(
                        f"transitive code identity path escapes root: {candidate}"
                    ) from exc
                if relative not in resolved:
                    resolved[relative] = candidate
                    pending.append(candidate)
                    if len(resolved) > _MAX_CLOSURE_FILES:
                        raise ValueError("transitive code identity closure is too large")

    return {
        relative: file_sha256(resolved[relative])
        for relative in sorted(resolved)
    }
