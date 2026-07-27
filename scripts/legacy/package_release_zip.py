import argparse
import os
import zipfile
from pathlib import Path


EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    "outputs",
    ".mypy_cache",
    ".pytest_cache",
}
EXCLUDE_FILES = {
    ".env",
}


def should_exclude(path: Path, project_root: Path):
    rel = path.relative_to(project_root)
    parts = set(rel.parts)
    if parts & EXCLUDE_DIRS:
        return True
    if path.name in EXCLUDE_FILES:
        return True
    if path.suffix in {".pyc", ".pyo"}:
        return True
    return False


def build_zip(project_root: Path, output_zip: Path):
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in project_root.rglob("*"):
            if not file_path.is_file():
                continue
            if should_exclude(file_path, project_root):
                continue
            rel = file_path.relative_to(project_root)
            zf.write(file_path, rel.as_posix())
    return output_zip


def main():
    parser = argparse.ArgumentParser(description="Build clean release zip for GFS")
    parser.add_argument(
        "--out",
        default=str(Path.home() / "Desktop" / "Generative_Football_Society_V13_FINAL_Cinderella_release.zip"),
        help="Output zip path",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    out_zip = Path(args.out).resolve()
    built = build_zip(project_root, out_zip)
    print(f"Release zip created: {built}")


if __name__ == "__main__":
    main()
