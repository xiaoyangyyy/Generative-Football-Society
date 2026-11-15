import tomllib
from pathlib import Path

from gfs_core.kernel import load_manifests, resolve_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = PROJECT_ROOT.parent / "gfs-core"


def all_manifest_metadata():
    core = load_manifests(CORE_ROOT / "src" / "gfs_core" / "plugins")
    studio = load_manifests(PROJECT_ROOT / "src" / "gfs_studio" / "plugins")
    return core + studio


def test_combined_manifests_have_closed_acyclic_dependencies():
    metadata = all_manifest_metadata()
    assert len(metadata) == 69
    resolved = resolve_metadata(metadata)
    assert {item.id for item in resolved} == {item.id for item in metadata}


def test_every_declared_contract_module_exists():
    for item in all_manifest_metadata():
        root = CORE_ROOT if item.id.startswith("gfs.core") else PROJECT_ROOT
        contract = root / "src" / Path(*item.contract.split(".")).with_suffix(".py")
        assert contract.is_file(), item.contract


def test_profiles_resolve_their_complete_plugin_closure():
    known = {item.id for item in all_manifest_metadata()}
    for profile in (PROJECT_ROOT / "profiles").glob("*.toml"):
        with profile.open("rb") as handle:
            selected = tomllib.load(handle)["profile"]["plugins"]
        assert set(selected) <= known
        resolved = resolve_metadata(all_manifest_metadata(), selected)
        assert set(selected) <= {item.id for item in resolved}
