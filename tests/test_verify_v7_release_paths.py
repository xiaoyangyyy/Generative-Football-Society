from scripts.verify_v7_release import ROOT, hash_matches, portable_artifact_path


def test_legacy_windows_registry_path_is_rebased_to_current_root():
    path = portable_artifact_path(
        r"D:\eeg2street\data\world_model\joint_pass_v7_candidate.json"
    )
    assert path == ROOT / "data/world_model/joint_pass_v7_candidate.json"
    assert path.is_file()


def test_existing_path_is_preserved(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"model")
    assert portable_artifact_path(str(artifact)) == artifact


def test_hash_matches_only_reversible_line_endings(tmp_path):
    import hashlib

    path = tmp_path / "text.txt"
    path.write_bytes(b"a\r\nb\r\n")
    expected = hashlib.sha256(b"a\nb\n").hexdigest()
    assert hash_matches(path, expected)
    path.write_bytes(b"a\r\nchanged\r\n")
    assert not hash_matches(path, expected)
