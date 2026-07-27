from scripts.audit_release_boundary import audit

def test_raw_provider_data_is_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr('scripts.audit_release_boundary.ROOT',tmp_path)
    path=tmp_path/'data/external/provider/raw/match.json'
    path.parent.mkdir(parents=True);path.write_text('x')
    assert audit(['data/external/provider/raw/match.json'])

def test_checkpoint_is_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr('scripts.audit_release_boundary.ROOT',tmp_path)
    path=tmp_path/'reports/acceptance/run.checkpoint.json'
    path.parent.mkdir(parents=True);path.write_text('{}')
    assert audit(['reports/acceptance/run.checkpoint.json'])
