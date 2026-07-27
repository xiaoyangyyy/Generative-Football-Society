from scripts.evaluate_large_shadow_v98 import fixtures,paired_bootstrap
from src.simulation.random_control import derive_seed

def test_fixture_design_is_unique_and_mirrored():
    rows=fixtures(['A','B','C','D','E','F'])
    assert len(rows)>=12 and len(rows)==len(set(rows))
    assert all((away,home) in rows for home,away in rows)

def test_cluster_bootstrap_uses_unordered_fixture():
    rows=[{'home':'A','away':'B','delta':{'xg':x}} for x in (1.,2.)]
    rows +=[{'home':'B','away':'A','delta':{'xg':x}} for x in (2.,3.)]
    rows +=[{'home':'C','away':'D','delta':{'xg':0.}}]
    result=paired_bootstrap(rows,'xg',repeats=100,seed=1)
    assert result['clusters']==2 and result['mean']==1.0

def test_default_teams_use_canonical_world_ids():
    from scripts.evaluate_large_shadow_v98 import TEAMS
    assert 'South Korea' in TEAMS and 'United States' in TEAMS

def test_strict_mirror_seed_is_orientation_invariant():
    def seed(home,away):return derive_seed(7,'strict-mirror-shadow',*sorted((home,away)),3)
    assert seed('Brazil','Argentina')==seed('Argentina','Brazil')

def test_model_loaders_cache_inference_assets():
    from pathlib import Path
    from src.match_engine.frame_world.pass_triplet import load_pass_triplet
    path=Path('data/frame_world/frame_world_v85_pass_triplet_lodo_skillcorner.pt')
    assert load_pass_triplet(path)[0] is load_pass_triplet(path)[0]
