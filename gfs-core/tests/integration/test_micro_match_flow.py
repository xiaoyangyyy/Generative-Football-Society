from pathlib import Path

import pytest

from gfs_core.api import run_micro_match
from gfs_core.contracts.micro import MicroMatchConfig
from gfs_core.contracts.society import TeamProfile
from gfs_core.kernel import PluginKernel, PluginRegistry
from gfs_core.plugins.data.squads.rosters import RostersPlugin
from gfs_core.plugins.match.actions.plugin import ActionsPlugin
from gfs_core.plugins.match.physics.plugin import PhysicsPlugin
from gfs_core.plugins.match.plugin import MatchPlugin
from gfs_core.plugins.match.scoring.plugin import ScoringPlugin
from gfs_core.plugins.match.spatial.plugin import SpatialPlugin
from gfs_core.plugins.society.agents.agent import AgentsPlugin
from gfs_core.plugins.society.psychology.plugin import PsychologyPlugin

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TACTICAL = {
    "build_up_short": 0.5,
    "compactness": 0.5779090358549934,
    "counter_attack": 0.45016600268752216,
    "counterpress": 0.47701620894755165,
    "cross_frequency": 0.45016600268752216,
    "high_press": 0.4905011430016176,
    "line_height": 0.5,
    "long_ball_bias": 0.35434369377420455,
    "low_block": 0.40695558576103513,
    "man_oriented_press": 0.40131233988754805,
    "offside_trap": 0.4319253656434116,
    "overlap_fullbacks": 0.5224848247918,
    "possession_orientation": 0.49550012149606365,
    "pressing_intensity": 0.5299640517645717,
    "risk_budget": 0.5,
    "rotation_aggressiveness": 0.5,
    "target_man": 0.40131233988754805,
    "tempo": 0.5359379207244089,
    "through_ball_bias": 0.45016600268752216,
    "verticality": 0.5224848247918,
    "width_play": 0.5423772818736944,
    "wing_focus": 0.5473318407966955,
}


def profile(team):
    return TeamProfile(
        team_name=team,
        status_score=90,
        formation="4-2-3-1",
        tactical=TACTICAL,
        emotion_profile={
            "pride": 0.3, "anger": 0.1, "fear": 0.2, "determination": 0.4,
        },
        psychology_profile={"confidence": 0.7},
        fatigue=0.2,
        media_exposure=0.6,
        conflict_heat=0.1,
        team_cohesion=0.8,
        coach_name="Coach",
    )


def test_full_micro_match_has_stable_metrics_and_substitutions():
    config = MicroMatchConfig.fast_demo()
    kernel = PluginKernel(PluginRegistry([
        RostersPlugin(PROJECT_ROOT),
        SpatialPlugin(),
        AgentsPlugin(),
        PhysicsPlugin(),
        ActionsPlugin(config, base_dir=PROJECT_ROOT),
        PsychologyPlugin(),
        ScoringPlugin(),
        MatchPlugin(),
    ])).start(["gfs.core.match"])
    summary = kernel.context.require("match-runner").run(
        profile("Brazil"),
        profile("Argentina"),
        goals_home=0,
        goals_away=3,
        xg_home=0.03,
        xg_away=0.81,
        referee={"strictness": 0.58, "bias_t1": 0.04},
        config=config,
        seed=42,
    )
    assert summary.ticks == 180
    assert (summary.goals_micro_home, summary.goals_micro_away) == (0, 3)
    assert summary.micro_xg_home == pytest.approx(0.08390000508428358)
    assert summary.micro_xg_away == pytest.approx(0.7390550519475726)
    assert summary.possession_home == pytest.approx(0.6555555555555556)
    assert (summary.passes_home, summary.passes_away) == (117, 61)
    assert summary.pass_completion_home == pytest.approx(0.8290598290598291)
    assert summary.pass_completion_away == pytest.approx(0.6885245901639344)
    assert (summary.shots_home, summary.shots_away) == (0, 4)
    assert (summary.shots_on_target_home, summary.shots_on_target_away) == (0, 2)
    assert (summary.fouls_committed_home, summary.fouls_committed_away) == (12, 16)
    assert (summary.tackles_home, summary.tackles_away) == (4, 5)
    assert summary.crosses_attempted == 18
    assert summary.lambda_home == pytest.approx(1.0436741317086835)
    assert summary.lambda_away == pytest.approx(1.4285317517228804)
    assert summary.final_psi == pytest.approx(-0.11727095364131597)
    assert summary.ref_strictness_mean == pytest.approx(0.5226694849246147)
    assert summary.phi_integral_home == pytest.approx(0.4877621045633133)
    assert summary.phi_integral_away == pytest.approx(0.4834577715505998)
    assert summary.substitutions == [
        {"team_id": "Brazil", "off": "tm_29241", "on": "tm_401530", "minute": 57.5},
        {"team_id": "Argentina", "off": "tm_576024", "on": "tm_206050", "minute": 57.5},
        {"team_id": "Brazil", "off": "tm_79960", "on": "tm_80562", "minute": 69.5},
        {"team_id": "Argentina", "off": "tm_406625", "on": "tm_486031", "minute": 69.5},
        {"team_id": "Brazil", "off": "tm_68290", "on": "tm_378710", "minute": 81.5},
        {"team_id": "Argentina", "off": "tm_255901", "on": "tm_166237", "minute": 81.5},
    ]
    kernel.stop()


def test_default_micro_api_runs_end_to_end():
    summary = run_micro_match(
        "Brazil", "Argentina", project_root=PROJECT_ROOT, fast=True, seed=42,
    )
    assert (summary.goals_micro_home, summary.goals_micro_away) == (0, 3)
    assert summary.ticks == 180
    assert summary.micro_xg_home == pytest.approx(0.04265602496027286)
    assert summary.micro_xg_away == pytest.approx(0.8175550324510238)
    assert (summary.passes_home, summary.passes_away) == (105, 72)
    assert (summary.shots_home, summary.shots_away) == (0, 4)
