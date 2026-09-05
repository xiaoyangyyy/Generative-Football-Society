import os

import numpy as np

from src.simulation.fusion_controller import FusionController
from src.simulation.narrative_events import NarrativeEventBus
from src.simulation.referee_policy import RefereePolicy
from src.simulation.social_dialogue import SocialDialogueEngine
from src.simulation.tournament_checkpoint import new_reflection_journal
from src.simulation.tournament_lifecycle import TournamentLifecycleMixin
from src.simulation.tournament_match import TournamentMatchMixin
from src.simulation.tournament_state import TournamentStateMixin

# Note:
# These 48-team groups follow the 2026 final draw structure, while play-off slots
# are represented by assumed qualifier winners that are already present in this dataset.
# Injury model stays continuous; report prose is tied monotonically to injury_load.
_MEDICAL_CONCERN_SOFT_THRESHOLD = 0.56

WORLD_CUP_2026_GROUPS = {
    "Group A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "Group B": ["Canada", "Switzerland", "Bosnia and Herzegovina", "Qatar"],
    "Group C": ["Brazil", "Scotland", "Morocco", "Haiti"],
    "Group D": ["United States", "Turkey", "Australia", "Paraguay"],
    "Group E": ["Germany", "Ecuador", "Ivory Coast", "Curaçao"],
    "Group F": ["Netherlands", "Sweden", "Japan", "Tunisia"],
    "Group G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "Group H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "Group I": ["France", "Norway", "Senegal", "Iraq"],
    "Group J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "Group K": ["Portugal", "Colombia", "DR Congo", "Uzbekistan"],
    "Group L": ["England", "Croatia", "Ghana", "Panama"]
}

class TournamentManager(
    TournamentLifecycleMixin,
    TournamentStateMixin,
    TournamentMatchMixin,
):
    def __init__(
        self,
        world_engine,
        referee_profile_weights=None,
        referee_stage_morph_strength=1.0,
        referee_profiles=None,
        base_dir=None,
        run_identity_sha256=None,
    ):
        self.world = world_engine
        self.root_seed = int(getattr(world_engine, "root_seed", 42))
        self.run_identity_sha256 = run_identity_sha256
        self.groups = WORLD_CUP_2026_GROUPS
        self.standings = {group: {team: {"pts": 0, "gf": 0, "ga": 0, "gd": 0} for team in teams} for group, teams in self.groups.items()}
        self.qualified_teams = []
        self.qualified_entries = []
        self.r32_fixtures: list[tuple[str, str]] = []
        self.phase = "group"
        self.match_index = 0
        self.completed_matches: list[str] = []
        self.match_results: dict[str, str] = {}
        self.ko_round: str | None = None
        self.ko_fixture_index = 0
        self.post_group_reflection_done = False
        self.reflection_journal = new_reflection_journal()

        self.referee_policy = RefereePolicy(
            profiles=referee_profiles or RefereePolicy().profiles,
            weights=referee_profile_weights or RefereePolicy().weights,
            stage_morph_strength=referee_stage_morph_strength,
        )
        # Compatibility attributes for integrations that inspect manager state.
        self.referee_profiles = self.referee_policy.profiles
        self.referee_profile_weights = self.referee_policy.weights
        self.referee_stage_morph_strength = self.referee_policy.stage_morph_strength
        self.dialogue_engine = SocialDialogueEngine(self.world.feed, turns_per_match=4)
        self.fusion_controller = FusionController()
        self.narrative_event_bus = NarrativeEventBus()
        self.base_dir = os.path.abspath(base_dir or os.path.join(
            os.path.dirname(__file__), "..", "..",
        ))
        self.final_result = {}

    @staticmethod
    def _match_key(stage_name: str, t1: str, t2: str) -> str:
        a, b = sorted([t1, t2])
        return f"{stage_name}::{a}::{b}"

    @staticmethod
    def _map_micro_score(home_micro: str, fixture_t1: str, goals_home: int, goals_away: int) -> tuple[int, int]:
        if home_micro == fixture_t1:
            return int(goals_home), int(goals_away)
        return int(goals_away), int(goals_home)







    def _stage_pressure(self, stage_name, is_knockout):
        # Continuous round-depth pressure, no discrete threshold jumps.
        depth_map = {
            "Group": 0.30,
            "Round of 32": 0.45,
            "Round of 16": 0.60,
            "Quarter-Finals": 0.72,
            "Semi-Finals": 0.86,
            "Final": 1.00,
        }
        depth = 0.30
        if is_knockout:
            depth = depth_map.get(stage_name, 0.72)
        a = 6.0
        b = 0.45
        return float(1.0 / (1.0 + np.exp(-(a * (depth - b)))))

    def _normalize_weights(self, weights, keys):
        from src.simulation.referee_policy import normalize_weights

        return normalize_weights(weights, keys)

    def _stage_referee_distribution(self, stage_pressure):
        return self.referee_policy.stage_distribution(stage_pressure)

    def _sample_referee_profile(
        self, a1, a2, stage_pressure, *, rng=None,
    ):
        return self.referee_policy.sample(
            a1, a2, stage_pressure, rng=rng,
        )
