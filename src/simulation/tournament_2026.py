import itertools
import json
import os
import time
import numpy as np
from src.memory_engine.macro_goal_dynamics import expected_match_xg
from src.memory_engine.macro_micro_fusion import resolve_unified_score
from src.simulation.group_context import build_coach_match_context, snapshot_standings_table
from src.simulation.knockout_bracket import build_qualified_entries, build_r32_pairings
from src.simulation.match_pipeline import (
    finalize_match_feedback,
    micro_layer_enabled,
    micro_physics_score_enabled,
    prepare_match_agents,
    print_micro_match_logs,
    run_extra_time_micro,
    run_micro_layer,
    run_physics_first_micro,
)
from src.simulation.score_path import (
    ScorePathMode,
    finalize_official_score_from_micro,
    resolve_score_path_mode,
    score_path_label,
)
from src.simulation.tournament_checkpoint import (
    checkpoint_root_seed,
    checkpoint_run_identity,
    load_checkpoint,
    new_reflection_journal,
    restore_r32_fixtures,
    save_checkpoint,
    validate_reflection_journal,
)
from src.simulation.world_state import (
    preflight_world_state, restore_world_state, snapshot_world_state,
)
from src.simulation.venue_policy import resolve_match_venue
from src.simulation.tactics_sync import apply_coach_tactics_from_llm
from src.memory_engine.poisson_simulator import (
    simulate_penalty_shootout,
    score_xg_anomaly_note,
    finalize_stage_xg_context,
)
from src.simulation.tactical_matchup import compute_matchup_bonus
from src.simulation.social_dialogue import SocialDialogueEngine
from src.simulation.fusion_controller import FusionController
from src.simulation.narrative_events import NarrativeEventBus
from src.simulation.standings import apply_group_result
from src.simulation.referee_policy import RefereePolicy
from src.simulation.tournament_match import TournamentMatchMixin
from src.simulation.random_control import derive_seed, named_py_rng

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

class TournamentManager(TournamentMatchMixin):
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

    def _reflection_with_retry(
        self, agent, llm, operation_id: str, attempts: int = 6,
    ) -> None:
        if (
            not isinstance(operation_id, str)
            or not operation_id
            or len(operation_id) > 512
        ):
            raise ValueError("Invalid reflection operation ID")
        journal = validate_reflection_journal(self.reflection_journal)
        if operation_id in journal["applied"]:
            return
        receipt = journal["receipts"].get(operation_id)
        if receipt is None:
            for i in range(attempts):
                try:
                    payload = agent.request_reflection_payload(llm)
                    encoded = json.dumps(
                        payload, ensure_ascii=False, allow_nan=False,
                    ).encode("utf-8")
                    if len(encoded) > 256 * 1024:
                        raise ValueError("Reflection response exceeds 256 KiB")
                    break
                except Exception as exc:
                    wait = min(24.0, 2.0 * (2 ** i))
                    print(
                        f"  [REFLECTION] {agent.name} attempt {i + 1}/{attempts} "
                        f"failed: {exc} (retry in {wait:.0f}s)"
                    )
                    if i + 1 < attempts:
                        time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Reflection failed for {agent.name} after {attempts} attempts"
                )
            receipt = {"agent": agent.name, "payload": payload}
            journal["receipts"][operation_id] = receipt
            # Persist the provider result before it can mutate the world.
            self._save_checkpoint()
        if receipt["agent"] != agent.name:
            raise ValueError("Reflection receipt agent identity mismatch")
        agent.apply_reflection_payload(
            receipt["payload"], operation_id=operation_id,
        )
        journal["applied"].append(operation_id)
        self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        self._require_run_identity()
        save_checkpoint(
            self.base_dir,
            standings=self.standings,
            qualified_teams=self.qualified_teams,
            phase=self.phase,
            group_schedule_progress={},
            ko_round=self.ko_round,
            ko_fixture_index=self.ko_fixture_index,
            r32_fixtures=self.r32_fixtures,
            completed_matches=self.completed_matches,
            final_result=self.final_result,
            match_index=self.match_index,
            root_seed=self.root_seed,
            run_identity_sha256=self.run_identity_sha256,
            match_results=self.match_results,
            post_group_reflection_done=self.post_group_reflection_done,
            world_state=snapshot_world_state(self),
            reflection_journal=self.reflection_journal,
        )

    def _restore_from_checkpoint(self, ckpt: dict) -> None:
        self._require_run_identity()
        stored_seed = checkpoint_root_seed(ckpt)
        if stored_seed != self.root_seed:
            raise ValueError(
                "Tournament checkpoint root seed does not match the current world"
            )
        if checkpoint_run_identity(ckpt) != self.run_identity_sha256:
            raise ValueError(
                "Tournament checkpoint run identity does not match the current runtime"
            )
        # Keep the existing manager untouched when live world identities disagree.
        preflight_world_state(self, ckpt["world_state"])
        journal = validate_reflection_journal(ckpt["reflection_journal"])
        self.standings = ckpt.get("standings", self.standings)
        self.qualified_teams = ckpt.get("qualified_teams", [])
        self.phase = ckpt.get("phase", "group")
        self.completed_matches = list(ckpt.get("completed_matches", []))
        self.match_results = dict(ckpt.get("match_results", {}))
        self.ko_round = ckpt.get("ko_round")
        self.ko_fixture_index = int(ckpt.get("ko_fixture_index", 0))
        self.r32_fixtures = restore_r32_fixtures(ckpt)
        self.final_result = ckpt.get("final_result", {})
        self.match_index = int(ckpt.get("match_index", 0))
        self.post_group_reflection_done = bool(ckpt.get("post_group_reflection_done", False))
        self.reflection_journal = journal
        restore_world_state(self, ckpt["world_state"])
        print(
            f"[CHECKPOINT] Resumed phase={self.phase} matches_done={len(self.completed_matches)} "
            f"qualified={len(self.qualified_teams)}"
        )

    def run_full_tournament(self, *, resume: bool = False):
        from src.match_engine.calibration.narrative_isolation import resolve_tournament_llm

        self._require_run_identity()
        llm = resolve_tournament_llm()
        if resume:
            ckpt = load_checkpoint(self.base_dir)
            if ckpt:
                self._restore_from_checkpoint(ckpt)
            else:
                print("[CHECKPOINT] No checkpoint found — starting fresh.")

        if self.phase == "complete":
            print("[CHECKPOINT] Tournament already marked complete.")
            return

        print("[METRICS] conflict_heat scale: 0.00-1.05 (saturation above 1.00 is allowed by design).")
        if self.phase == "group":
            self.simulate_group_stage(llm)
            self.resolve_advancements()
            self.phase = "post_group"
            self._save_checkpoint()

        if not self.post_group_reflection_done:
            print("\n" + "*"*60 + "\n[V13 GLOBAL SUMMIT] Teams performing deep reflection...\n" + "*"*60)
            for t_name in self.qualified_teams:
                self._reflection_with_retry(
                    self.world.agents[t_name], llm,
                    f"post_group:{t_name}",
                )
            self.post_group_reflection_done = True
            self._save_checkpoint()

        self.phase = "knockout"
        for round_name, num_teams in [
            ("Round of 32", 32),
            ("Round of 16", 16),
            ("Quarter-Finals", 8),
            ("Semi-Finals", 4),
            ("Final", 2),
        ]:
            if len(self.qualified_teams) < num_teams:
                continue
            self.ko_round = round_name
            self.simulate_knockout_round(round_name, num_teams, llm)
            self._save_checkpoint()

        self.phase = "complete"
        self._save_checkpoint()

    def _require_run_identity(self) -> None:
        value = self.run_identity_sha256
        if not isinstance(value, str) or len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value
        ):
            raise RuntimeError(
                "Full tournament execution requires a verified run identity"
            )

    def simulate_group_stage(self, llm):
        print("\n" + "="*60 + "\n🚀 PHASE 1: GROUP STAGE (CINDERELLA FIELD ACTIVE)\n" + "="*60)
        from src.simulation.wc2026_schedule import official_group_matchdays, fixture_meta, validate_groups

        validate_groups(self.groups)

        for g_name, teams in self.groups.items():
            matchdays = official_group_matchdays(g_name)
            for md_idx, md_matches in enumerate(matchdays, start=1):
                standings_snapshot = snapshot_standings_table(self.standings, g_name)
                print(f"\n  [GROUP {g_name}] Matchday {md_idx}/3 (parallel kickoff snapshot)")
                for t1, t2 in md_matches:
                    mk = self._match_key(g_name, t1, t2)
                    if mk in self.completed_matches:
                        continue
                    meta = fixture_meta(g_name, t1, t2)
                    if meta is not None:
                        print(
                            f"  [FIXTURE] #{meta.match_number} {meta.date} "
                            f"{t1} vs {t2} @ {meta.venue}, {meta.city}"
                        )
                    fixture_seed = derive_seed(
                        self.root_seed, "group_fixture",
                        g_name, md_idx, t1, t2,
                    )
                    self.play_match(
                        t1,
                        t2,
                        g_name,
                        llm,
                        is_knockout=False,
                        matchday=md_idx,
                        standings_snapshot=standings_snapshot,
                        fixture_seed=fixture_seed,
                        scheduled_home=t1,
                    )

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


    def update_standings(self, g, t1, t2, s1, s2):
        apply_group_result(self.standings[g], t1, t2, int(s1), int(s2))

    def resolve_advancements(self):
        self.qualified_entries = build_qualified_entries(self.groups, self.standings)
        self.qualified_teams = [e.team for e in self.qualified_entries]
        self.r32_fixtures = build_r32_pairings(
            self.qualified_entries,
            rng=named_py_rng(self.root_seed, "round_of_32_pairings"),
        )
        print(f"[ADVANCE] {len(self.qualified_teams)} teams qualified; R32 bracket seeded ({len(self.r32_fixtures)} fixtures).")

    def simulate_knockout_round(self, round_name, num_teams, llm):
        print(f"\n" + "="*60 + f"\n🏆 {round_name.upper()}\n" + "="*60)
        next_round = []
        current_batch = self.qualified_teams[:num_teams]
        if round_name == "Round of 32" and self.r32_fixtures:
            fixtures = list(self.r32_fixtures)
        else:
            fixtures = [
                (current_batch[i], current_batch[i + 1])
                for i in range(0, len(current_batch), 2)
            ]
        for t1, t2 in fixtures:
            mk = self._match_key(round_name, t1, t2)
            if mk in self.completed_matches:
                winner = self.match_results.get(mk)
                if winner:
                    self._record_final_result(
                        round_name, t1=t1, t2=t2, winner=winner,
                    )
                    self._reflection_with_retry(
                        self.world.agents[winner], llm,
                        f"knockout:{mk}:{winner}",
                    )
                    print(f"  [SKIP] {t1} vs {t2} (checkpoint) → {winner}")
                    next_round.append(winner)
                    continue
            fixture_seed = derive_seed(
                self.root_seed, "knockout_fixture", round_name, t1, t2,
            )
            winner = self.play_match(
                t1,
                t2,
                round_name,
                llm,
                is_knockout=True,
                fixture_seed=fixture_seed,
            )
            next_round.append(winner)
            self._record_final_result(
                round_name, t1=t1, t2=t2, winner=winner,
            )
            self._reflection_with_retry(
                self.world.agents[winner], llm,
                f"knockout:{mk}:{winner}",
            )
        self.qualified_teams = next_round

    def _record_final_result(self, round_name, *, t1, t2, winner):
        if round_name == "Final":
            runner_up = t2 if winner == t1 else t1
            self.final_result = {"champion": winner, "runner_up": runner_up}
