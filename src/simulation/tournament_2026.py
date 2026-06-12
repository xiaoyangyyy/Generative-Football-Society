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
    load_checkpoint,
    restore_r32_fixtures,
    save_checkpoint,
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

class TournamentManager:
    def __init__(
        self,
        world_engine,
        referee_profile_weights=None,
        referee_stage_morph_strength=1.0,
        referee_profiles=None,
    ):
        self.world = world_engine
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

        # Referee archetypes: configurable distribution + stage morphing.
        self.referee_profiles = referee_profiles or {
            "lenient": {
                "strictness_mean": 0.34,
                "strictness_kappa": 16.0,
                "bias_noise": 0.22,
                "exposure_bias_weight": 0.34,
            },
            "balanced": {
                "strictness_mean": 0.52,
                "strictness_kappa": 22.0,
                "bias_noise": 0.15,
                "exposure_bias_weight": 0.42,
            },
            "strict": {
                "strictness_mean": 0.71,
                "strictness_kappa": 28.0,
                "bias_noise": 0.11,
                "exposure_bias_weight": 0.50,
            },
        }
        base_weights = referee_profile_weights or {"lenient": 0.30, "balanced": 0.48, "strict": 0.22}
        self.referee_profile_weights = self._normalize_weights(base_weights, self.referee_profiles.keys())
        self.referee_stage_morph_strength = float(np.clip(referee_stage_morph_strength, 0.1, 3.0))
        self.dialogue_engine = SocialDialogueEngine(self.world.feed, turns_per_match=4)
        self.fusion_controller = FusionController()
        self.base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
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

    def _reflection_with_retry(self, agent, llm, attempts: int = 6) -> None:
        for i in range(attempts):
            try:
                agent.perform_reflection(llm)
                return
            except Exception as exc:
                wait = min(24.0, 2.0 * (2 ** i))
                print(
                    f"  [REFLECTION] {agent.name} attempt {i + 1}/{attempts} failed: {exc} "
                    f"(retry in {wait:.0f}s)"
                )
                time.sleep(wait)
        raise RuntimeError(f"Reflection failed for {agent.name} after {attempts} attempts")

    def _save_checkpoint(self) -> None:
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
            match_results=self.match_results,
            post_group_reflection_done=self.post_group_reflection_done,
        )

    def _restore_from_checkpoint(self, ckpt: dict) -> None:
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
        print(
            f"[CHECKPOINT] Resumed phase={self.phase} matches_done={len(self.completed_matches)} "
            f"qualified={len(self.qualified_teams)}"
        )

    def run_full_tournament(self, *, resume: bool = False):
        from src.match_engine.calibration.narrative_isolation import resolve_tournament_llm

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
                self._reflection_with_retry(self.world.agents[t_name], llm)
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
                    fixture_seed = hash((g_name, t1, t2, md_idx)) % 10000
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
        normalized = {k: max(1e-6, float(weights.get(k, 0.0))) for k in keys}
        total = sum(normalized.values())
        if total <= 0:
            uniform = 1.0 / max(1, len(normalized))
            return {k: uniform for k in normalized}
        return {k: v / total for k, v in normalized.items()}

    def _stage_referee_distribution(self, stage_pressure):
        pressure = float(np.clip(stage_pressure, 0.0, 1.0))
        p = self.referee_stage_morph_strength * pressure

        # Softmax on morphed logits:
        # more pressure -> less lenient, slightly more strict.
        logits = {}
        for name, base_w in self.referee_profile_weights.items():
            base_log = np.log(max(1e-9, base_w))
            if name == "lenient":
                morph = -1.10 * p
            elif name == "balanced":
                morph = -0.15 * p
            elif name == "strict":
                morph = 1.20 * p
            else:
                morph = 0.0
            logits[name] = base_log + morph

        max_logit = max(logits.values())
        exp_scores = {k: np.exp(v - max_logit) for k, v in logits.items()}
        total = sum(exp_scores.values())
        return {k: (v / total) for k, v in exp_scores.items()}

    def _sample_referee_profile(self, a1, a2, stage_pressure):
        dist = self._stage_referee_distribution(stage_pressure)
        profile_names = list(dist.keys())
        probs = np.array([dist[n] for n in profile_names], dtype=float)
        if (not np.all(np.isfinite(probs))) or float(np.sum(probs)) <= 0.0:
            raise ValueError(f"Invalid referee profile probabilities: {probs}")
        chosen = np.random.choice(profile_names, p=probs / probs.sum())
        profile = self.referee_profiles[chosen]

        # Sample strictness from beta(mean, concentration).
        mean = float(np.clip(profile["strictness_mean"], 0.02, 0.98))
        kappa = float(max(4.0, profile["strictness_kappa"]))
        alpha = max(1e-6, mean * kappa)
        beta = max(1e-6, (1.0 - mean) * kappa)
        strictness = np.random.beta(alpha, beta)
        # Knockout pressure pushes whistles towards conservative officiating.
        strictness = float(np.clip(strictness + 0.18 * stage_pressure * (1.0 - strictness), 0.05, 0.98))

        exp_1 = float(np.clip(np.nan_to_num(a1.media_exposure, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0))
        exp_2 = float(np.clip(np.nan_to_num(a2.media_exposure, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0))
        base_bias = profile["exposure_bias_weight"] * np.tanh((exp_1 - exp_2) * 1.8)
        noise = np.random.normal(0.0, profile["bias_noise"] * (1.0 - 0.30 * stage_pressure))
        bias_t1 = float(np.clip(base_bias + noise, -0.85, 0.85))
        return {
            "profile_name": chosen,
            "profile_distribution": dist,
            "strictness": strictness,
            "bias_t1": bias_t1,
            "bias_t2": -bias_t1,
        }

    def play_match(
        self,
        t1_name,
        t2_name,
        stage_name,
        llm,
        is_knockout=True,
        *,
        matchday: int = 0,
        standings_snapshot=None,
        fixture_seed: int = 0,
        scheduled_home: str | None = None,
    ):
        a1, a2 = self.world.agents[t1_name], self.world.agents[t2_name]
        home_micro, away_micro, neutral_venue, venue_label = resolve_match_venue(
            t1_name, t2_name, matchday=matchday, fixture_seed=fixture_seed, scheduled_home=scheduled_home
        )
        ah, aa = self.world.agents[home_micro], self.world.agents[away_micro]

        def _venue_for(team: str) -> str:
            if team == home_micro:
                return f"{venue_label} — we are HOME"
            if team == away_micro:
                return f"{venue_label} — we are AWAY"
            return venue_label

        print(
            f"  [VENUE] micro home={home_micro} | {venue_label} | "
            f"home boost= crowd ψ/morale (not possession skew)"
        )

        # 0. REST/RECOVERY + cross-match carryover
        pressure = self._stage_pressure(stage_name, is_knockout)
        rest_units = 1.1 + 1.25 * pressure
        a1.recover(rest_units=rest_units)
        a2.recover(rest_units=rest_units)
        prepare_match_agents(ah, aa, self.base_dir)

        # 1. COACHING → full 21-dim tactical vector (group table + venue in context)
        ctx1 = build_coach_match_context(
            a1.get_context_for_llm(),
            group_name=stage_name,
            matchday=matchday,
            team=t1_name,
            opponent=t2_name,
            standings_snapshot=standings_snapshot,
            venue_label=_venue_for(t1_name),
        )
        ctx2 = build_coach_match_context(
            a2.get_context_for_llm(),
            group_name=stage_name,
            matchday=matchday,
            team=t2_name,
            opponent=t1_name,
            standings_snapshot=standings_snapshot,
            venue_label=_venue_for(t2_name),
        )
        t1_tactics_json = llm.coach_decide_tactics(t1_name, ctx1, t2_name, ctx2)
        t2_tactics_json = llm.coach_decide_tactics(t2_name, ctx2, t1_name, ctx1)

        try:
            t1_data = apply_coach_tactics_from_llm(a1, t1_tactics_json)
            a1.apply_beliefs_to_tactics(stage_name=stage_name, opponent_style=a2.style_archetype)
            a1.coach_intervention(t1_tactics_json)
            print(f"  [TACTICS] {t1_name}: {t1_data.get('reasoning', 'No reasoning')}")

            t2_data = apply_coach_tactics_from_llm(a2, t2_tactics_json)
            a2.apply_beliefs_to_tactics(stage_name=stage_name, opponent_style=a1.style_archetype)
            a2.coach_intervention(t2_tactics_json)
            print(f"  [TACTICS] {t2_name}: {t2_data.get('reasoning', 'No reasoning')}")
            try:
                from src.simulation.narrative_debug import log_llm_tactics, narrative_debug_enabled

                if narrative_debug_enabled():
                    log_llm_tactics(
                        stage=stage_name,
                        home=home_micro,
                        away=away_micro,
                        agent_home=ah,
                        agent_away=aa,
                        llm_home=t1_data if home_micro == t1_name else t2_data,
                        llm_away=t2_data if away_micro == t2_name else t1_data,
                    )
            except Exception as _dbg_exc:
                print(f"  [DEBUG-NARRATIVE] tactics log skipped: {_dbg_exc}")
        except Exception as e:
            print(f"  [ERROR] Tactics parsing failed: {e}")

        # 1.5 INTERNAL GAME (Players vs Coach) + REFEREE GAME
        internal_1 = a1.simulate_internal_game(stage_pressure=pressure)
        internal_2 = a2.simulate_internal_game(stage_pressure=pressure)
        referee = self._sample_referee_profile(a1, a2, pressure)
        ref_1 = a1.apply_referee_dynamics(referee["strictness"], referee["bias_t1"])
        ref_2 = a2.apply_referee_dynamics(referee["strictness"], referee["bias_t2"])
        print(
            f"  [GAME] {t1_name} coord={internal_1['coordination']:.2f} heat={internal_1['conflict_heat']:.2f}/1.05 | "
            f"{t2_name} coord={internal_2['coordination']:.2f} heat={internal_2['conflict_heat']:.2f}/1.05"
        )
        print(
            f"  [REF] {referee['profile_name']} strictness={referee['strictness']:.2f} "
            f"bias={referee['bias_t1']:+.2f} grievance={ref_1['grievance']:.2f}/{ref_2['grievance']:.2f}"
        )

        # 2. Match simulation: physics micro first (spectate) → score; macro-only fallback
        matchup_bonus_1 = compute_matchup_bonus(a1.style_archetype, a2.style_archetype)
        matchup_bonus_2 = -matchup_bonus_1
        tactical_1 = a1.tactical_effects()
        tactical_2 = a2.tactical_effects()
        context = self.fusion_controller.build_context(
            stage_pressure=pressure,
            referee_strictness=referee["strictness"],
            social_chaos_hint=0.0,
        )
        exp_1 = self.fusion_controller.export_expert_signals(a1, internal_1, ref_1, tactical_1)
        exp_2 = self.fusion_controller.export_expert_signals(a2, internal_2, ref_2, tactical_2)
        fused_1 = self.fusion_controller.fuse(exp_1, context)
        fused_2 = self.fusion_controller.fuse(exp_2, context)
        def _safe_eff(agent, bonus: float, fused: dict) -> float:
            delta = 0.55 * float(fused.get("status_delta", 0.0))
            if not np.isfinite(delta):
                delta = 0.0
            val = float(agent.get_effective_status(matchup_bonus=bonus + delta))
            vol = float(fused.get("volatility", 0.0))
            if not np.isfinite(vol):
                vol = 0.0
            val *= max(0.82, 1.0 + np.random.normal(0.0, 0.016 * vol))
            if not np.isfinite(val):
                val = max(12.0, float(agent.status_score))
            return float(max(12.0, val))

        eff_status_1 = _safe_eff(a1, matchup_bonus_1, fused_1)
        eff_status_2 = _safe_eff(a2, matchup_bonus_2, fused_2)
        if home_micro == t1_name:
            eff_micro_home, eff_micro_away = eff_status_1, eff_status_2
            internal_micro_h, internal_micro_a = internal_1, internal_2
            fused_vol_h = float(fused_1.get("volatility", 0.0))
            fused_vol_a = float(fused_2.get("volatility", 0.0))
        else:
            eff_micro_home, eff_micro_away = eff_status_2, eff_status_1
            internal_micro_h, internal_micro_a = internal_2, internal_1
            fused_vol_h = float(fused_2.get("volatility", 0.0))
            fused_vol_a = float(fused_1.get("volatility", 0.0))
        print(
            f"  [CTRL] {t1_name} p={a1.tactical_controls['pressing_intensity']:.2f} r={a1.tactical_controls['risk_budget']:.2f} "
            f"lh={a1.tactical_controls['line_height']:.2f} rot={a1.tactical_controls['rotation_aggressiveness']:.2f} | "
            f"{t2_name} p={a2.tactical_controls['pressing_intensity']:.2f} r={a2.tactical_controls['risk_budget']:.2f} "
            f"lh={a2.tactical_controls['line_height']:.2f} rot={a2.tactical_controls['rotation_aggressiveness']:.2f}"
        )
        print(
            f"  [FUSION] {t1_name} Δ={fused_1['status_delta']:+.2f} vol={fused_1['volatility']:.2f} "
            f"| {t2_name} Δ={fused_2['status_delta']:+.2f} vol={fused_2['volatility']:.2f}"
        )
        score_path = resolve_score_path_mode()
        physics_first = score_path == ScorePathMode.PHYSICS_OFFICIAL
        micro_replay_mode = score_path == ScorePathMode.MICRO_REPLAY
        legacy_poisson = score_path == ScorePathMode.POISSON_LEGACY
        drama_pre = min(1.0, 0.25 + 0.15 * referee["strictness"])
        micro_summary = None
        score_meta: dict = {}
        gfs_seed = int(os.environ.get("GFS_SEED", "42"))
        seed = gfs_seed + hash((t1_name, t2_name, stage_name)) % 10000
        rng_score = np.random.default_rng(seed)
        xg_prior_h, xg_prior_a = 1.0, 1.0

        if physics_first:
            xg_prior_h, xg_prior_a, _xg_meta = expected_match_xg(
                ah,
                aa,
                eff_micro_home,
                eff_micro_away,
                is_knockout=is_knockout,
                stage_pressure=pressure,
                fused_volatility_home=fused_vol_h,
                fused_volatility_away=fused_vol_a,
                rng=rng_score,
            )
            from src.memory_engine.macro_goal_dynamics import clamp_match_xg

            if not np.isfinite(xg_prior_h):
                xg_prior_h = 1.0
            if not np.isfinite(xg_prior_a):
                xg_prior_a = 1.0
            xg_prior_h = clamp_match_xg(xg_prior_h)
            xg_prior_a = clamp_match_xg(xg_prior_a)
            try:
                micro_summary = run_physics_first_micro(
                    ah,
                    aa,
                    xg_prior_home=xg_prior_h,
                    xg_prior_away=xg_prior_a,
                    eff_status_home=eff_micro_home,
                    eff_status_away=eff_micro_away,
                    referee=referee,
                    stage_pressure=pressure,
                    drama_score=drama_pre,
                    internal_home=internal_micro_h,
                    internal_away=internal_micro_a,
                    seed=seed,
                    stage_name=stage_name,
                    neutral_venue=neutral_venue,
                )
                print_micro_match_logs(micro_summary, home_micro, away_micro, ah, aa)
                s1, s2, xg1, xg2, score_meta = finalize_official_score_from_micro(
                    micro_summary,
                    home_micro=home_micro,
                    fixture_home=t1_name,
                    macro_xg_prior_home=xg_prior_h,
                    macro_xg_prior_away=xg_prior_a,
                )
                score_meta["xg_prior"] = [xg_prior_h, xg_prior_a]
            except Exception as exc:
                strict = os.environ.get("MATCH_MICRO_STRICT", "").strip().lower() in ("1", "true", "yes")
                print(f"  [MICRO] physics-first failed ({exc}); falling back to macro score.")
                if strict:
                    raise RuntimeError(
                        f"Micro physics-first failed ({t1_name} vs {t2_name}): {exc}"
                    ) from exc
                physics_first = False
                micro_summary = None

        if not physics_first:
            if legacy_poisson:
                from src.memory_engine.macro_goal_dynamics import simulate_match_score_dynamics

                s1, s2, xg1, xg2, score_meta = simulate_match_score_dynamics(
                    a1,
                    a2,
                    eff_status_1,
                    eff_status_2,
                    is_knockout=is_knockout,
                    stage_pressure=pressure,
                    fused_volatility_home=float(fused_1.get("volatility", 0.0)),
                    fused_volatility_away=float(fused_2.get("volatility", 0.0)),
                    rng=rng_score,
                )
            else:
                s1, s2, xg1, xg2, score_meta = resolve_unified_score(
                    a1,
                    a2,
                    eff_status_1,
                    eff_status_2,
                    is_knockout=is_knockout,
                    stage_pressure=pressure,
                    fused_volatility_home=float(fused_1.get("volatility", 0.0)),
                    fused_volatility_away=float(fused_2.get("volatility", 0.0)),
                    micro_summary=micro_summary,
                    rng=rng_score,
                )

        pen_note = ""
        winner_name = None
        pen1 = pen2 = None
        aet1 = aet2 = 0
        et_xg1 = et_xg2 = 0.0
        went_to_extra_time = False
        reg_s1, reg_s2 = s1, s2
        if is_knockout and s1 == s2:
            went_to_extra_time = True
            print(f"  [AET] Full micro extra time ({home_micro} vs {away_micro})...")
            if physics_first:
                et_summary = run_extra_time_micro(
                    ah,
                    aa,
                    xg_prior_home=xg_prior_h,
                    xg_prior_away=xg_prior_a,
                    eff_status_home=eff_micro_home,
                    eff_status_away=eff_micro_away,
                    referee=referee,
                    stage_pressure=pressure,
                    drama_score=drama_pre,
                    internal_home=internal_micro_h,
                    internal_away=internal_micro_a,
                    seed=seed,
                    stage_name=stage_name,
                    neutral_venue=neutral_venue,
                )
                et_gh, et_ga = int(et_summary.goals_micro_home), int(et_summary.goals_micro_away)
                et_xgh, et_xga = float(et_summary.micro_xg_home), float(et_summary.micro_xg_away)
                aet1, aet2 = self._map_micro_score(home_micro, t1_name, et_gh, et_ga)
                et_xg1 = et_xgh if home_micro == t1_name else et_xga
                et_xg2 = et_xga if home_micro == t1_name else et_xgh
                print_micro_match_logs(et_summary, home_micro, away_micro, ah, aa)
            else:
                from src.memory_engine.macro_goal_dynamics import simulate_extra_time_dynamics

                aet1, aet2, et_xg1, et_xg2 = simulate_extra_time_dynamics(
                    a1, a2, eff_status_1, eff_status_2, stage_pressure=pressure
                )
            s1 += aet1
            s2 += aet2
            if s1 == s2:
                pen1, pen2 = simulate_penalty_shootout()
                winner_name = t1_name if pen1 > pen2 else t2_name
                pen_note = f" (Pens {pen1}-{pen2})"
            else:
                winner_name = t1_name if s1 > s2 else t2_name
        elif s1 > s2:
            winner_name = t1_name
        elif s2 > s1:
            winner_name = t2_name

        drama_score = min(
            1.0,
            0.25 + abs(s1 - s2) * 0.12 + (0.2 if s1 == s2 else 0.0) + 0.15 * referee["strictness"],
        )
        key_event = f"xG battle {xg1}-{xg2} with styles {a1.style_archetype} vs {a2.style_archetype}{pen_note}"
        score_model = score_path_label(score_path)
        verdict_json = json.dumps(
            {
                "winner": winner_name,
                "score": f"{s1}-{s2}",
                "key_event": key_event,
                "drama_score": round(drama_score, 2),
                "model": score_model,
            },
            ensure_ascii=False,
        )
        print(
            f"  [SCORE] {t1_name} {s1}-{s2} {t2_name} | μxG {xg1:.2f}-{xg2:.2f} | "
            f"EffStatus {eff_status_1:.1f}-{eff_status_2:.1f} | Matchup {matchup_bonus_1:+.1f} | {score_model}"
        )
        if physics_first:
            print(
                f"  [MACRO-PRIOR] team-dynamics xG prior {xg_prior_h:.2f}-{xg_prior_a:.2f} "
                f"(event density only; goals from physics)"
            )
        if went_to_extra_time:
            print(
                f"  [AET] {t1_name} {reg_s1}-{reg_s2} {t2_name} after 90' → "
                f"{s1}-{s2} after ET (+{aet1}/+{aet2} goals, ET μxG +{et_xg1:.2f}/+{et_xg2:.2f})."
            )
        if is_knockout and pen1 is not None and winner_name:
            print(
                f"  [PENALTIES] {winner_name} wins {pen1}-{pen2} on penalties."
            )
        xg_note = score_xg_anomaly_note(t1_name, t2_name, s1, s2, xg1, xg2)
        is_final_fixture = stage_name.strip().lower() == "final"
        final_xg = None
        if is_final_fixture:
            final_xg = finalize_stage_xg_context(
                t1_name, t2_name, s1, s2, xg1, xg2, went_to_penalties=(pen1 is not None)
            )
        merged_xg = []
        if final_xg:
            merged_xg.append(final_xg)
        if xg_note:
            merged_xg.append(xg_note)
        xg_context_line = "; ".join(list(dict.fromkeys(merged_xg))) if merged_xg else ""
        if xg_context_line:
            print(f"  [XG_CONTEXT] {xg_context_line}")

        try:
            from src.simulation.narrative_debug import log_match_debug, narrative_debug_enabled

            if narrative_debug_enabled():
                log_match_debug(
                    stage=stage_name,
                    home=home_micro,
                    away=away_micro,
                    score_home=s1 if home_micro == t1_name else s2,
                    score_away=s2 if away_micro == t2_name else s1,
                    xg_home=xg1 if home_micro == t1_name else xg2,
                    xg_away=xg2 if away_micro == t2_name else xg1,
                    agent_home=ah,
                    agent_away=aa,
                    micro=micro_summary,
                    xg_context=xg_context_line,
                )
        except Exception as _dbg_exc:
            print(f"  [DEBUG-NARRATIVE] match log skipped: {_dbg_exc}")

        aff_on = os.environ.get("MATCH_AFFECTIVE", "").strip().lower() in ("1", "true", "yes")
        if micro_replay_mode or (aff_on and micro_summary is None):
            try:
                if micro_replay_mode:
                    aff = run_micro_layer(
                        a1,
                        a2,
                        goals_home=s1,
                        goals_away=s2,
                        xg_home=xg1,
                        xg_away=xg2,
                        referee=referee,
                        stage_pressure=pressure,
                        drama_score=drama_score,
                        internal_home=internal_1,
                        internal_away=internal_2,
                        seed=seed,
                        stage_name=stage_name,
                    )
                    print_micro_match_logs(aff, t1_name, t2_name, a1, a2)
                    print(
                        f"  [REPLAY] macro score {s1}-{s2} anchored; set MATCH_MICRO_SCORE=1 (default with MICRO) for physics-first."
                    )
                    micro_summary = aff
                else:
                    from src.match_engine.match_affective_runner import run_match_affective_simulation

                    aff = run_match_affective_simulation(
                        a1,
                        a2,
                        goals_home=s1,
                        goals_away=s2,
                        xg_home=xg1,
                        xg_away=xg2,
                        referee=referee,
                        stage_pressure=pressure,
                        drama_score=drama_score,
                        internal_home=internal_1,
                        internal_away=internal_2,
                        seed=seed,
                        writeback_agents=True,
                    )
                    print(
                        f"  [AFFECTIVE] ψ={aff.final_psi:+.2f} | coach stress {aff.home_coach_stress:.2f}/{aff.away_coach_stress:.2f} "
                        f"| ref strict {aff.ref_strictness_mean:.2f} | drift {aff.tactical_drift_home:.2f}/{aff.tactical_drift_away:.2f}"
                    )
            except Exception as exc:
                print(f"  [MICRO/AFFECTIVE] skipped: {exc}")

        # Optional LLM narrative layer (no control over scoreline)
        facts_ledger = {
            "t1": t1_name,
            "t2": t2_name,
            "score": f"{s1}-{s2}",
            "winner": winner_name if winner_name is not None else "draw",
            "xg": f"{xg1}-{xg2}",
            "stage": stage_name,
            "t1_pressing_intensity": round(float(a1.tactical_controls.get("pressing_intensity", 0.5)), 2),
            "t1_risk_budget": round(float(a1.tactical_controls.get("risk_budget", 0.5)), 2),
            "t1_line_height": round(float(a1.tactical_controls.get("line_height", 0.5)), 2),
            "t1_rotation_aggressiveness": round(float(a1.tactical_controls.get("rotation_aggressiveness", 0.5)), 2),
            "t2_pressing_intensity": round(float(a2.tactical_controls.get("pressing_intensity", 0.5)), 2),
            "t2_risk_budget": round(float(a2.tactical_controls.get("risk_budget", 0.5)), 2),
            "t2_line_height": round(float(a2.tactical_controls.get("line_height", 0.5)), 2),
            "t2_rotation_aggressiveness": round(float(a2.tactical_controls.get("rotation_aggressiveness", 0.5)), 2),
            "drama_score": round(float(drama_score), 2),
            "effective_status": f"{eff_status_1:.1f}-{eff_status_2:.1f}",
            "t1_tactical_preset": getattr(getattr(a1, "coach_profile", None), "preferred_preset", ""),
            "t2_tactical_preset": getattr(getattr(a2, "coach_profile", None), "preferred_preset", ""),
            "t1_through_ball_bias": round(float(getattr(a1, "tactical_vector", {}).get("through_ball_bias", 0.5)), 2),
            "t2_through_ball_bias": round(float(getattr(a2, "tactical_vector", {}).get("through_ball_bias", 0.5)), 2),
        }
        if xg_context_line:
            facts_ledger["xg_alignment_note"] = xg_context_line
        if micro_summary is not None and getattr(micro_summary, "cognitive_plans", None):
            facts_ledger["cognitive_events"] = len(micro_summary.cognitive_plans)
            facts_ledger["cognitive_tier_usage"] = getattr(
                micro_summary, "cognitive_tier_usage", {}
            )
        facts_ledger["regulation_score"] = f"{reg_s1}-{reg_s2}" if went_to_extra_time else f"{s1}-{s2}"
        facts_ledger["aet_played"] = bool(went_to_extra_time)
        facts_ledger["aet_score"] = f"{s1}-{s2}" if went_to_extra_time else "not_played"
        facts_ledger["went_to_penalties"] = bool(pen1 is not None)
        facts_ledger["penalty_score"] = f"{pen1}-{pen2}" if pen1 is not None else "not_played"
        tg = int(s1 + s2)
        facts_ledger["total_goals"] = tg
        if tg >= 5:
            facts_ledger["tone_contract"] = (
                "High total goals in FACTS_LEDGER — do not describe the match as goalless, 0-0, or devoid of scoring "
                "action; you may still discuss tension, errors, or tactical tradeoffs."
            )
        elif tg <= 1:
            facts_ledger["tone_contract"] = (
                "Low total goals — do not claim a multi-goal shootout or cricket-score thriller."
            )
        else:
            facts_ledger["tone_contract"] = (
                "Moderate scoring — keep tone aligned with listed score and drama_score; avoid contradictory extremes."
            )
        if pen1 is not None:
            facts_ledger["tone_contract"] += (
                " If penalties occurred, ensure wording is consistent with AET and penalty_score fields."
            )
        if float(drama_score) >= 0.55:
            facts_ledger["tone_contract"] += (
                " drama_score is relatively high — emotional stakes can be sharp without inventing stats."
            )

        narrative_json = llm.predict_match_result(
            t1_name, a1.get_context_for_llm(), t2_name, a2.get_context_for_llm(), stage_name
        )
        try:
            narrative = json.loads(narrative_json)
            if isinstance(narrative, dict) and narrative.get("key_event"):
                verdict_data = json.loads(verdict_json)
                verdict_data["llm_key_event"] = narrative.get("key_event")
                verdict_json = json.dumps(verdict_data, ensure_ascii=False)
        except Exception:
            pass

        if not is_knockout: self.update_standings(stage_name, t1_name, t2_name, s1, s2)

        # 4. MEDIA & RIVALRY
        media_json = llm.generate_media_matrix(
            verdict_json, t1_name, t2_name, a1.media_exposure, a2.media_exposure, facts_ledger=facts_ledger
        )
        a1.update_rivalry(t2_name, drama_score)
        a2.update_rivalry(t1_name, drama_score)
        
        try:
            media_data = json.loads(media_json)
            metrics = media_data.get("metrics", {})
            prof_score = float(metrics.get("professional_score", 0.0))
            social_chaos = (
                float(metrics.get("social_chaos", 0.0))
                + 0.5 * (ref_1["chaos_delta"] + ref_2["chaos_delta"])
                + 0.25 * (fused_1["chaos_push"] + fused_2["chaos_push"])
            )
            mainstream = media_data.get("mainstream", "")
            social_post = media_data.get("social_chaos_post", "")
            if mainstream:
                self.world.feed.publish(
                    "GLOBAL_FOOTBALL_NEWS",
                    f"{t1_name} vs {t2_name} | {stage_name}: {mainstream}",
                    tags=["media", "analysis", stage_name]
                )
            if social_post:
                self.world.feed.publish(
                    "SOCIAL_TREND",
                    social_post,
                    tags=["social", "chaos", stage_name]
                )
        except Exception as e:
            print(f"  [ERROR] Media matrix parsing failed: {e}")
            prof_score, social_chaos = 0.0, 0.0

        # 5.5 SOCIAL DIALOGUE LAYER (speech acts + meme market + signal compression)
        dialogue_pack = self.dialogue_engine.run_post_match_dialogue(
            a1,
            a2,
            stage_name,
            context={
                "winner": winner_name or "draw",
                "score": f"{s1}-{s2}",
                "score_diff": float(s1 - s2),
                "key_event": key_event,
                "drama_score": drama_score,
                "stage_pressure": pressure,
                "ref_strictness": referee["strictness"],
                "cognitive_plans": (
                    micro_summary.cognitive_plans[:6]
                    if micro_summary is not None and getattr(micro_summary, "cognitive_plans", None)
                    else []
                ),
            },
        )
        sig_1 = dialogue_pack.get("signals", {}).get(t1_name, {})
        sig_2 = dialogue_pack.get("signals", {}).get(t2_name, {})
        social_fb_1 = a1.apply_social_signal(sig_1, stage_pressure=pressure)
        social_fb_2 = a2.apply_social_signal(sig_2, stage_pressure=pressure)
        social_chaos += (
            float(dialogue_pack.get("market_chaos", 0.0))
            + float(social_fb_1.get("chaos_delta", 0.0))
            + float(social_fb_2.get("chaos_delta", 0.0))
        )
        a1.compress_social_memory(stage_name)
        a2.compress_social_memory(stage_name)
        top_topics = dialogue_pack.get("topic_snapshot", [])[:3]
        print(
            f"  [DIALOGUE] market_chaos={dialogue_pack.get('market_chaos', 0.0):.2f} "
            f"topics={top_topics}"
        )
        
        # 6. RECURSIVE UPDATE (Momentum-Aware)
        res_1 = "win" if s1 > s2 else ("loss" if s2 > s1 else "draw")
        res_2 = "win" if s2 > s1 else ("loss" if s1 > s2 else "draw")
        a1.recursive_update(res_1, s1-s2, prof_score, social_chaos, t2_name, a2.status_score)
        a2.recursive_update(res_2, s2-s1, prof_score, social_chaos, t1_name, a1.status_score)
        governance_signal_1 = internal_1["coordination"] - 0.25 * ref_1["grievance"]
        governance_signal_2 = internal_2["coordination"] - 0.25 * ref_2["grievance"]
        a1.relax_referee_grievance_post_match(res_1, referee["bias_t1"], drama_score)
        a2.relax_referee_grievance_post_match(res_2, referee["bias_t2"], drama_score)

        # 8. PHYSICAL WEAR (continuous stage pressure + match intensity)
        wear_intensity = 0.24 + 0.36 * pressure + 0.06 * drama_score
        pre_fatigue_a = a1.fatigue
        pre_fatigue_b = a2.fatigue
        injury_a = a1.apply_match_wear(intensity=wear_intensity * tactical_1["fatigue_load_multiplier"])
        injury_b = a2.apply_match_wear(intensity=wear_intensity * tactical_2["fatigue_load_multiplier"])
        for agent, lbl, _spike in (
            (a1, t1_name, injury_a),
            (a2, t2_name, injury_b),
        ):
            print(f"  [MEDICAL] {lbl} injury_load={agent.injury_load:.2f}")
            # Monotonic report probability: only a function of injury_load.
            p_report = float(1.0 / (1.0 + np.exp(-10.0 * (float(agent.injury_load) - 0.62))))
            if np.random.random() < p_report:
                print(f"  [MEDICAL] {lbl} reported injury concerns.")

        a1.record_decision_event(
            opponent=t2_name,
            stage_name=stage_name,
            controls=a1.tactical_controls,
            outcomes={
                "goal_diff": s1 - s2,
                "xg_for": xg1,
                "xg_against": xg2,
                "fatigue_delta": a1.fatigue - pre_fatigue_a,
                "chaos": social_chaos,
                "result": res_1,
            },
            opponent_style=a2.style_archetype,
            stage_pressure=pressure,
        )
        a2.record_decision_event(
            opponent=t1_name,
            stage_name=stage_name,
            controls=a2.tactical_controls,
            outcomes={
                "goal_diff": s2 - s1,
                "xg_for": xg2,
                "xg_against": xg1,
                "fatigue_delta": a2.fatigue - pre_fatigue_b,
                "chaos": social_chaos,
                "result": res_2,
            },
            opponent_style=a1.style_archetype,
            stage_pressure=pressure,
        )
        
        finalize_match_feedback(
            a1,
            a2,
            base_dir=self.base_dir,
            result_home=res_1,
            result_away=res_2,
            score_diff_home=s1 - s2,
            xg_home=xg1,
            xg_away=xg2,
            prof_score=prof_score,
            social_chaos=social_chaos,
            stage_name=stage_name,
            micro_summary=micro_summary if micro_layer_enabled() else None,
        )

        # 7. CASCADE — continuous locker tension; "explosion" headline is rare
        for agent, label in ((a1, t1_name), (a2, t2_name)):
            tense = agent.locker_room_tension()
            print(f"  [LOCKER_ROOM] {label} tension={tense:.2f}")
            if agent.rare_locker_room_explosion():
                print(f"  [EMERGENCY] {label} Locker Room Explosion!")

        winner = winner_name or (t1_name if s1 >= s2 else t2_name)
        mk = self._match_key(stage_name, t1_name, t2_name)
        self.completed_matches.append(mk)
        self.match_results[mk] = winner
        self.match_index += 1
        self._save_checkpoint()
        return winner

    def update_standings(self, g, t1, t2, s1, s2):
        st = self.standings[g]
        st[t1]["pts"] += (3 if s1 > s2 else (1 if s1 == s2 else 0))
        st[t2]["pts"] += (3 if s2 > s1 else (1 if s1 == s2 else 0))
        st[t1]["gf"] += s1; st[t1]["ga"] += s2; st[t1]["gd"] = st[t1]["gf"] - st[t1]["ga"]
        st[t2]["gf"] += s2; st[t2]["ga"] += s1; st[t2]["gd"] = st[t2]["gf"] - st[t2]["ga"]

    def resolve_advancements(self):
        self.qualified_entries = build_qualified_entries(self.groups, self.standings)
        self.qualified_teams = [e.team for e in self.qualified_entries]
        gfs_seed = int(os.environ.get("GFS_SEED", "42"))
        import random

        self.r32_fixtures = build_r32_pairings(
            self.qualified_entries, rng=random.Random(gfs_seed + 2026)
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
                    print(f"  [SKIP] {t1} vs {t2} (checkpoint) → {winner}")
                    next_round.append(winner)
                    continue
            fixture_seed = hash((round_name, t1, t2)) % 10000
            winner = self.play_match(
                t1,
                t2,
                round_name,
                llm,
                is_knockout=True,
                fixture_seed=fixture_seed,
            )
            next_round.append(winner)
            if round_name == "Final":
                runner_up = t2 if winner == t1 else t1
                self.final_result = {"champion": winner, "runner_up": runner_up}
            self._reflection_with_retry(self.world.agents[winner], llm)
        self.qualified_teams = next_round
