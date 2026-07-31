"""Tournament setup, tactics, referee, and expert-fusion stages."""

import copy
import inspect

import numpy as np

from src.simulation.group_context import build_coach_match_context
from src.simulation.match_pipeline import prepare_match_agents
from src.simulation.tactical_matchup import compute_matchup_bonus
from src.simulation.tactics_sync import apply_coach_tactics_from_llm
from src.simulation.venue_policy import resolve_match_venue


class TournamentSetupMixin:
    @staticmethod
    def _venue_description(team, *, home_micro, away_micro, venue_label):
        if team == home_micro:
            return f"{venue_label} — we are HOME"
        if team == away_micro:
            return f"{venue_label} — we are AWAY"
        return venue_label

    def _resolve_match_participants(
        self, *, t1_name, t2_name, matchday, fixture_seed, scheduled_home,
    ):
        a1, a2 = self.world.agents[t1_name], self.world.agents[t2_name]
        venue = resolve_match_venue(
            t1_name, t2_name, matchday=matchday,
            fixture_seed=fixture_seed, scheduled_home=scheduled_home,
        )
        home_micro, away_micro, neutral_venue, venue_label = venue
        ah, aa = self.world.agents[home_micro], self.world.agents[away_micro]
        print(
            f"  [VENUE] micro home={home_micro} | {venue_label} | "
            f"home boost= crowd ψ/morale (not possession skew)"
        )
        return a1, a2, ah, aa, home_micro, away_micro, neutral_venue, venue_label

    def _prepare_team_recovery(self, *, a1, a2, ah, aa, pressure):
        rest_units = 1.1 + 1.25 * pressure
        a1.recover(rest_units=rest_units)
        a2.recover(rest_units=rest_units)
        prepare_match_agents(ah, aa, self.base_dir)

    def _build_coaching_contexts(
        self, *, a1, a2, t1_name, t2_name, stage_name, matchday,
        standings_snapshot, home_micro, away_micro, venue_label,
    ):
        common = {
            "group_name": stage_name,
            "matchday": matchday,
            "standings_snapshot": standings_snapshot,
        }
        ctx1 = build_coach_match_context(
            a1.get_context_for_llm(), team=t1_name, opponent=t2_name,
            venue_label=self._venue_description(
                t1_name, home_micro=home_micro, away_micro=away_micro,
                venue_label=venue_label,
            ),
            **common,
        )
        ctx2 = build_coach_match_context(
            a2.get_context_for_llm(), team=t2_name, opponent=t1_name,
            venue_label=self._venue_description(
                t2_name, home_micro=home_micro, away_micro=away_micro,
                venue_label=venue_label,
            ),
            **common,
        )
        return ctx1, ctx2

    def _apply_match_tactics(
        self, *, a1, a2, ah, aa, t1_name, t2_name, stage_name,
        home_micro, away_micro, llm, ctx1, ctx2,
    ):
        packets = self._build_prematch_world_model_packets(
            ah=ah, aa=aa, home_micro=home_micro, away_micro=away_micro,
        )
        packets = self._enrich_prematch_packets_with_history(
            packets, t1_name=t1_name, t2_name=t2_name,
        )
        t1_packet = packets[t1_name]
        t2_packet = packets[t2_name]
        t1_tactics_json = self._request_coach_tactics(
            llm, t1_name, ctx1, t2_name, ctx2, t1_packet,
        )
        t2_tactics_json = self._request_coach_tactics(
            llm, t2_name, ctx2, t1_name, ctx1, t2_packet,
        )
        try:
            t1_data = apply_coach_tactics_from_llm(
                a1, t1_tactics_json, decision_support=t1_packet,
            )
            a1.apply_beliefs_to_tactics(
                stage_name=stage_name, opponent_style=a2.style_archetype,
            )
            a1.coach_intervention(t1_tactics_json)
            print(f"  [TACTICS] {t1_name}: {t1_data.get('reasoning', 'No reasoning')}")

            t2_data = apply_coach_tactics_from_llm(
                a2, t2_tactics_json, decision_support=t2_packet,
            )
            a2.apply_beliefs_to_tactics(
                stage_name=stage_name, opponent_style=a1.style_archetype,
            )
            a2.coach_intervention(t2_tactics_json)
            print(f"  [TACTICS] {t2_name}: {t2_data.get('reasoning', 'No reasoning')}")
            self._log_match_tactics(
                stage_name=stage_name, home_micro=home_micro,
                away_micro=away_micro, ah=ah, aa=aa,
                t1_name=t1_name, t2_name=t2_name,
                t1_data=t1_data, t2_data=t2_data,
            )
        except Exception as exc:
            print(f"  [ERROR] Tactics parsing failed: {exc}")

    @staticmethod
    def _request_coach_tactics(
        llm, team_name, my_info, opp_name, opp_info, decision_support,
    ):
        """Keep third-party coach adapters compatible while adding evidence."""
        method = llm.coach_decide_tactics
        try:
            parameters = inspect.signature(method).parameters.values()
            accepts_packet = any(
                parameter.name == "world_model_decision_support"
                or parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
        except (TypeError, ValueError):
            accepts_packet = False
        if accepts_packet:
            return method(
                team_name, my_info, opp_name, opp_info,
                world_model_decision_support=decision_support,
            )
        return method(team_name, my_info, opp_name, opp_info)

    def _build_prematch_world_model_packets(
        self, *, ah, aa, home_micro, away_micro,
    ):
        """Create symmetric pre-match evidence from one immutable base state."""
        from src.match_engine.world_model.config import world_model_plan_enabled
        from src.match_engine.world_model.decision_support import (
            build_prematch_tactical_packet,
        )

        unavailable = {
            "version": 1,
            "available": False,
            "reason": "world_model_planning_disabled",
            "recommended_tactical_preset": "none",
            "candidates": [],
        }
        if not world_model_plan_enabled():
            return {home_micro: dict(unavailable), away_micro: dict(unavailable)}

        if not hasattr(self, "_prematch_world_model_runtime"):
            from src.match_engine.world_model.inference import WorldModelRuntime

            self._prematch_world_model_runtime = WorldModelRuntime.load_default(
                self.base_dir,
            )
        runtime = self._prematch_world_model_runtime
        if runtime is None:
            missing = dict(unavailable)
            missing["reason"] = "world_model_unavailable"
            return {home_micro: dict(missing), away_micro: dict(missing)}

        try:
            from src.match_engine.macro_bridge import build_match_affective_state

            base_state = build_match_affective_state(ah, aa)

            def representative_state(team_id):
                state = copy.deepcopy(base_state)
                team = state.team(team_id)
                carrier = next(
                    (player for player in team.players if player.role == "CM"),
                    team.players[0],
                )
                state.ball.position = carrier.position.copy()
                state.ball.velocity = np.zeros(2, dtype=float)
                state.ball.possessor_id = carrier.player_id
                state.ball.possession_team_id = team.team_id
                return state

            packets = {
                home_micro: build_prematch_tactical_packet(
                    runtime, representative_state(home_micro), home_micro,
                ),
                away_micro: build_prematch_tactical_packet(
                    runtime, representative_state(away_micro), away_micro,
                ),
            }
            return packets
        except Exception as exc:
            failed = dict(unavailable)
            failed["reason"] = f"world_model_error:{type(exc).__name__}"
            return {home_micro: dict(failed), away_micro: dict(failed)}

    def _enrich_prematch_packets_with_history(
        self, packets, *, t1_name, t2_name,
    ):
        """Attach persistent strategy memory and conservative trust guidance."""
        try:
            from src.simulation.fusion_audit import (
                fusion_audit_path,
                load_fusion_audits,
            )
            from src.simulation.counterfactual_evidence import (
                counterfactual_evidence_path,
                load_counterfactual_evidence,
            )
            from src.simulation.fusion_reliability import (
                enrich_decision_packet_with_history,
            )

            audits = load_fusion_audits(fusion_audit_path(self.base_dir))
            counterfactuals = load_counterfactual_evidence(
                counterfactual_evidence_path(self.base_dir)
            )
            return {
                t1_name: enrich_decision_packet_with_history(
                    packets[t1_name], team=t1_name, opponent=t2_name,
                    audit_records=audits,
                    counterfactual_records=counterfactuals,
                ),
                t2_name: enrich_decision_packet_with_history(
                    packets[t2_name], team=t2_name, opponent=t1_name,
                    audit_records=audits,
                    counterfactual_records=counterfactuals,
                ),
            }
        except (OSError, TypeError, ValueError) as exc:
            print(f"  [WORLD_MODEL] Historical fusion context skipped: {exc}")
            return packets

    @staticmethod
    def _log_match_tactics(
        *, stage_name, home_micro, away_micro, ah, aa,
        t1_name, t2_name, t1_data, t2_data,
    ):
        try:
            from src.simulation.narrative_debug import (
                log_llm_tactics,
                narrative_debug_enabled,
            )

            if narrative_debug_enabled():
                log_llm_tactics(
                    stage=stage_name, home=home_micro, away=away_micro,
                    agent_home=ah, agent_away=aa,
                    llm_home=t1_data if home_micro == t1_name else t2_data,
                    llm_away=t2_data if away_micro == t2_name else t1_data,
                )
        except Exception as exc:
            print(f"  [DEBUG-NARRATIVE] tactics log skipped: {exc}")

    def _resolve_internal_and_referee_game(
        self, *, a1, a2, t1_name, t2_name, pressure,
    ):
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
        return internal_1, internal_2, referee, ref_1, ref_2

    def _prepare_match(
        self,
        t1_name,
        t2_name,
        stage_name,
        llm,
        is_knockout,
        *,
        matchday,
        standings_snapshot,
        fixture_seed,
        scheduled_home,
    ):
        (
            a1, a2, ah, aa, home_micro, away_micro, neutral_venue, venue_label,
        ) = self._resolve_match_participants(
            t1_name=t1_name, t2_name=t2_name, matchday=matchday,
            fixture_seed=fixture_seed, scheduled_home=scheduled_home,
        )
        pressure = self._stage_pressure(stage_name, is_knockout)
        self._prepare_team_recovery(
            a1=a1, a2=a2, ah=ah, aa=aa, pressure=pressure,
        )
        ctx1, ctx2 = self._build_coaching_contexts(
            a1=a1, a2=a2, t1_name=t1_name, t2_name=t2_name,
            stage_name=stage_name, matchday=matchday,
            standings_snapshot=standings_snapshot, home_micro=home_micro,
            away_micro=away_micro, venue_label=venue_label,
        )
        self._apply_match_tactics(
            a1=a1, a2=a2, ah=ah, aa=aa, t1_name=t1_name, t2_name=t2_name,
            stage_name=stage_name, home_micro=home_micro,
            away_micro=away_micro, llm=llm, ctx1=ctx1, ctx2=ctx2,
        )
        internal_1, internal_2, referee, ref_1, ref_2 = (
            self._resolve_internal_and_referee_game(
                a1=a1, a2=a2, t1_name=t1_name, t2_name=t2_name,
                pressure=pressure,
            )
        )

        return (
            a1, a2, ah, aa, home_micro, away_micro, neutral_venue,
            pressure, internal_1, internal_2, referee, ref_1, ref_2,
        )

    def _build_fused_match_context(
        self, *, a1, a2, t1_name, t2_name, home_micro,
        pressure, referee, internal_1, internal_2, ref_1, ref_2,
    ):
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
        return {
            "matchup_bonus_1": matchup_bonus_1,
            "tactical_1": tactical_1, "tactical_2": tactical_2,
            "fused_1": fused_1, "fused_2": fused_2,
            "eff_status_1": eff_status_1, "eff_status_2": eff_status_2,
            "eff_micro_home": eff_micro_home, "eff_micro_away": eff_micro_away,
            "internal_micro_h": internal_micro_h, "internal_micro_a": internal_micro_a,
            "fused_vol_h": fused_vol_h, "fused_vol_a": fused_vol_a,
        }
