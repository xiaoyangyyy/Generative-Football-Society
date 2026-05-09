import itertools
import json
import numpy as np
from src.memory_engine.poisson_simulator import (
    simulate_match_score,
    simulate_extra_time_score,
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
    "Group A": ["Mexico", "Czech Republic", "South Africa", "South Korea"],
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
        self.final_result = {}

    def run_full_tournament(self):
        from src.simulation.llm_engine import SimulationLLM
        llm = SimulationLLM()
        print("[METRICS] conflict_heat scale: 0.00-1.05 (saturation above 1.00 is allowed by design).")
        self.simulate_group_stage(llm)
        self.resolve_advancements()
        
        print("\n" + "*"*60 + "\n[V13 GLOBAL SUMMIT] Teams performing deep reflection...\n" + "*"*60)
        for t_name in self.qualified_teams:
            self.world.agents[t_name].perform_reflection(llm)
            
        for round_name, num_teams in [("Round of 32", 32), ("Round of 16", 16), ("Quarter-Finals", 8), ("Semi-Finals", 4), ("Final", 2)]:
            self.simulate_knockout_round(round_name, num_teams, llm)

    def simulate_group_stage(self, llm):
        print("\n" + "="*60 + "\n🚀 PHASE 1: GROUP STAGE (CINDERELLA FIELD ACTIVE)\n" + "="*60)
        for g_name, teams in self.groups.items():
            for t1, t2 in itertools.combinations(teams, 2):
                self.play_match(t1, t2, g_name, llm, is_knockout=False)

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

    def play_match(self, t1_name, t2_name, stage_name, llm, is_knockout=True):
        a1, a2 = self.world.agents[t1_name], self.world.agents[t2_name]

        # 0. REST/RECOVERY between fixtures (smooth by stage pressure)
        pressure = self._stage_pressure(stage_name, is_knockout)
        rest_units = 1.1 + 1.25 * pressure
        a1.recover(rest_units=rest_units)
        a2.recover(rest_units=rest_units)
        
        # 1. COACHING (Dynamic Game Theory)
        t1_tactics_json = llm.coach_decide_tactics(t1_name, a1.get_context_for_llm(), t2_name, a2.get_context_for_llm())
        t2_tactics_json = llm.coach_decide_tactics(t2_name, a2.get_context_for_llm(), t1_name, a1.get_context_for_llm())
        
        try:
            t1_data = json.loads(t1_tactics_json)
            a1.formation = t1_data.get("formation", a1.formation)
            a1.set_tactical_controls(t1_data.get("controls", {}))
            a1.apply_beliefs_to_tactics(stage_name=stage_name, opponent_style=a2.style_archetype)
            a1.coach_intervention(t1_tactics_json)
            print(f"  [TACTICS] {t1_name}: {t1_data.get('reasoning', 'No reasoning')}")
            
            t2_data = json.loads(t2_tactics_json)
            a2.formation = t2_data.get("formation", a2.formation)
            a2.set_tactical_controls(t2_data.get("controls", {}))
            a2.apply_beliefs_to_tactics(stage_name=stage_name, opponent_style=a1.style_archetype)
            a2.coach_intervention(t2_tactics_json)
            print(f"  [TACTICS] {t2_name}: {t2_data.get('reasoning', 'No reasoning')}")
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

        # 2. MATH-FIRST MATCH SIMULATION + TACTICAL MATCHUP
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
        eff_status_1 = a1.get_effective_status(matchup_bonus=matchup_bonus_1 + fused_1["status_delta"])
        eff_status_2 = a2.get_effective_status(matchup_bonus=matchup_bonus_2 + fused_2["status_delta"])
        # Tactical risk affects execution variance and shot-quality volatility.
        eff_status_1 *= max(0.72, 1.0 + np.random.normal(0.0, 0.035 * fused_1["volatility"]))
        eff_status_2 *= max(0.72, 1.0 + np.random.normal(0.0, 0.035 * fused_2["volatility"]))
        s1, s2, xg1, xg2 = simulate_match_score(eff_status_1, eff_status_2, is_knockout=is_knockout)

        pen_note = ""
        winner_name = None
        pen1 = pen2 = None
        aet1 = aet2 = 0
        et_xg1 = et_xg2 = 0.0
        went_to_extra_time = False
        if is_knockout and s1 == s2:
            went_to_extra_time = True
            aet1, aet2, et_xg1, et_xg2 = simulate_extra_time_score(eff_status_1, eff_status_2)
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
        verdict_json = json.dumps(
            {
                "winner": winner_name,
                "score": f"{s1}-{s2}",
                "key_event": key_event,
                "drama_score": round(drama_score, 2),
                "model": "poisson_math_first",
            },
            ensure_ascii=False,
        )
        print(
            f"  [MODEL] {t1_name} {s1}-{s2} {t2_name} | xG {xg1}-{xg2} | "
            f"EffStatus {eff_status_1:.1f}-{eff_status_2:.1f} | Matchup {matchup_bonus_1:+.1f}"
        )
        if went_to_extra_time:
            print(
                f"  [AET] {t1_name} {s1}-{s2} {t2_name} after extra time "
                f"(ET xG +{et_xg1:.2f}/+{et_xg2:.2f})."
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
        }
        if xg_context_line:
            facts_ledger["xg_alignment_note"] = xg_context_line
        facts_ledger["regulation_score"] = f"{s1-aet1}-{s2-aet2}" if went_to_extra_time else f"{s1}-{s2}"
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
        
        # 7. CASCADE — continuous locker tension; "explosion" headline is rare
        for agent, label in ((a1, t1_name), (a2, t2_name)):
            tense = agent.locker_room_tension()
            print(f"  [LOCKER_ROOM] {label} tension={tense:.2f}")
            if agent.rare_locker_room_explosion():
                print(f"  [EMERGENCY] {label} Locker Room Explosion!")

        return winner_name or (t1_name if s1 >= s2 else t2_name)

    def update_standings(self, g, t1, t2, s1, s2):
        st = self.standings[g]
        st[t1]["pts"] += (3 if s1 > s2 else (1 if s1 == s2 else 0))
        st[t2]["pts"] += (3 if s2 > s1 else (1 if s1 == s2 else 0))
        st[t1]["gf"] += s1; st[t1]["ga"] += s2; st[t1]["gd"] = st[t1]["gf"] - st[t1]["ga"]
        st[t2]["gf"] += s2; st[t2]["ga"] += s1; st[t2]["gd"] = st[t2]["gf"] - st[t2]["ga"]

    def resolve_advancements(self):
        thirds = []
        for g_name, teams_stats in self.standings.items():
            sorted_teams = sorted(teams_stats.keys(), key=lambda x: (teams_stats[x]['pts'], teams_stats[x]['gd'], teams_stats[x]['gf']), reverse=True)
            self.qualified_teams.append(sorted_teams[0])
            self.qualified_teams.append(sorted_teams[1])
            thirds.append((g_name, sorted_teams[2], teams_stats[sorted_teams[2]]))
        thirds.sort(key=lambda x: (x[2]['pts'], x[2]['gd'], x[2]['gf']), reverse=True)
        for i in range(8): self.qualified_teams.append(thirds[i][1])

    def simulate_knockout_round(self, round_name, num_teams, llm):
        print(f"\n" + "="*60 + f"\n🏆 {round_name.upper()}\n" + "="*60)
        next_round = []
        current_batch = self.qualified_teams[:num_teams]
        for i in range(0, len(current_batch), 2):
            winner = self.play_match(current_batch[i], current_batch[i+1], round_name, llm, is_knockout=True)
            next_round.append(winner)
            if round_name == "Final":
                runner_up = current_batch[i] if winner == current_batch[i+1] else current_batch[i+1]
                self.final_result = {
                    "champion": winner,
                    "runner_up": runner_up,
                }
            self.world.agents[winner].perform_reflection(llm)
        self.qualified_teams = next_round
