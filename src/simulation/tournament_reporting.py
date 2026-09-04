"""Match reporting, LLM narrative, media, and post-match social stages."""

import json


from src.memory_engine.poisson_simulator import (
    finalize_stage_xg_context,
    score_xg_anomaly_note,
)
from src.simulation.match_pipeline import print_micro_match_logs, run_micro_layer
from src.simulation.score_path import score_path_label
from src.simulation.runtime import environment_snapshot, env_bool
from src.simulation.random_control import named_rng


class TournamentReportingMixin:
    def _build_result_payload(
        self, *, a1, a2, t1_name, t2_name, is_knockout,
        referee, eff_status_1, eff_status_2, matchup_bonus_1,
        score_path, physics_first, xg_prior_h, xg_prior_a,
        s1, s2, xg1, xg2, winner_name, pen_note, pen1, pen2,
        aet1, aet2, et_xg1, et_xg2, went_to_extra_time, reg_s1, reg_s2,
    ):
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
        return drama_score, key_event, verdict_json


    def _build_xg_context(
        self, *, t1_name, t2_name, stage_name, s1, s2, xg1, xg2, pen1,
    ):
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
        return xg_context_line


    def _audit_match_debug(
        self, *, ah, aa, t1_name, t2_name, stage_name,
        home_micro, away_micro, s1, s2, xg1, xg2,
        micro_summary, xg_context_line,
    ):
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


    def _run_report_replay(
        self, *, a1, a2, t1_name, t2_name, stage_name,
        s1, s2, xg1, xg2, referee, pressure, drama_score,
        internal_1, internal_2, seed, micro_replay_mode, micro_summary,
    ):
        aff_on = env_bool(environment_snapshot(), "MATCH_AFFECTIVE", False)
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
                        base_dir=self.base_dir,
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
                        base_dir=self.base_dir,
                    )
                    print(
                        f"  [AFFECTIVE] ψ={aff.final_psi:+.2f} | coach stress {aff.home_coach_stress:.2f}/{aff.away_coach_stress:.2f} "
                        f"| ref strict {aff.ref_strictness_mean:.2f} | drift {aff.tactical_drift_home:.2f}/{aff.tactical_drift_away:.2f}"
                    )
            except Exception as exc:
                print(f"  [MICRO/AFFECTIVE] skipped: {exc}")
        return micro_summary


    def _report_match_result(
        self, *, a1, a2, ah, aa, t1_name, t2_name, stage_name,
        is_knockout, home_micro, away_micro, pressure, referee,
        internal_1, internal_2, eff_status_1, eff_status_2,
        matchup_bonus_1, score_path, physics_first, micro_replay_mode,
        micro_summary, seed, xg_prior_h, xg_prior_a,
        s1, s2, xg1, xg2, winner_name, pen_note, pen1, pen2,
        aet1, aet2, et_xg1, et_xg2, went_to_extra_time, reg_s1, reg_s2,
    ):
        drama_score, key_event, verdict_json = self._build_result_payload(
            a1=a1, a2=a2, t1_name=t1_name, t2_name=t2_name,
            is_knockout=is_knockout, referee=referee,
            eff_status_1=eff_status_1, eff_status_2=eff_status_2,
            matchup_bonus_1=matchup_bonus_1, score_path=score_path,
            physics_first=physics_first, xg_prior_h=xg_prior_h,
            xg_prior_a=xg_prior_a, s1=s1, s2=s2, xg1=xg1, xg2=xg2,
            winner_name=winner_name, pen_note=pen_note, pen1=pen1, pen2=pen2,
            aet1=aet1, aet2=aet2, et_xg1=et_xg1, et_xg2=et_xg2,
            went_to_extra_time=went_to_extra_time, reg_s1=reg_s1, reg_s2=reg_s2,
        )
        xg_context_line = self._build_xg_context(
            t1_name=t1_name, t2_name=t2_name, stage_name=stage_name,
            s1=s1, s2=s2, xg1=xg1, xg2=xg2, pen1=pen1,
        )

        self._audit_match_debug(
            ah=ah, aa=aa, t1_name=t1_name, t2_name=t2_name,
            stage_name=stage_name, home_micro=home_micro,
            away_micro=away_micro, s1=s1, s2=s2, xg1=xg1, xg2=xg2,
            micro_summary=micro_summary, xg_context_line=xg_context_line,
        )

        micro_summary = self._run_report_replay(
            a1=a1, a2=a2, t1_name=t1_name, t2_name=t2_name,
            stage_name=stage_name, s1=s1, s2=s2, xg1=xg1, xg2=xg2,
            referee=referee, pressure=pressure, drama_score=drama_score,
            internal_1=internal_1, internal_2=internal_2, seed=seed,
            micro_replay_mode=micro_replay_mode, micro_summary=micro_summary,
        )
        return {
            "drama_score": drama_score, "key_event": key_event,
            "verdict_json": verdict_json, "xg_context_line": xg_context_line,
            "micro_summary": micro_summary,
        }

    def _build_facts_ledger(
        self, *, a1, a2, t1_name, t2_name, stage_name,
        s1, s2, xg1, xg2, winner_name, drama_score,
        eff_status_1, eff_status_2, micro_summary, xg_context_line,
        reg_s1, reg_s2, went_to_extra_time, pen1, pen2,
    ):
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
        return facts_ledger


    def _apply_narrative_atmosphere(
        self, *, llm, a1, a2, t1_name, t2_name, stage_name, verdict_json,
    ):
        narrative_json = llm.predict_match_result(
            t1_name, a1.get_context_for_llm(), t2_name, a2.get_context_for_llm(), stage_name
        )
        atmosphere_chaos = 0.0
        try:
            narrative = json.loads(narrative_json)
            if isinstance(narrative, dict) and narrative.get("key_event"):
                verdict_data = json.loads(verdict_json)
                verdict_data["llm_key_event"] = narrative.get("key_event")
                verdict_data["narrative_signals"] = narrative.get("signals", {})
                verdict_json = json.dumps(verdict_data, ensure_ascii=False)
                event_record = self.narrative_event_bus.publish(
                    narrative,
                    [a1, a2],
                    stage=stage_name,
                )
                atmosphere_chaos = sum(
                    float(item.get("feedback", {}).get("chaos_delta", 0.0))
                    for item in event_record.get("applied", {}).values()
                )
        except Exception:
            pass
        return verdict_json, atmosphere_chaos


    def _publish_media_matrix(
        self, *, llm, a1, a2, t1_name, t2_name, stage_name,
        verdict_json, facts_ledger, drama_score, ref_1, ref_2,
        fused_1, fused_2, atmosphere_chaos,
    ):
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
                + 0.35 * atmosphere_chaos
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
        return prof_score, social_chaos


    def _run_post_match_dialogue(
        self, *, a1, a2, t1_name, t2_name, stage_name,
        winner_name, s1, s2, key_event, drama_score, pressure,
        referee, micro_summary, social_chaos, match_seed,
    ):
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
            rng=named_rng(match_seed, "social_dialogue"),
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
        return social_chaos


    def _run_narrative_and_social(
        self,
        *,
        llm, a1, a2, t1_name, t2_name, stage_name, is_knockout,
        s1, s2, xg1, xg2, winner_name, drama_score,
        eff_status_1, eff_status_2, micro_summary, xg_context_line,
        reg_s1, reg_s2, went_to_extra_time, pen1, pen2,
        verdict_json, key_event, pressure, referee, ref_1, ref_2,
        fused_1, fused_2, match_seed,
    ):
        facts_ledger = self._build_facts_ledger(
            a1=a1, a2=a2, t1_name=t1_name, t2_name=t2_name,
            stage_name=stage_name, s1=s1, s2=s2, xg1=xg1, xg2=xg2,
            winner_name=winner_name, drama_score=drama_score,
            eff_status_1=eff_status_1, eff_status_2=eff_status_2,
            micro_summary=micro_summary, xg_context_line=xg_context_line,
            reg_s1=reg_s1, reg_s2=reg_s2,
            went_to_extra_time=went_to_extra_time, pen1=pen1, pen2=pen2,
        )

        verdict_json, atmosphere_chaos = self._apply_narrative_atmosphere(
            llm=llm, a1=a1, a2=a2, t1_name=t1_name, t2_name=t2_name,
            stage_name=stage_name, verdict_json=verdict_json,
        )

        if not is_knockout: self.update_standings(stage_name, t1_name, t2_name, s1, s2)

        prof_score, social_chaos = self._publish_media_matrix(
            llm=llm, a1=a1, a2=a2, t1_name=t1_name, t2_name=t2_name,
            stage_name=stage_name, verdict_json=verdict_json,
            facts_ledger=facts_ledger, drama_score=drama_score,
            ref_1=ref_1, ref_2=ref_2, fused_1=fused_1, fused_2=fused_2,
            atmosphere_chaos=atmosphere_chaos,
        )

        social_chaos = self._run_post_match_dialogue(
            a1=a1, a2=a2, t1_name=t1_name, t2_name=t2_name,
            stage_name=stage_name, winner_name=winner_name,
            s1=s1, s2=s2, key_event=key_event, drama_score=drama_score,
            pressure=pressure, referee=referee, micro_summary=micro_summary,
            social_chaos=social_chaos,
            match_seed=match_seed,
        )
        
        return prof_score, social_chaos
