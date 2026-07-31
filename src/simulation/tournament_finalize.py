"""Post-match state updates, physical wear, memory, and checkpointing."""

import numpy as np

from src.simulation.match_pipeline import finalize_match_feedback, micro_layer_enabled


class TournamentFinalizeMixin:
    @staticmethod
    def _match_result_labels(s1, s2):
        res_1 = "win" if s1 > s2 else ("loss" if s2 > s1 else "draw")
        res_2 = "win" if s2 > s1 else ("loss" if s1 > s2 else "draw")
        return res_1, res_2

    def _update_post_match_agents(
        self, *, a1, a2, t1_name, t2_name, s1, s2,
        prof_score, social_chaos, referee, drama_score,
    ):
        res_1, res_2 = self._match_result_labels(s1, s2)
        a1.recursive_update(
            res_1, s1 - s2, prof_score, social_chaos,
            t2_name, a2.status_score,
        )
        a2.recursive_update(
            res_2, s2 - s1, prof_score, social_chaos,
            t1_name, a1.status_score,
        )
        a1.relax_referee_grievance_post_match(
            res_1, referee["bias_t1"], drama_score,
        )
        a2.relax_referee_grievance_post_match(
            res_2, referee["bias_t2"], drama_score,
        )
        return res_1, res_2

    @staticmethod
    def _apply_physical_wear(
        *, a1, a2, t1_name, t2_name, pressure, drama_score,
        tactical_1, tactical_2,
    ):
        wear_intensity = 0.24 + 0.36 * pressure + 0.06 * drama_score
        pre_fatigue_a, pre_fatigue_b = a1.fatigue, a2.fatigue
        injury_a = a1.apply_match_wear(
            intensity=wear_intensity * tactical_1["fatigue_load_multiplier"],
        )
        injury_b = a2.apply_match_wear(
            intensity=wear_intensity * tactical_2["fatigue_load_multiplier"],
        )
        for agent, label, _ in (
            (a1, t1_name, injury_a),
            (a2, t2_name, injury_b),
        ):
            print(f"  [MEDICAL] {label} injury_load={agent.injury_load:.2f}")
            p_report = float(
                1.0 / (1.0 + np.exp(-10.0 * (float(agent.injury_load) - 0.62)))
            )
            if np.random.random() < p_report:
                print(f"  [MEDICAL] {label} reported injury concerns.")
        return pre_fatigue_a, pre_fatigue_b

    @staticmethod
    def _record_match_decisions(
        *, a1, a2, t1_name, t2_name, stage_name, s1, s2, xg1, xg2,
        social_chaos, res_1, res_2, pressure, pre_fatigue_a, pre_fatigue_b,
    ):
        a1.record_decision_event(
            opponent=t2_name, stage_name=stage_name,
            controls=a1.tactical_controls,
            outcomes={
                "goal_diff": s1 - s2, "xg_for": xg1, "xg_against": xg2,
                "fatigue_delta": a1.fatigue - pre_fatigue_a,
                "chaos": social_chaos, "result": res_1,
            },
            opponent_style=a2.style_archetype, stage_pressure=pressure,
        )
        a2.record_decision_event(
            opponent=t1_name, stage_name=stage_name,
            controls=a2.tactical_controls,
            outcomes={
                "goal_diff": s2 - s1, "xg_for": xg2, "xg_against": xg1,
                "fatigue_delta": a2.fatigue - pre_fatigue_b,
                "chaos": social_chaos, "result": res_2,
            },
            opponent_style=a1.style_archetype, stage_pressure=pressure,
        )

    @staticmethod
    def _report_locker_room_state(*, a1, a2, t1_name, t2_name):
        for agent, label in ((a1, t1_name), (a2, t2_name)):
            tension = agent.locker_room_tension()
            print(f"  [LOCKER_ROOM] {label} tension={tension:.2f}")
            if agent.rare_locker_room_explosion():
                print(f"  [EMERGENCY] {label} Locker Room Explosion!")

    def _commit_match_result(
        self, *, stage_name, t1_name, t2_name, s1, s2, winner_name,
    ):
        winner = winner_name or (t1_name if s1 >= s2 else t2_name)
        match_key = self._match_key(stage_name, t1_name, t2_name)
        self.completed_matches.append(match_key)
        self.match_results[match_key] = winner
        self.match_index += 1
        self._save_checkpoint()
        return winner

    def _finalize_match_state(
        self,
        *,
        a1, a2, t1_name, t2_name, stage_name,
        s1, s2, xg1, xg2, winner_name, drama_score,
        prof_score, social_chaos, pressure, referee,
        internal_1, internal_2, ref_1, ref_2,
        tactical_1, tactical_2, micro_summary,
    ):
        res_1, res_2 = self._update_post_match_agents(
            a1=a1, a2=a2, t1_name=t1_name, t2_name=t2_name,
            s1=s1, s2=s2, prof_score=prof_score, social_chaos=social_chaos,
            referee=referee, drama_score=drama_score,
        )
        pre_fatigue_a, pre_fatigue_b = self._apply_physical_wear(
            a1=a1, a2=a2, t1_name=t1_name, t2_name=t2_name,
            pressure=pressure, drama_score=drama_score,
            tactical_1=tactical_1, tactical_2=tactical_2,
        )
        self._record_match_decisions(
            a1=a1, a2=a2, t1_name=t1_name, t2_name=t2_name,
            stage_name=stage_name, s1=s1, s2=s2, xg1=xg1, xg2=xg2,
            social_chaos=social_chaos, res_1=res_1, res_2=res_2,
            pressure=pressure, pre_fatigue_a=pre_fatigue_a,
            pre_fatigue_b=pre_fatigue_b,
        )
        finalize_match_feedback(
            a1, a2, base_dir=self.base_dir, result_home=res_1,
            result_away=res_2, score_diff_home=s1 - s2,
            xg_home=xg1, xg_away=xg2, prof_score=prof_score,
            social_chaos=social_chaos, stage_name=stage_name,
            micro_summary=micro_summary if micro_layer_enabled() else None,
        )
        self._report_locker_room_state(
            a1=a1, a2=a2, t1_name=t1_name, t2_name=t2_name,
        )
        return self._commit_match_result(
            stage_name=stage_name, t1_name=t1_name, t2_name=t2_name,
            s1=s1, s2=s2, winner_name=winner_name,
        )
