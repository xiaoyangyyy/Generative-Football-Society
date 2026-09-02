"""Single-match execution extracted from the tournament orchestrator."""

from src.simulation.tournament_finalize import TournamentFinalizeMixin
from src.simulation.tournament_reporting import TournamentReportingMixin
from src.simulation.tournament_scoring import TournamentScoringMixin
from src.simulation.tournament_setup import TournamentSetupMixin
from src.simulation.random_control import derive_seed


class TournamentMatchMixin(
    TournamentSetupMixin,
    TournamentScoringMixin,
    TournamentReportingMixin,
    TournamentFinalizeMixin,
):
    def _simulate_match(
        self,
        *,
        a1,
        a2,
        ah,
        aa,
        t1_name,
        t2_name,
        stage_name,
        is_knockout,
        home_micro,
        away_micro,
        neutral_venue,
        pressure,
        internal_1,
        internal_2,
        referee,
        ref_1,
        ref_2,
        match_seed,
    ):
        fused_context = self._build_fused_match_context(
            a1=a1, a2=a2, t1_name=t1_name, t2_name=t2_name,
            home_micro=home_micro, pressure=pressure, referee=referee,
            internal_1=internal_1, internal_2=internal_2, ref_1=ref_1, ref_2=ref_2,
            match_seed=match_seed,
        )
        matchup_bonus_1 = fused_context["matchup_bonus_1"]
        tactical_1, tactical_2 = fused_context["tactical_1"], fused_context["tactical_2"]
        fused_1, fused_2 = fused_context["fused_1"], fused_context["fused_2"]
        eff_status_1, eff_status_2 = fused_context["eff_status_1"], fused_context["eff_status_2"]
        eff_micro_home, eff_micro_away = fused_context["eff_micro_home"], fused_context["eff_micro_away"]
        internal_micro_h, internal_micro_a = fused_context["internal_micro_h"], fused_context["internal_micro_a"]
        fused_vol_h, fused_vol_a = fused_context["fused_vol_h"], fused_context["fused_vol_a"]
        regulation = self._resolve_regulation_score(
            a1=a1, a2=a2, ah=ah, aa=aa, t1_name=t1_name, t2_name=t2_name,
            stage_name=stage_name, is_knockout=is_knockout,
            home_micro=home_micro, away_micro=away_micro,
            neutral_venue=neutral_venue, pressure=pressure, referee=referee,
            internal_micro_h=internal_micro_h, internal_micro_a=internal_micro_a,
            eff_status_1=eff_status_1, eff_status_2=eff_status_2,
            eff_micro_home=eff_micro_home, eff_micro_away=eff_micro_away,
            fused_1=fused_1, fused_2=fused_2,
            fused_vol_h=fused_vol_h, fused_vol_a=fused_vol_a,
            match_seed=match_seed,
        )
        s1, s2, xg1, xg2 = regulation["s1"], regulation["s2"], regulation["xg1"], regulation["xg2"]
        score_path, physics_first = regulation["score_path"], regulation["physics_first"]
        micro_replay_mode, micro_summary = regulation["micro_replay_mode"], regulation["micro_summary"]
        seed = regulation["seed"]
        xg_prior_h, xg_prior_a = regulation["xg_prior_h"], regulation["xg_prior_a"]
        drama_pre = regulation["drama_pre"]
        knockout = self._resolve_knockout_score(
            a1=a1, a2=a2, ah=ah, aa=aa, t1_name=t1_name, t2_name=t2_name,
            stage_name=stage_name, is_knockout=is_knockout,
            home_micro=home_micro, away_micro=away_micro,
            neutral_venue=neutral_venue, pressure=pressure, referee=referee,
            internal_micro_h=internal_micro_h, internal_micro_a=internal_micro_a,
            eff_status_1=eff_status_1, eff_status_2=eff_status_2,
            eff_micro_home=eff_micro_home, eff_micro_away=eff_micro_away,
            physics_first=physics_first, xg_prior_h=xg_prior_h,
            xg_prior_a=xg_prior_a, drama_pre=drama_pre, seed=seed, s1=s1, s2=s2,
        )
        s1, s2 = knockout["s1"], knockout["s2"]
        winner_name, pen_note = knockout["winner_name"], knockout["pen_note"]
        pen1, pen2 = knockout["pen1"], knockout["pen2"]
        aet1, aet2 = knockout["aet1"], knockout["aet2"]
        et_xg1, et_xg2 = knockout["et_xg1"], knockout["et_xg2"]
        went_to_extra_time = knockout["went_to_extra_time"]
        reg_s1, reg_s2 = knockout["reg_s1"], knockout["reg_s2"]

        report = self._report_match_result(
            a1=a1, a2=a2, ah=ah, aa=aa, t1_name=t1_name, t2_name=t2_name,
            stage_name=stage_name, is_knockout=is_knockout,
            home_micro=home_micro, away_micro=away_micro, pressure=pressure,
            referee=referee, internal_1=internal_1, internal_2=internal_2,
            eff_status_1=eff_status_1, eff_status_2=eff_status_2,
            matchup_bonus_1=matchup_bonus_1, score_path=score_path,
            physics_first=physics_first, micro_replay_mode=micro_replay_mode,
            micro_summary=micro_summary, seed=seed,
            xg_prior_h=xg_prior_h, xg_prior_a=xg_prior_a,
            s1=s1, s2=s2, xg1=xg1, xg2=xg2, winner_name=winner_name,
            pen_note=pen_note, pen1=pen1, pen2=pen2, aet1=aet1, aet2=aet2,
            et_xg1=et_xg1, et_xg2=et_xg2,
            went_to_extra_time=went_to_extra_time, reg_s1=reg_s1, reg_s2=reg_s2,
        )
        drama_score, key_event = report["drama_score"], report["key_event"]
        verdict_json, xg_context_line = report["verdict_json"], report["xg_context_line"]
        micro_summary = report["micro_summary"]

        return {
            "s1": s1, "s2": s2, "xg1": xg1, "xg2": xg2,
            "winner_name": winner_name, "drama_score": drama_score,
            "key_event": key_event, "verdict_json": verdict_json,
            "xg_context_line": xg_context_line, "micro_summary": micro_summary,
            "reg_s1": reg_s1, "reg_s2": reg_s2,
            "went_to_extra_time": went_to_extra_time,
            "pen1": pen1, "pen2": pen2,
            "tactical_1": tactical_1, "tactical_2": tactical_2,
            "fused_1": fused_1, "fused_2": fused_2,
            "eff_status_1": eff_status_1, "eff_status_2": eff_status_2,
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
        match_seed = derive_seed(
            getattr(self, "root_seed", 42),
            "tournament_match", fixture_seed, stage_name, t1_name, t2_name,
        )
        (
            a1, a2, ah, aa, home_micro, away_micro, neutral_venue,
            pressure, internal_1, internal_2, referee, ref_1, ref_2,
        ) = self._prepare_match(
            t1_name,
            t2_name,
            stage_name,
            llm,
            is_knockout,
            matchday=matchday,
            standings_snapshot=standings_snapshot,
            fixture_seed=fixture_seed,
            scheduled_home=scheduled_home,
            match_seed=match_seed,
        )
        match = self._simulate_match(
            a1=a1, a2=a2, ah=ah, aa=aa,
            t1_name=t1_name, t2_name=t2_name, stage_name=stage_name,
            is_knockout=is_knockout, home_micro=home_micro,
            away_micro=away_micro, neutral_venue=neutral_venue,
            pressure=pressure, internal_1=internal_1, internal_2=internal_2,
            referee=referee, ref_1=ref_1, ref_2=ref_2,
            match_seed=match_seed,
        )
        s1, s2, xg1, xg2 = match["s1"], match["s2"], match["xg1"], match["xg2"]
        winner_name, drama_score = match["winner_name"], match["drama_score"]
        key_event, verdict_json = match["key_event"], match["verdict_json"]
        xg_context_line, micro_summary = match["xg_context_line"], match["micro_summary"]
        reg_s1, reg_s2 = match["reg_s1"], match["reg_s2"]
        went_to_extra_time, pen1, pen2 = match["went_to_extra_time"], match["pen1"], match["pen2"]
        tactical_1, tactical_2 = match["tactical_1"], match["tactical_2"]
        fused_1, fused_2 = match["fused_1"], match["fused_2"]
        eff_status_1, eff_status_2 = match["eff_status_1"], match["eff_status_2"]
        prof_score, social_chaos = self._run_narrative_and_social(
            llm=llm, a1=a1, a2=a2, t1_name=t1_name, t2_name=t2_name,
            stage_name=stage_name, is_knockout=is_knockout,
            s1=s1, s2=s2, xg1=xg1, xg2=xg2, winner_name=winner_name,
            drama_score=drama_score, eff_status_1=eff_status_1,
            eff_status_2=eff_status_2, micro_summary=micro_summary,
            xg_context_line=xg_context_line, reg_s1=reg_s1, reg_s2=reg_s2,
            went_to_extra_time=went_to_extra_time, pen1=pen1, pen2=pen2,
            verdict_json=verdict_json, key_event=key_event, pressure=pressure,
            referee=referee, ref_1=ref_1, ref_2=ref_2,
            fused_1=fused_1, fused_2=fused_2,
            match_seed=match_seed,
        )
        return self._finalize_match_state(
            a1=a1, a2=a2, t1_name=t1_name, t2_name=t2_name,
            stage_name=stage_name, s1=s1, s2=s2, xg1=xg1, xg2=xg2,
            winner_name=winner_name, drama_score=drama_score,
            prof_score=prof_score, social_chaos=social_chaos,
            pressure=pressure, referee=referee, internal_1=internal_1,
            internal_2=internal_2, ref_1=ref_1, ref_2=ref_2,
            tactical_1=tactical_1, tactical_2=tactical_2,
            micro_summary=micro_summary,
            match_seed=match_seed,
        )
