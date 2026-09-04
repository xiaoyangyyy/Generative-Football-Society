"""Regulation, micro-physics, extra-time, and penalty score resolution."""

from src.memory_engine.macro_goal_dynamics import (
    clamp_match_xg,
    expected_match_xg,
    simulate_match_score_dynamics,
)
from src.memory_engine.macro_micro_fusion import resolve_unified_score
from src.memory_engine.poisson_simulator import simulate_penalty_shootout
from src.simulation.match_pipeline import (
    print_micro_match_logs,
    run_extra_time_micro,
    run_physics_first_micro,
)
from src.simulation.score_path import (
    ScorePathMode,
    finalize_official_score_from_micro,
    resolve_score_path_mode,
    validate_official_xg_prior,
)
from src.simulation.random_control import named_rng


class PhysicsOfficialScoreError(RuntimeError):
    """A selected physics-official score path failed before score commit."""


class TournamentScoringMixin:
    def _build_regulation_context(
        self, *, stage_name, t1_name, t2_name, referee, match_seed,
    ):
        score_path = resolve_score_path_mode()
        seed = int(match_seed)
        return {
            "score_path": score_path,
            "physics_first": score_path == ScorePathMode.PHYSICS_OFFICIAL,
            "micro_replay_mode": score_path == ScorePathMode.MICRO_REPLAY,
            "legacy_poisson": score_path == ScorePathMode.POISSON_LEGACY,
            "drama_pre": min(1.0, 0.25 + 0.15 * referee["strictness"]),
            "seed": seed,
            "rng_score": named_rng(seed, "score"),
        }

    def _run_physics_official_score(
        self, *, ah, aa, t1_name, t2_name, stage_name, is_knockout,
        home_micro, away_micro, neutral_venue, pressure, referee,
        internal_micro_h, internal_micro_a, eff_micro_home, eff_micro_away,
        fused_vol_h, fused_vol_a, seed, rng_score, drama_pre,
    ):
        try:
            xg_prior_h, xg_prior_a, _ = expected_match_xg(
                ah, aa, eff_micro_home, eff_micro_away,
                is_knockout=is_knockout,
                stage_pressure=pressure,
                fused_volatility_home=fused_vol_h,
                fused_volatility_away=fused_vol_a,
                rng=rng_score,
            )
            xg_prior_h = clamp_match_xg(
                validate_official_xg_prior(
                    xg_prior_h, "macro_xg_prior_home"
                )
            )
            xg_prior_a = clamp_match_xg(
                validate_official_xg_prior(
                    xg_prior_a, "macro_xg_prior_away"
                )
            )
            micro_summary = run_physics_first_micro(
                ah, aa, xg_prior_home=xg_prior_h, xg_prior_away=xg_prior_a,
                eff_status_home=eff_micro_home, eff_status_away=eff_micro_away,
                referee=referee, stage_pressure=pressure, drama_score=drama_pre,
                internal_home=internal_micro_h, internal_away=internal_micro_a,
                seed=seed, stage_name=stage_name, neutral_venue=neutral_venue,
                base_dir=self.base_dir,
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
            return s1, s2, xg1, xg2, micro_summary, xg_prior_h, xg_prior_a
        except Exception as exc:
            raise PhysicsOfficialScoreError(
                "Physics-official regulation failed before score commit "
                f"({t1_name} vs {t2_name}, {stage_name}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _run_macro_regulation_score(
        self, *, a1, a2, eff_status_1, eff_status_2, is_knockout,
        pressure, fused_1, fused_2, micro_summary, rng_score, legacy_poisson,
    ):
        kwargs = {
            "is_knockout": is_knockout,
            "stage_pressure": pressure,
            "fused_volatility_home": float(fused_1.get("volatility", 0.0)),
            "fused_volatility_away": float(fused_2.get("volatility", 0.0)),
            "rng": rng_score,
        }
        if legacy_poisson:
            return simulate_match_score_dynamics(
                a1, a2, eff_status_1, eff_status_2, **kwargs,
            )
        return resolve_unified_score(
            a1, a2, eff_status_1, eff_status_2,
            micro_summary=micro_summary, **kwargs,
        )

    def _resolve_regulation_score(
        self, *, a1, a2, ah, aa, t1_name, t2_name, stage_name,
        is_knockout, home_micro, away_micro, neutral_venue, pressure,
        referee, internal_micro_h, internal_micro_a,
        eff_status_1, eff_status_2, eff_micro_home, eff_micro_away,
        fused_1, fused_2, fused_vol_h, fused_vol_a,
        match_seed,
    ):
        context = self._build_regulation_context(
            stage_name=stage_name, t1_name=t1_name, t2_name=t2_name,
            referee=referee,
            match_seed=match_seed,
        )
        micro_summary = None
        xg_prior_h, xg_prior_a = 1.0, 1.0

        if context["physics_first"]:
            physics_result = self._run_physics_official_score(
                ah=ah, aa=aa, t1_name=t1_name, t2_name=t2_name,
                stage_name=stage_name, is_knockout=is_knockout,
                home_micro=home_micro, away_micro=away_micro,
                neutral_venue=neutral_venue, pressure=pressure, referee=referee,
                internal_micro_h=internal_micro_h, internal_micro_a=internal_micro_a,
                eff_micro_home=eff_micro_home, eff_micro_away=eff_micro_away,
                fused_vol_h=fused_vol_h, fused_vol_a=fused_vol_a,
                seed=context["seed"], rng_score=context["rng_score"],
                drama_pre=context["drama_pre"],
            )
            s1, s2, xg1, xg2, micro_summary, xg_prior_h, xg_prior_a = (
                physics_result
            )

        if not context["physics_first"]:
            s1, s2, xg1, xg2, _ = self._run_macro_regulation_score(
                a1=a1, a2=a2, eff_status_1=eff_status_1,
                eff_status_2=eff_status_2, is_knockout=is_knockout,
                pressure=pressure, fused_1=fused_1, fused_2=fused_2,
                micro_summary=micro_summary, rng_score=context["rng_score"],
                legacy_poisson=context["legacy_poisson"],
            )

        return {
            "s1": s1, "s2": s2, "xg1": xg1, "xg2": xg2,
            "score_path": context["score_path"],
            "physics_first": context["physics_first"],
            "micro_replay_mode": context["micro_replay_mode"],
            "micro_summary": micro_summary, "seed": context["seed"],
            "xg_prior_h": xg_prior_h, "xg_prior_a": xg_prior_a,
            "drama_pre": context["drama_pre"],
        }


    def _resolve_knockout_score(
        self, *, a1, a2, ah, aa, t1_name, t2_name, stage_name,
        is_knockout, home_micro, away_micro, neutral_venue, pressure,
        referee, internal_micro_h, internal_micro_a,
        eff_status_1, eff_status_2, eff_micro_home, eff_micro_away,
        physics_first, xg_prior_h, xg_prior_a, drama_pre, seed, s1, s2,
    ):
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
                try:
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
                        base_dir=self.base_dir,
                    )
                    aet1, aet2, et_xg1, et_xg2, _ = (
                        finalize_official_score_from_micro(
                            et_summary,
                            home_micro=home_micro,
                            fixture_home=t1_name,
                            macro_xg_prior_home=xg_prior_h,
                            macro_xg_prior_away=xg_prior_a,
                        )
                    )
                    print_micro_match_logs(
                        et_summary, home_micro, away_micro, ah, aa
                    )
                except Exception as exc:
                    raise PhysicsOfficialScoreError(
                        "Physics-official extra time failed before score commit "
                        f"({t1_name} vs {t2_name}, {stage_name}): "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
            else:
                from src.memory_engine.macro_goal_dynamics import simulate_extra_time_dynamics

                aet1, aet2, et_xg1, et_xg2 = simulate_extra_time_dynamics(
                    a1, a2, eff_status_1, eff_status_2, stage_pressure=pressure,
                    rng=named_rng(seed, "extra_time"),
                )
            s1 += aet1
            s2 += aet2
            if s1 == s2:
                pen1, pen2 = simulate_penalty_shootout(named_rng(seed, "penalties"))
                winner_name = t1_name if pen1 > pen2 else t2_name
                pen_note = f" (Pens {pen1}-{pen2})"
            else:
                winner_name = t1_name if s1 > s2 else t2_name
        elif s1 > s2:
            winner_name = t1_name
        elif s2 > s1:
            winner_name = t2_name
        return {
            "s1": s1, "s2": s2, "winner_name": winner_name,
            "pen_note": pen_note, "pen1": pen1, "pen2": pen2,
            "aet1": aet1, "aet2": aet2,
            "et_xg1": et_xg1, "et_xg2": et_xg2,
            "went_to_extra_time": went_to_extra_time,
            "reg_s1": reg_s1, "reg_s2": reg_s2,
        }
