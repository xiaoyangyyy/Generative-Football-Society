"""Full-tournament scheduling, advancement, and reflection lifecycle."""

import json
import time

from src.simulation.group_context import snapshot_standings_table
from src.simulation.knockout_bracket import (
    build_qualified_entries,
    build_r32_pairings,
)
from src.simulation.random_control import derive_seed, named_py_rng
from src.simulation.standings import apply_group_result
from src.simulation.tournament_checkpoint import (
    load_checkpoint,
    validate_reflection_journal,
    verify_state_artifacts,
)


class TournamentLifecycleMixin:
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

    def run_full_tournament(
        self, *, resume: bool = False, checkpoint: dict | None = None,
    ):
        from src.match_engine.calibration.narrative_isolation import resolve_tournament_llm

        self._require_run_identity()
        llm = resolve_tournament_llm()
        if resume:
            ckpt = checkpoint
            if ckpt is None:
                ckpt = load_checkpoint(self.base_dir)
            else:
                # The application parsed and validated this payload before
                # identity-gated recovery; require the restored files to match.
                verify_state_artifacts(self.base_dir, ckpt)
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
        print("\n" + "="*60 + f"\n🏆 {round_name.upper()}\n" + "="*60)
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
