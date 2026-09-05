"""Identity-bound tournament checkpoint persistence and world restoration."""

from src.simulation.tournament_checkpoint import (
    checkpoint_root_seed,
    checkpoint_run_identity,
    restore_r32_fixtures,
    save_checkpoint,
    validate_reflection_journal,
)
from src.simulation.world_state import (
    preflight_world_state,
    restore_world_state,
    snapshot_world_state,
)


class TournamentStateMixin:
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

    def _require_run_identity(self) -> None:
        value = self.run_identity_sha256
        if not isinstance(value, str) or len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value
        ):
            raise RuntimeError(
                "Full tournament execution requires a verified run identity"
            )
