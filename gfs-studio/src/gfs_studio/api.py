from pathlib import Path

from gfs_studio.bootstrap.application import StudioApplication


def m2_status(project_root: str | Path) -> dict:
    return StudioApplication(project_root).m2_status()


def release_status(project_root: str | Path) -> dict:
    return StudioApplication(project_root).release_status()


def evaluate_current_simulation(project_root: str | Path, *, fixtures=None, sample_indices=None, fast: bool = False) -> dict:
    return StudioApplication(project_root).evaluate_current_simulation(
        fixtures=fixtures, sample_indices=sample_indices, fast=fast,
    )


def world_model_status(project_root: str | Path, checkpoint: str = "latent_wm_rollout_calibrated_candidate.pt") -> dict:
    return StudioApplication(project_root).world_model_status(checkpoint)


def imagine(project_root: str | Path, observation, action, *, checkpoint: str = "latent_wm_rollout_calibrated_candidate.pt", steps: int | None = None):
    return StudioApplication(project_root).imagine(observation, action, checkpoint=checkpoint, steps=steps)


def train(project_root: str | Path, trace: str | Path, run_name: str, *, epochs: int = 1, resume: bool = False, run_seed: int = 42):
    return StudioApplication(project_root).train(
        trace, run_name, epochs=epochs, resume=resume, run_seed=run_seed,
    )


def evaluate_rollout(project_root: str | Path, trace: str | Path, checkpoint: str, *, horizons=(1, 2, 4)) -> dict:
    return StudioApplication(project_root).evaluate_rollout(
        trace, checkpoint, horizons=horizons,
    )


def export_traces(project_root: str | Path, output: str | Path, *, home: str = "Brazil", away: str = "Argentina", seed: int = 42, fast: bool = True, stop_at_tick: int | None = None):
    return StudioApplication(project_root).export_traces(
        output, home=home, away=away, seed=seed, fast=fast, stop_at_tick=stop_at_tick,
    )


def export_cohort(project_root: str | Path, specifications: list[dict], output_dir: str | Path, *, fast: bool = True, stop_at_tick: int | None = None, holdout_teams: list[str] | None = None, validation_time_fraction: float = 0.2) -> dict:
    return StudioApplication(project_root).export_cohort(
        specifications,
        output_dir,
        fast=fast,
        stop_at_tick=stop_at_tick,
        holdout_teams=holdout_teams,
        validation_time_fraction=validation_time_fraction,
    )


def cognition_status(project_root: str | Path) -> dict:
    return StudioApplication(project_root).cognition_status()


def reflect(project_root: str | Path, *, role: str, topic: str, match_id: str, seed: int = 42):
    return StudioApplication(project_root).reflect(role=role, topic=topic, match_id=match_id, seed=seed)


def web_gateway(project_root: str | Path):
    return StudioApplication(project_root).web_gateway()


def backup_create(project_root: str | Path) -> dict:
    return StudioApplication(project_root).backup_create()


def backup_verify(project_root: str | Path, backup_id: str) -> dict:
    return StudioApplication(project_root).backup_verify(backup_id)


def backup_restore(project_root: str | Path, backup_id: str, destination: str | Path, *, replace: bool = False) -> dict:
    return StudioApplication(project_root).backup_restore(backup_id, destination, replace=replace)


def submit_task(project_root: str | Path, *, home: str, away: str, seed: int, idempotency_key: str) -> dict:
    return StudioApplication(project_root).submit_task(home, away, seed, idempotency_key)


def task_get(project_root: str | Path, task_id: str) -> dict | None:
    return StudioApplication(project_root).task_get(task_id)


def task_list(project_root: str | Path) -> list[dict]:
    return StudioApplication(project_root).task_list()


def task_wait(project_root: str | Path, task_id: str, timeout: float = 30.0) -> dict:
    return StudioApplication(project_root).task_wait(task_id, timeout)


def task_recover(project_root: str | Path) -> int:
    return StudioApplication(project_root).task_recover()


def task_resume(project_root: str | Path, task_id: str | None = None) -> int:
    return StudioApplication(project_root).task_resume(task_id)


def plan(project_root: str | Path, observation, action, *, steps: int | None = None, kind: str = "general") -> dict:
    return StudioApplication(project_root).plan(observation, action, steps=steps, kind=kind)


def recommend(project_root: str | Path, observation, action, *, baseline_action: str = "pass") -> dict:
    return StudioApplication(project_root).recommend(observation, action, baseline_action=baseline_action)


def frame_manifest(project_root: str | Path) -> dict:
    return StudioApplication(project_root).frame_manifest()
