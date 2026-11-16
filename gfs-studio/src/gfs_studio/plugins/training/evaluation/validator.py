from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("Evaluation requires the ML extra: pip install 'gfs-studio[ml]'")


class CandidateEvaluator:
    @staticmethod
    def evaluate(model, dataset) -> dict:
        _require_torch()
        model_mse = _model_mse(model, dataset)
        persistence_mse = _persistence_mse(dataset)
        return {
            "groups": len(set(dataset.groups.tolist())),
            "rows": len(dataset.observations),
            "model_mse": model_mse,
            "persistence_mse": persistence_mse,
            "eligible": model_mse < persistence_mse,
            "scope": "transition_prediction_only_not_policy_or_deployment",
        }

    @staticmethod
    def evaluate_rollout(model, dataset, codec, *, horizons=(1, 2, 4)) -> dict:
        _require_torch()
        horizons = sorted(set(horizons))
        if not horizons or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in horizons):
            raise ValueError("rollout horizons must be positive integers")
        model.eval()
        results = {}
        for horizon in horizons:
            model_losses = []
            persistence_losses = []
            drifts = []
            for group in sorted(set(dataset.groups.tolist())):
                indices = np.flatnonzero(dataset.groups == group).tolist()
                if dataset.ticks is not None:
                    indices.sort(key=lambda index: int(dataset.ticks[index]))
                for start in range(0, len(indices) - horizon + 1):
                    selected = indices[start:start + horizon]
                    initial = dataset.observations[selected[0]]
                    predicted = initial.copy()
                    with torch.no_grad():
                        for index in selected:
                            action = codec.strip_action(dataset.actions[index])
                            prediction = model(
                                torch.from_numpy(predicted[None, :]),
                                torch.from_numpy(np.asarray(action, dtype=np.float32)[None, :]),
                            )[1]
                            predicted = prediction[0].detach().cpu().numpy()
                    target = dataset.targets[selected[-1]]
                    model_losses.append(float(np.mean((predicted - target) ** 2)))
                    persistence_losses.append(float(np.mean((initial - target) ** 2)))
                    drifts.append(float(np.sqrt(np.mean((predicted - initial) ** 2))))
            results[str(horizon)] = {
                "samples": len(model_losses),
                "model_mse": float(np.mean(model_losses)) if model_losses else None,
                "persistence_mse": float(np.mean(persistence_losses)) if persistence_losses else None,
                "mean_drift": float(np.mean(drifts)) if drifts else None,
                "finite": bool(model_losses) and bool(np.isfinite(model_losses).all()),
            }
        return {
            "horizons": results,
            "scope": "multi_step_transition_prediction_not_policy_or_deployment",
            "eligible_for_deployment": False,
            "groups": len(set(dataset.groups.tolist())),
            "cross_match_sequences": False,
        }


def _model_mse(model, dataset) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(dataset.observations), 128):
            observations = torch.from_numpy(dataset.observations[start:start + 128])
            actions = torch.from_numpy(dataset.actions[start:start + 128])
            targets = torch.from_numpy(dataset.targets[start:start + 128])
            predictions = model(observations, actions)[1]
            losses.append(float(torch.mean((predictions - targets) ** 2)))
    return float(np.mean(losses))


def _persistence_mse(dataset) -> float:
    return float(np.mean((dataset.observations - dataset.targets) ** 2))
