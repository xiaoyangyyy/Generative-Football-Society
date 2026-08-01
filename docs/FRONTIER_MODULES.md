# Frontier Modules and Validation State

## Implemented

- Immutable `SimulationConfig` plus atomic run manifests with data/model hashes,
  code revision, root seed, and seed-derivation version.
- Matched-seed counterfactual evaluator reporting paired ATE, standard error,
  confidence interval, and directional support.
- Transactional meta-learning proposals with shadow state, evaluation gate,
  commit conflict detection, rejection, and rollback.
- Memory provenance with causal parents, contradiction links, validity window,
  retrieval counts, and delayed downstream utility attribution.
- Continuous-time competing-risk event clock with covariate-modulated hazards.
- Coach/unit/player hierarchical tactical options behind an experiment flag.
- Dynamic player interaction graphs and a v6 graph-message world-model encoder.
- Active-sampling queue combining ensemble uncertainty, observation coverage,
  event rarity, target deficit, and planner advantage.

## Authority Gates

Implementation does not imply authority. Experimental modules must pass these
gates before they may change official tournament outcomes:

1. A matched-seed ablation must show a beneficial confidence interval.
2. Output distributions must remain inside calibration tolerances.
3. Deterministic replay and checkpoint round-trip tests must pass.
4. A learned model must satisfy grouped validation and OOD uncertainty tests.
5. A new score path must retain singular ownership of official goals.

`enable_hierarchical_policy` and `enable_continuous_event_clock` therefore
default to false. Their algorithms and tests are present, but coefficients need
event-data fitting before activation.

## World Model Deployment

The deployed `data/world_model/latent_wm.pt` is now a trained v6 dynamic-graph
checkpoint. The recorder contract was repaired so every action tick is retained
with its real outcome label. Collection produced 17,580 transitions across 98
match groups, including 527 shots, 160 shots on target, and 38 goals.

On the same grouped holdout, v5 improved weighted observation MSE from 0.01595
to 0.01330, pass balanced accuracy from 0.500 to 0.555, and shot Brier from
0.13382 to 0.08413 relative to v4. Independent replay over 400 transitions gave
MSE 0.00513. v6 pass planning is authorized at branch quality 0.2115 after
sealed BA/AUC/Brier/ECE gates. Shot planning remains disabled.

## Self-Validation Coverage

Tests cover paired-noise cancellation, event hazard statistics, hierarchy role
relevance, meta shadow/commit/rollback semantics, memory provenance and utility,
graph shapes/no-self-edge invariants, active-sampling priority, immutable
manifests, legacy checkpoint loading, and v7 transition-ensemble checkpoint
round trips.
The current v6 completion branch adds a bounded residual around the physics
success prior and is validated by BA, AUC, Brier, and ECE on sealed matches.
