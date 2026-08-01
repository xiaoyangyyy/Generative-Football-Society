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
manifests, legacy checkpoint loading, and v8 transition/semantic-ensemble checkpoint
round trips.
Strategic fusion also keeps a match-local Bayesian posterior over opponent
tactical intent, runs each action under every supported opponent hypothesis,
and admits only bounded, observable-feature-grounded LLM pseudo-evidence into
that posterior. A checkpoint/environment-scoped cross-match meta-memory seeds a
weak observation-grounded prior without recursively training on its own prior,
while an online change detector releases stale history after a confirmed
tactical regime shift. LLM change explanations are directionally verified and
cannot control the detector. Decision logs retain the posterior,
counterfactual sensitivity, meta provenance, and regime audit without
pretending the latent intent has a direct ground-truth label.
Second-order planning learns action-conditioned opponent belief responses only
after chronological, match-clustered held-out improvement over a sticky
structural baseline. A two-ply belief-space planner selects one continuation
policy against that response distribution. The dynamics ensemble is trained on
changing-action, autoregressive two-step sequences after a one-step curriculum
warmup, with independent member bootstraps. When grouped two-step holdout error
beats persistence and ensemble disagreement tracks error, a budgeted search
re-encodes continuation actions from predicted states and blends them with the
current-state proxy under capped, calibration-reduced authority. Older or
unvalidated checkpoints use the proxy without speculative search. Bounded LLM
response scenarios can stress an
evaluated branch but cannot persist evidence, alter the learned memory, or claim
causal opponent reactions. Realized forecasts have a separate online readiness
gate and match-clustered Brier audit; predicted-state planning has an independent
budget/provenance readiness gate.
The LLM also emits optional semantic residual critiques tied to its selected
action and an evaluated horizon. These claims are scored against realized
policy utility, match-clustered, checkpoint/environment isolated, and receive
bounded ranking authority only after chronological held-out MSE improvement.
They never mutate the neural forecast or self-author their reliability. This
creates a falsifiable semantic-numeric correction channel instead of an
unverifiable LLM override.
Every horizon also exposes short, tactical, and strategic projections from all
dynamics members. The LLM may make one schema-limited event forecast over those
projections. Its probability and the neural member-frequency probability are
frozen, joined to the exact same later simulator outcome, and compared with
paired match-clustered Brier scores. This semantic event path remains
non-controlling even when its evaluation gate is ready.
V8 additionally trains member-aligned semantic event heads from realized
one- and two-step future states with proper BCE. Each event/depth must beat both
the training prevalence and transparent geometry projection under grouped
validation before receiving capped blending authority; unsupported longer
horizons remain projection-only.
The current v6 completion branch adds a bounded residual around the physics
success prior and is validated by BA, AUC, Brier, and ECE on sealed matches.
