# Formal Experiment Protocol V2

This protocol freezes the next confirmatory paper experiment without running
it. Earlier 18-match M1 results remain calibration evidence and retain the
`research_only_default_off` decision; they are not relabeled as a new
confirmatory study.

## One question, two frozen arms

The experiment asks whether the sealed M1 world-model planner improves the
external calibration loss of frozen M0 by a meaningful amount while preserving
every validity gate and changing observable match behavior. No additional
model, cognition layer, prompt, or ablation is part of this experiment.

The immutable machine-readable registration is
`data/evaluation/formal_experiment_protocol_v2.json`. It binds the exact M1
checkpoint SHA-256 and the critical simulation, calibration, objective, and
runner code files.

## Fixed compute budget

- Six fixed fixtures.
- Five deterministic matched seeds per fixture.
- Thirty paired experimental units.
- Thirty runs per arm; sixty runs total.
- Full 90-minute simulated duration.
- No interim analysis, optional stopping, or post-hoc fixture/metric selection.
- Integrity failures may only rerun the same arm, fixture, and sample index.

The experiment is resumable, but resume fails closed after protocol,
checkpoint, or critical-code drift.

## Preregistered inference

There is one confirmatory primary estimand: M1 minus M0 external calibration
loss. Uncertainty uses a 10,000-draw paired bootstrap stratified by fixture, so
each resample preserves both pairing and the frozen fixture composition.

Promotion requires all of the following:

1. M1 passes every external-validity gate.
2. The upper 95% bootstrap bound is below `-0.10`, the frozen minimum
   meaningful loss improvement.
3. At least 10% of matched pairs change an observable outcome above the frozen
   numerical tolerance.
4. Protocol, checkpoint, and critical-code identities still match.

The fifteen secondary metrics are descriptive only and cannot trigger
promotion. If the complete interval lies inside `[-0.10, 0.10]`, the result is
classified as no meaningful difference. Every other non-promoting outcome is
inconclusive and remains research-only.

## Safe commands

Read-only status and identity preflight:

```bash
python scripts/run_formal_experiment.py
```

Run or resume the fixed experiment:

```bash
python scripts/run_formal_experiment.py --execute --authorization I_AUTHORIZE_GFS_FORMAL_EXPERIMENT_V2
```

Analyze only after all sixty runs are present:

```bash
python scripts/run_formal_experiment.py --analyze --authorization I_AUTHORIZE_GFS_FORMAL_EXPERIMENT_V2
```

Neither status nor analysis can start simulation. Analysis refuses partial,
unpaired, or identity-mismatched evidence.

## V4.06 ledger and authority hardening

The runner now validates the exact frozen fixture/sample prefix, M0-then-M1 arm
order, all calibration inputs consumed by the loss, bounded action-mechanism
counts and exact arm/overall completion before resume or analysis. M0 rows with
world-model action activity fail closed. Execution identity also binds the
observable contract, StatsBomb match baselines and joint baselines.

`--execute` cannot start simulations without the exact authorization phrase.
The CLI form of `--analyze` requires the same phrase because it writes the
decision artifact; pure status and independent verification remain read-only.
Structurally valid historical rows keep their historical identity and cannot
be relabelled as current evidence.

## Paper package

The registered-report draft, claim registry, model/data cards, environment
limitations, and independent review instructions are audited without running
the experiment:

```bash
python scripts/verify_paper_package.py --out data/evaluation/paper_package_verification_v1.json
```

The audit must report `confirmatory_result_available: false` until the fixed
experiment and complete-only analysis have actually produced the registered
decision artifact.
