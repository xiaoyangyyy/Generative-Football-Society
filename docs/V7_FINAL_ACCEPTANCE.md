# V7 Final Acceptance

V7 is frozen and deployable only when the machine-readable release gates, fixed-seed regression, SkillCorner tracking validation, and strict all-provider LODO report all pass. V6 remains immutable and is the explicit rollback target.

## Final evidence

- Joint pass completion: BA 0.7440, AUC 0.8035, Brier 0.1220, ECE 0.0116.
- Joint receiver holdout: Top-1 0.3810, Top-3 0.8400, MRR 0.6193.
- Strict receiver LODO MRR: Metrica 0.5322, SkillCorner 0.6871, Sportec 0.5080.
- Shot probability: AUC 0.7484, Brier 0.0922 versus 0.1072 baseline.
- Probabilistic transitions: log loss 0.4381 versus 0.5710 global baseline.
- SkillCorner: 10 matches and 642,687 tracking frames; adapter quality gate passed.
- Fixed regression: 80 Monte Carlo tournaments and four fast micro matches; determinism, distribution, finite-value, and p95 latency gates passed.

Sportec completion LODO remains evidence-insufficient because only four failed passes contain intended-receiver labels. It is excluded by an explicit minimum-outcome gate, not counted as a pass or failure.

## Release controls

- Frozen manifest: `data/releases/v7.0.0.json`
- Lifecycle registry: `data/releases/model_registry.json`
- Active/rollback pointer: `data/releases/current.json`
- V6 rollback manifest: `data/releases/v6.0.0.json`

Sealed artifacts must not be overwritten. Further research starts a new candidate version.
