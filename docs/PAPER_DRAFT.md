# Evidence-Gated World-Model Planning in a Reproducible Football Society Simulator

> MANUSCRIPT_STAGE: REGISTERED_REPORT_DRAFT
>
> CONFIRMATORY_RESULT_STATUS: NOT_EXECUTED
>
> INDEPENDENT_REPRODUCTION_STATUS: NOT_PERFORMED

## Abstract

Generative Football Society (GFS) is a deterministic, evidence-gated football
simulation and research platform that separates a frozen stable simulator from
default-off world-model and language-model research layers. This registered-
report-stage manuscript defines the system boundary, summarizes existing
repository evidence, and preregisters a confirmatory comparison between frozen
M0 and a sealed M1 world-model planner. Existing evidence is not presented as
the confirmatory study: a prior 18-match sealed M1 review found no measurable
behavior change and no resolved external-calibration improvement, so M1
remains research-only. The new protocol fixes six fixtures, five seeds per
fixture, 30 matched pairs, 60 total runs, one primary estimand, a fixture-
stratified paired bootstrap, and fail-closed code/checkpoint identity. At this
stage no confirmatory estimate, interval, decision, or independent
reproduction exists. The contribution is therefore an auditable system and
analysis contract, not evidence that M1 improves football simulation.

## 1. Introduction

Football simulation research often combines event rules, learned components,
tracking-derived features, and stochastic match outcomes. This creates two
recurring risks: a promising research component can silently become part of a
stable product, and exploratory evaluations can be rewritten as confirmatory
evidence after their outcomes are known. GFS addresses these risks by placing
release identity, model identity, experiment identity, and claim scope behind
machine-checked boundaries.

The paper asks one bounded question: does the sealed M1 world-model planner
improve external calibration loss over frozen M0 by at least the
preregistered meaningful amount, while preserving all validity gates and
changing observable match behavior? The answer is deliberately absent until
the fixed experiment is authorized and completed.

The present contributions are:

1. a layered simulator/research boundary with integrity-checked stable and
   research artifacts;
2. a multi-provider evidence inventory with explicit provider and claim
   limitations;
3. a sealed, paired, compute-bounded confirmatory protocol with no optional
   stopping; and
4. a machine-readable claim registry that prevents existing or pending
   evidence from being described outside its support.

## 2. Related work

GFS draws on world-model research that separates learned environmental
dynamics from downstream control [@ha2018worldmodels], and on probabilistic
calibration practice that evaluates confidence quality rather than accuracy
alone [@guo2017calibration]. Its football observations are derived from
provider-specific public or research datasets, including StatsBomb Open Data
[@statsbombopendata], SkillCorner Open Data [@skillcorneropendata], Metrica
Sports sample data [@metricasampledata], and Sportec IDSSE records. Dataset
availability does not imply common licensing or interchangeable observation
semantics; the data card records those boundaries.

Unlike a benchmark paper that selects the best result after broad model
search, this manuscript freezes one M0-versus-M1 comparison. Earlier
calibration and ablation work remains useful context but cannot determine the
new confirmatory decision.

## 3. System and methods

### 3.1 System boundary

<!-- claim:CLM-001 -->
GFS separates a frozen stable simulator from default-off research world-model
and cognition layers. The active release pointer, release manifest, and
non-empty release artifacts are verified as one identity chain. Research
checkpoint presence alone cannot alter the stable default.

M0 is the frozen baseline pipeline. M1 adds the sealed calibrated world-model
planner through an explicit preset and checkpoint identity. The planner can
propose behavior only through existing action and quality gates; a prediction
artifact is not itself evidence that simulated actions changed.

### 3.2 External evidence and observation scope

<!-- claim:CLM-002 -->
The frozen evidence report covers a 128 team-match StatsBomb benchmark and
derived observations from SkillCorner, StatsBomb 360, Sportec IDSSE, and
Metrica sources. These sources differ in sampling, event semantics, tracking
coverage, class balance, and licensing metadata. Provider-isolated evaluation
is therefore retained where possible, and insufficient provider evidence is
reported rather than pooled away.

### 3.3 Existing M0 evidence

<!-- claim:CLM-003 -->
Before the new registration, M0 passed the repository external calibration
contract on 18 full-length simulations across six fixtures and three seeds.
This establishes a frozen baseline for repository-internal evaluation. It is
not a comparison under the new five-seed confirmatory design.

### 3.4 Existing M1 negative evidence

<!-- claim:CLM-004 -->
A prior sealed 18-match M1 review produced external-calibration loss
8.6341489501 versus 8.6341489511 for M0. Its paired delta interval crossed
zero, no registered observable changed above tolerance, and the run consumed
8,550.7 seconds. The resulting decision was
`research_only_default_off`. These are existing negative results, not the
outcome of the new confirmatory experiment.

### 3.5 Confirmatory design

<!-- claim:CLM-005 -->
The preregistration fixes six fixtures and five deterministic sample indices
per fixture, producing 30 matched fixture-seed pairs and 60 total runs. Each
run simulates 90 minutes. M0 always precedes M1 for a deterministic pair, and
arm order is not a tunable factor.

The primary estimand is candidate-minus-baseline external calibration loss.
Uncertainty is a 10,000-draw paired bootstrap stratified by fixture. Promotion
requires all external validity gates, an upper 95% bound below `-0.10`, at
least 10% behavior-changing pairs, and unchanged protocol, checkpoint, and
critical-code identities. Fifteen secondary metrics are descriptive only.
Integrity failures may rerun only the same arm, fixture, and sample index.

## 4. Existing evidence

The following table reports evidence created before this registered protocol.
It must not be merged with the pending confirmatory results.

| Evidence | Units | Recorded outcome | Claim scope |
|---|---:|---|---|
| Stable release integrity | 16 artifacts | identity chain valid | repository integrity |
| M0 frozen evaluation | 18 simulations | all external gates passed | prior baseline |
| Sealed M1 review | 18 matched simulations | no resolved loss or behavior improvement | prior negative result |
| Pass evidence | SkillCorner and StatsBomb 360 sufficient; Sportec completion labels insufficient | provider-specific | predictive evidence only |
| Live LLM benefit | 0 prospective provider treatment/control logs | unavailable | no benefit claim |

## 5. Confirmatory results

**Status: NOT EXECUTED.** The result cells below are intentionally null. They
may be populated only from a complete, identity-matched
`formal_confirmatory_v2/decision.json` generated by the preregistered
analyzer.

| Confirmatory quantity | Value |
|---|---|
| Completed M0 runs | not available |
| Completed M1 runs | not available |
| Point delta in external loss | not available |
| 95% paired interval | not available |
| Changed-pair fraction | not available |
| Validity gates | not available |
| Decision | not available |
| Promotion supported | not available |

No wording elsewhere in this manuscript should be read as filling these
cells.

## 6. Release interpretation

<!-- claim:CLM-006 -->
The active stable release remains 7.0.0. Neither the existing M1 negative
review nor the preregistration changes the production pointer. Even a
confirmatory promotion-candidate result would still require a separate
release review; statistical evidence is necessary but not sufficient for
deployment.

## 7. Discussion

The prior M1 result suggests that predictive machinery can fail to influence
the action path even when its checkpoint and inference contracts are valid.
The new protocol distinguishes three outcomes. A promotion-candidate result
supports further mechanism replication and release review. An equivalence
result redirects work toward prediction-to-action adoption. An inconclusive
result requires variance diagnosis before any new sample registration.

Negative evidence is treated as a product result: M1 remains default-off when
benefit is unresolved. This reduces the incentive to reinterpret a technically
complex component as successful merely because it executes.

## 8. Limitations, validity, and ethics

<!-- claim:CLM-007 -->
All effects in this paper are simulator-internal. The frozen research scope
forbids causal claims about real teams, players, coaching decisions, or social
behavior. External datasets have different clocks, labels, and selection
processes. The source-specific release review approves only the DFL-authorized
CC BY 4.0 IDSSE/Sportec-derived supplement, with attribution and a change
notice. StatsBomb, Metrica, SkillCorner, and Transfermarkt-derived files remain
excluded, so no blanket redistribution claim is made.

The environment contract pins the supported Python window, build backend,
container base version, and two 64-package transitive runtime closures: a
Python 3.12 x86_64 Linux CPU deployment profile and a Python 3.13 x86_64
Windows CPU development profile. Every exact package has SHA-256 evidence,
both uv and pip accept both install plans under `--require-hashes`, and two
deterministic CycloneDX 1.6 SBOMs cover the closures. The observed Windows
runtime imports all eight exact direct dependencies. Python and Caddy image
tags are bound to recorded official OCI index digests. Docker has not been
built in the present verification environment, and the two-profile matrix does
not claim every Python/OS combination permitted by the package metadata. The
formal design has 30 pairs; it is bounded
rather than universally powered. The bootstrap quantifies paired simulation
variability, not uncertainty over all football populations.

No live LLM benefit is claimed. Provider credentials are excluded from
artifacts, and the manuscript package verifier makes no network call.

## 9. Reproducibility and independent review

The reproduction manifest classifies commands by side effect. Package,
release-contract, evidence, and experiment-status audits are read-only. The
release-contract audit also verifies the two-profile transitive hash-lock
matrix, current SBOMs, immutable image pins, imported local runtime, and the
source-bounded licensed data supplement. Only an explicit `--execute` command can start the fixed
60-run experiment, and analysis fails closed until evidence is complete and
identity-matched.

The independent-reproduction handoff is also registered before results exist.
It requires a reviewer independent of implementation, a clean source commit,
the complete 60-run budget, current container and release evidence, exact
categorical decision agreement, scalar agreement within `1e-12`, disclosure
of every deviation, and a signed public identity or ORCID. A valid negative or
inconclusive reproduction remains reportable evidence and is not relabeled as
success. No such review has yet been performed.

<!-- claim:CLM-008 -->
No reproduction by a person independent of the implementer has been
completed. The independent-review checklist therefore remains an instruction,
not evidence. A complete paper requires the result table, a verified runtime
or container reproduction, dataset license review, and an external
reproduction report.

## 10. Artifact availability

The code, machine-readable protocol, claim registry, model card, data card,
direct dependency contract, source-license decision registry, environment
snapshot, and review commands are listed in
`data/evaluation/reproduction_manifest_v1.json`. Large or provider-governed
source data remain subject to their original terms. No archival DOI or
immutable reproduction archive is claimed at this stage.

## References

References are maintained in `docs/PAPER_REFERENCES.bib`. Repository and
dataset entries are access pointers; users must inspect the source terms
before redistribution.
