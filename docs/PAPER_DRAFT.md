# Evidence-Gated World-Model Planning in a Reproducible Football Society Simulator

> MANUSCRIPT_STAGE: COMPLETED_RESULTS_DRAFT
>
> CONFIRMATORY_RESULT_STATUS: COMPLETE_RESEARCH_ONLY
>
> INDEPENDENT_REPRODUCTION_STATUS: NOT_PERFORMED

## Abstract

Generative Football Society (GFS) is a deterministic, evidence-gated football
simulation and research platform that separates a frozen stable simulator from
default-off world-model and language-model research layers. This completed-
results draft defines the system boundary, separates historical evidence from
a frozen action-adoption mechanism study, and reports a preregistered
comparison between frozen M0 and a sealed M1 action-policy planner. The
mechanism study established that the learned signal can alter sampled
micro-actions. The downstream study completed 30 matched pairs and 60 runs:
all pairs changed behavior, but the external-loss delta was -0.0318925 with a
95% interval of [-26.0620, 5.1374]. The interval crossed zero, the external
validity and minimum-effect gates failed, and the sealed decision was
`inconclusive_keep_research_only`. M1 therefore remains default-off and no
independent reproduction or real-football causal benefit is claimed.

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
changing observable match behavior? The completed experiment answers only the
simulator-internal version of that question: behavior changed, while evidence
for a meaningful calibration improvement remained inconclusive.

The present contributions are:

1. a layered simulator/research boundary with integrity-checked stable and
   research artifacts;
2. a multi-provider evidence inventory with explicit provider and claim
   limitations;
3. a sealed, paired, compute-bounded confirmatory protocol with no optional
   stopping and deterministic result replay;
4. evidence that action-path influence and outcome efficacy are empirically
   distinct; and
5. a machine-readable claim registry that prevents historical, current, or
   unavailable evidence from being described outside its support.

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
All 60 registered runs completed. The sealed replay found an external-loss
delta of -0.03189252164278855 with a 95% interval of
[-26.061954645805613, 5.137429588662575]. All 30 pairs changed at least one
registered behavior metric, but the validity and minimum-effect gates failed;
the result is therefore `inconclusive_keep_research_only`, not a promotion
result.

## 4. Existing evidence

The following table reports historical evidence created before the current
action-policy study. It remains separate from the identity-bound result below.

| Evidence | Units | Recorded outcome | Claim scope |
|---|---:|---|---|
| Stable release integrity | 16 artifacts | identity chain valid | repository integrity |
| M0 frozen evaluation | 18 simulations | all external gates passed | prior baseline |
| Sealed M1 review | 18 matched simulations | no resolved loss or behavior improvement | prior negative result |
| Pass evidence | SkillCorner and StatsBomb 360 sufficient; Sportec completion labels insufficient | provider-specific | predictive evidence only |
| Live LLM benefit | 0 prospective provider treatment/control logs | unavailable | no benefit claim |

## 5. Confirmatory results

**Status: COMPLETE, RESEARCH-ONLY.** The current mechanism-confirmed
action-policy comparison completed under its frozen protocol and identity.
The authoritative decision is
`data/evaluation/action_outcome_v1/decision.json`, independently replayed by
`scripts/verify_action_outcome_result.py`.

| Confirmatory quantity | Value |
|---|---|
| Completed M0 runs | 30/30 |
| Completed M1 runs | 30/30 |
| Point delta in external loss | -0.03189252164278855 |
| 95% paired interval | [-26.061954645805613, 5.137429588662575] |
| Changed-pair fraction | 30/30 (1.0) |
| Validity gates | external validity fail; minimum-effect fail; behavior pass; identity pass |
| Decision | `inconclusive_keep_research_only` |
| Promotion supported | no |

The secondary metrics are descriptive only and cannot override this decision.

## 6. Release interpretation

<!-- claim:CLM-006 -->
The active stable release remains 7.0.0. The completed action-policy study did
not support promotion and therefore does not change the production pointer.
Any future promotion-candidate result would still require a separate release
review; statistical evidence is necessary but not sufficient for deployment.

## 7. Discussion

The mechanism study resolves the earlier structural failure: the learned
signal now changes sampled micro-actions, and the full-match study changed all
30 matched pairs. The downstream result remains inconclusive. Its interval is
wide, crosses zero and misses the frozen minimum-effect gate, while the
candidate also fails the external-validity gate. This separates action-path
success from outcome-efficacy evidence and requires variance and objective
diagnosis under a newly registered study before any further candidate claim.

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

The active V2 independent-reproduction handoff was registered after the
primary action-outcome result and before V2 external-replication results. It
requires a reviewer independent of implementation, a clean source commit,
reproduction of the complete 60-run primary budget followed by the fixed
72-run action-policy replication budget, current container and release evidence, exact
agreement for both categorical decisions, scalar agreement within `1e-12`,
disclosure of every deviation, and a signed public identity or ORCID. The
total independent budget is 132 simulations. A valid negative, failed, or
inconclusive result at either stage remains reportable evidence and is not
relabelled as success. No such review has yet been performed.

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
