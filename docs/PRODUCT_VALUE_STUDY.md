# Comparative product-value study

This preregistered study asks a narrower and more useful question than a generic
satisfaction survey: does GFS Studio help target users reach evidence-correct
football-analysis decisions faster than their current manual workflow without
reducing correctness or creating critical errors?

The design is a randomized, within-participant crossover with two matched case
packs. Sequence `AB` uses GFS Studio on alpha and the manual baseline on beta;
sequence `BA` reverses the condition order and case assignment. Each participant
receives a non-measured training task first. The frozen minimum is 24 valid
participants: at least eight in each target role and at least twelve in each
sequence. Optional stopping and performance-based exclusion are forbidden.

Incorrect or incomplete work receives the full 900-second penalty. The primary
endpoint is the geometric mean of each participant's GFS-to-baseline penalized
time ratio. Its deterministic participant-bootstrap 95% upper bound must be at
most 0.80. In addition, the lower bootstrap bound for the paired accuracy
difference must be at least -0.05 and GFS must produce zero critical errors.
These constraints prevent a superficially faster but less trustworthy workflow
from passing.

## Frozen cases and allocation

The concrete alpha and beta cases, answer choices and numeric thresholds are
frozen in `data/evaluation/product_value_case_packs_v1.json`. That participant
authority physically contains no scoring key. Moderator-only answers are held
in `data/evaluation/product_value_scoring_seal_v1.json`; the seal is bound to
the exact case-manifest SHA-256 and is consumed only by final analysis. The
protocol audit independently recomputes the sole threshold-passing
intervention from branch values before accepting the seal. The cases are
synthetic simulator-local decision problems and cannot support a real-football
causal claim.

Participants are allocated separately inside each target role using frozen
permuted blocks of four containing two `AB` and two `BA` assignments. At eight
valid participants per role this guarantees four of each sequence per role and
twelve of each sequence overall. The moderator audits aggregate readiness,
enrolls a consented pseudonymous participant and materializes the
identity-bound blinded packet through Studio:

```bash
python gfs.py --base-dir . studio value-study status
python gfs.py --base-dir . studio value-study register \
  --participant-id participant-replace0001 \
  --target-role football_analyst \
  --moderator-id moderator-replace0001 \
  --confirm-consent
python gfs.py --base-dir . studio value-study packet \
  --registration-id REG-REPLACE_WITH_ASSIGNED_ID
```

The returned sequence and `registration_id` are authoritative. They are bound
to the protocol hash, case-pack hash, role slot, consent state and registration
time. Exact retries are idempotent; role, moderator, sequence or time drift is
rejected.

The authenticated Web surface exposes the same boundary at
`GET /api/v1/studies/product-value`,
`POST /api/v1/studies/product-value/registrations`, and
`GET /api/v1/studies/product-value/packets/{registration_id}.json`. Mutation is
CSRF protected. Status is aggregate-only, packet downloads omit moderator
identity and scoring material, and telemetry collapses registration IDs into a
route template. Output materialization is confined to
`build/study-delivery/product-value-v1/`, rejects symbolic-link components and
uses atomic no-overwrite creation by default.

The repository is not a participant-delivery channel. A real study moderator
must expose only the generated packet and condition UI, keep repository/source
access and the scoring seal outside the participant session, and preserve the
original packet digest with the session evidence.

Analysis records are JSON Lines with an exact allowlist. Participant and
moderator IDs are pseudonymous; raw text and direct identifiers are forbidden.
Each completed condition requires an evidence SHA-256. Exclusions are limited to
withdrawn consent or infrastructure failure before the first measured condition,
and excluded records contain no behavioral measurements.

The controlled evidence archive is a flat content-addressed directory. A
completed condition's digest names a JSON receipt with exact protocol,
registration, participant, condition and case identities, three structured
answers and a submission timestamp inside the session window. During analysis
the file is read once, that byte snapshot is hashed, the same verified bytes
are parsed, and `correct` is recomputed from the frozen key. The frozen 64 KiB
receipt limit bounds memory use. Missing, reused,
symlinked or modified evidence and observer-declared correctness disagreement
all fail closed. The archive location is never written to the decision.

Audit the unexecuted protocol with:

```bash
python scripts/product_value_study.py
```

After independently collecting the frozen records, analyze them with:

```bash
python scripts/product_value_study.py --analyze \
  --records data/evaluation/product_value_validation_v1/participant_records.jsonl \
  --registry data/evaluation/product_value_validation_v1/session_registry.json \
  --evidence-root /controlled/archive/product-value-v1 \
  --scoring-seal data/evaluation/product_value_scoring_seal_v1.json \
  --out data/evaluation/product_value_validation_v1/decision.json
```

The analyzer makes no provider call, runs no match, observes no participant, and
does not overwrite an existing decision unless `--overwrite` is explicit. A
failed study is a valid result and does not satisfy the unified release gate.

The registration/allocation/evidence amendment and the later physical
participant-key separation amendment were both frozen while the authority
still recorded zero participants, measured sessions and results. Registration
is not measurement; once a records file exists, live Studio status stops
claiming zero observations and closes further registration pending analysis.
