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

The repository is not a participant-delivery channel. The isolated runner now
enforces that boundary instead of leaving it as an operator convention. First
register the full frozen sample before importing any measurement record. Then
provision each session into an absolute directory outside the repository:

```bash
python gfs.py --base-dir . studio value-study provision \
  --registration-id REG-REPLACE_WITH_ASSIGNED_ID \
  --session-root /controlled/sessions/product-value-v1 \
  --origin https://study.example
python gfs.py --base-dir . studio value-study serve \
  --session-file /controlled/sessions/product-value-v1/SES-REPLACE.json \
  --evidence-root /controlled/archive/product-value-v1 \
  --host 127.0.0.1 --port 8876
```

Provisioning returns a one-time launch URL whose capability is carried in the
URL fragment, never sent in an HTTP referrer, and stored only as a SHA-256 in
session state. The participant server is a separate WSGI application: it has
no Studio, registration, packet-download, scoring or repository route. Remote
operation requires an explicit `--allow-remote` Host allowlist and HTTPS at a
trusted reverse proxy on the same host. The participant service itself remains
loopback-bound and accepts forwarded HTTPS only from a loopback peer, so a
remote client cannot convert plaintext HTTP into trusted transport by forging
`X-Forwarded-Proto`. The page uses a per-response CSP nonce and the API uses
the high-entropy capability as a Bearer credential.

The unscored tutorial must be completed before measurement. A condition's
content remains hidden until its server-side 900-second clock starts. Only the
current condition is returned; the second case, condition name, allocation
sequence, participant pseudonym and moderator pseudonym are omitted. GFS is
rendered as an integrated provenance, threshold and limitation workflow while
the baseline is a flat read-only case. The first valid structured submission
wins, retries with the same submission ID are idempotent, timeouts advance
without fabricating a receipt, and correctness is never disclosed in-session.

Receipts and the completed record draft remain in controlled external storage.
The runner cannot write to the repository. After observing the completed
session, the moderator explicitly attests and imports it:

```bash
python gfs.py --base-dir . studio value-study import \
  --session-file /controlled/sessions/product-value-v1/SES-REPLACE.json \
  --evidence-root /controlled/archive/product-value-v1 \
  --attest-observed-session
```

Import revalidates the registration, protocol and case hashes, condition order,
server timestamps, durations, structured choices, critical-claim flag, exact
receipt bytes and content-addressed names before atomically appending the raw
record. It does not expose or use the scoring seal; final analysis remains the
only moderator scoring boundary. An exact repeated import is idempotent and a
conflicting import fails closed.

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

The registration/allocation/evidence amendment, physical participant-key
separation amendment and isolated-session-runner amendment were all frozen
while the authority still recorded zero participants, measured sessions and
results. Registration and provisioning are not measurement; once an attested
record is imported, live Studio status stops claiming zero observations and
closes further registration pending analysis. No session has been provisioned
or executed by the repository verification reported here.
