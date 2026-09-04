# GFS Studio target-user validation

This is the operator guide for
`data/evaluation/product_validation_protocol_v1.json`. The protocol is
preregistered but has not been executed. Preparing or checking it does not
create participants, run matches, train a model, or call a provider.

## Study boundary

Recruit at least ten valid adult participants from the three frozen target
roles, with at least two participants per role. Obtain informed consent before
recording. Use pseudonymous participant and moderator IDs. Do not place names,
email addresses, organizations, IP addresses, raw interview text, credentials,
or local paths in the analysis record.

The moderator may clarify the task wording but must not control the interface.
Record every valid participant, including failures. Only consent withdrawal or
infrastructure failure before the first task permits exclusion. Performance is
never an exclusion reason and there is no optional stopping.

## Authoritative registration

Every participant must be registered after consent and before the measured
session. The command writes one pseudonymous, protocol-hash-bound registration
under an operating-system lease. An exact retry is idempotent; a retry that
changes role or moderator fails. Once the records file contains any entry, new
registration is refused.

```bash
python scripts/product_validation_study.py --register \
  --participant-id P-REPLACE00000 \
  --target-role football_analyst \
  --moderator-id M-REPLACE0 \
  --confirm-consent
```

Copy the returned `registration_id` into the participant record. The analyzer
requires one and only one record for every registry entry, verifies role and
moderator identity, and rejects a session start earlier than its hash-bound
registration time.

## Record format

Store one JSON object per line at the protocol's `raw_records` path. Each
record must contain:

- `schema_version: 1` and the exact `protocol_id`;
- the exact `registration_id` returned by the registration command;
- a unique ID matching `P-[A-Z0-9]{12}`;
- one frozen target role, consent, start/end RFC 3339 timestamps, and a
  pseudonymous moderator ID matching `M-[A-Z0-9]{8}`;
- exactly five task records in protocol order with `completed`,
  `critical_error`, positive `duration_seconds`, non-negative
  `assistance_count`, and a SHA-256 evidence digest for completed tasks;
- exactly ten integer SUS responses from 1 through 5;
- zero or more frozen qualitative tags, never raw free text;
- `excluded` and, only when excluded, one predeclared exclusion reason; and
- the exact observer attestation:
  `recorded contemporaneously under the frozen protocol`.

Unknown fields are rejected. An excluded record retains the pseudonymous IDs,
role, consent state, session timestamps, exclusion reason, and observer
attestation, but its `tasks`, `sus_responses`, and `qualitative_tags` arrays
must be empty. This prevents withdrawn or pre-task-failure sessions from
leaking behavioral data into analysis.

The evidence digest must identify a redacted screen recording, task event
export, match report, or backup verification artifact held in the controlled
study archive. The archive is a flat content-addressed directory: each file is
named by its exact lowercase SHA-256 with no extension. Symlinks, missing
files, content mismatches and reuse of one digest for multiple observations
are rejected. The repository record contains digests, not participant media or
the archive path.

## Independent external review

The external-review JSON must identify separate accessibility and security
reviewers, their organizations, signing times, methods, source-artifact
digests, pass decisions, and unresolved critical counts. Accessibility targets
WCAG 2.2 AA. Critical findings cannot be risk-accepted. The issue-resolution
section must enumerate totals and prove every critical issue is closed.

## Commands

Read-only protocol audit:

```bash
python scripts/product_validation_study.py
```

Analysis is explicit and reads existing records only:

```bash
python scripts/product_validation_study.py --analyze \
  --records data/evaluation/product_validation_v1/participant_records.jsonl \
  --registry data/evaluation/product_validation_v1/session_registry.json \
  --evidence-root /controlled/archive/product-validation-v1 \
  --external-review data/evaluation/product_validation_v1/external_review.json \
  --out data/evaluation/product_validation_v1/decision.json
```

The analyzer never launches Studio or a match. It fails closed unless registry
coverage, byte-level evidence identity, sample, role coverage, task success,
SUS, critical-error, independent-review and issue-resolution gates all pass.

The registry and byte-verification amendment was made with zero observed
participants and zero executed sessions. It does not retroactively modify any
result because no result existed.
