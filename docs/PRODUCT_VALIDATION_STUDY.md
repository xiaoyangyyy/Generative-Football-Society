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

## Record format

Store one JSON object per line at the protocol's `raw_records` path. Each
record must contain:

- `schema_version: 1` and the exact `protocol_id`;
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

The evidence digest should identify a redacted screen recording, task event
export, match report, or backup verification artifact held in the controlled
study archive. The repository record contains the digest, not participant
media.

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
  --external-review data/evaluation/product_validation_v1/external_review.json \
  --out data/evaluation/product_validation_v1/decision.json
```

The analyzer never launches Studio or a match. It fails closed unless sample,
role coverage, task success, SUS, critical-error, independent-review, artifact
hash, and issue-resolution gates all pass.
