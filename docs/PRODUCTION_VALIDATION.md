# Production validation

This stage is a prospective, code-controlled release gate. It is not satisfied
by the existing 100-task control-plane test because that test intentionally runs
no football matches. The frozen protocol is
`data/evaluation/production_validation_protocol_v1.json`.

The workload is exactly 100 stable, fast matches: ten fixed fixtures over ten
rounds with seeds 91000 through 91099. It authorizes neither training nor model
provider calls. A normal invocation is read-only and only audits the protocol:

```bash
python scripts/run_production_validation.py
```

Execution requires a fresh dedicated Studio initialized in stable mode with
seed 91000, an exact authorization token, and a pseudonymous deployment instance
identifier. Run it inside the deployed container with the Studio data and output
directories on the named persistent volume. During one running match, terminate
the first container without a graceful worker shutdown. Start a second container
against the same volume and rerun the same command with a different instance ID.
The persistent task keeps its logical identity and is explicitly requeued.

Before the resumed run can produce a decision, an operator must place
`data/evaluation/production_validation_v1/deployment_attestation.json` on that
volume. Its exact schema is enforced by the runner and contains only a
pseudonymous operator ID, named-volume ID, pinned container image digest, the two
or more pseudonymous instance IDs, timestamp, fixed attestation sentence, and
boolean restart/privacy claims. Raw host names, IP addresses, and personal names
are forbidden.

The gate passes only when all 100 logical tasks and 100 unique match reports are
complete, every report accepts its integrity checks, provider calls remain zero,
at least one running task was recovered, the deployment attestation matches the
progress journal, and an integrity-checked backup restores an identical session
and match inventory into a clean staging root. Failed and incomplete outcomes
remain valid negative evidence; the runner does not silently retry failed tasks.

This protocol has been registered but not executed. Do not claim the production
operations gate until its generated decision exists and passes the unified
release control plane.
