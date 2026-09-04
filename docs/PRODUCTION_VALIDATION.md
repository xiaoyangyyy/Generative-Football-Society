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

The deployable three-phase transaction is recorded as the protocol's sole
pre-execution amendment. Its evidence states that zero matches and zero results
existed before the amendment; any second amendment or non-zero prior execution
causes the protocol audit to fail closed.

Execution uses the `production-validation` Compose profile. The service is
offline (`network_mode: none`), read-only, capability-free, and runs as the same
non-root image user as Studio. It has dedicated named volumes for output,
persistence, backups, validation evidence, and restore scratch. Always use the
dedicated Compose project name below: this prevents the prospective drill from
touching a live Studio deployment. The shared Compose file still requires the
normal deployment interpolation variables, so the operator's deployment env
file must define `GFS_WEB_HOST` and `GFS_WEB_ACCESS_TOKEN_FILE`; neither value is
mounted into or used by the validation service.

Build and initialize exactly one fresh stable workspace:

```bash
docker compose --env-file deploy/production.env \
  -p gfs-production-validation -f deploy/compose.yaml \
  --profile validation build production-validation

docker compose --env-file deploy/production.env \
  -p gfs-production-validation -f deploy/compose.yaml \
  --profile validation run --rm --entrypoint python production-validation \
  gfs.py --base-dir /app studio init \
  --name "Production Validation V1" --mode stable --seed 91000
```

Start the first workload instance. The exact authorization text is deliberately
long and must not be shortened:

```bash
docker compose --env-file deploy/production.env \
  -p gfs-production-validation -f deploy/compose.yaml \
  --profile validation run --rm --name gfs-validation-instance-a \
  production-validation --execute --workspace /app \
  --authorization I_AUTHORIZE_GFS_100_MATCH_PRODUCTION_VALIDATION \
  --deployment-instance-id instance-first
```

The runner atomically writes and then prints a
`production_validation_task_running` event immediately after a task is claimed
and before its match begins. Only after observing that event, capture the
container image ID and terminate that container from a second shell:

```bash
docker inspect --format '{{.Image}}' gfs-validation-instance-a
docker kill --signal KILL gfs-validation-instance-a
```

Start a second instance against the same Compose project and therefore the same
named volume set:

```bash
docker compose --env-file deploy/production.env \
  -p gfs-production-validation -f deploy/compose.yaml \
  --profile validation run --rm --name gfs-validation-instance-b \
  production-validation --execute --workspace /app \
  --authorization I_AUTHORIZE_GFS_100_MATCH_PRODUCTION_VALIDATION \
  --deployment-instance-id instance-second
```

The resumed workload stops after the complete fixed workload and does not write
a decision. An operator must then record the observed deployment facts through
the schema-controlled command below. Replace the digest with the exact
`sha256:...` value captured from the first container. Operator, volume and
instance values are pseudonyms; raw host names, IP addresses and personal names
are forbidden.

```bash
docker compose --env-file deploy/production.env \
  -p gfs-production-validation -f deploy/compose.yaml \
  --profile validation run --rm production-validation \
  --record-attestation --workspace /app \
  --operator-id operator-12345678 \
  --volume-id volume-validation-set \
  --container-image-digest sha256:REPLACE_WITH_64_HEX_DIGEST \
  --attested-deployment-instance-id instance-first \
  --attested-deployment-instance-id instance-second \
  --confirm-forced-termination --confirm-same-volume
```

The attestation is immutable and binds the exact final progress payload by
SHA-256. Any later execution invocation invalidates it. Finalization is a
separate authorized phase: it creates and verifies the backup, restores it to a
fresh directory on the external scratch volume, checks the restored session and
match inventory, and writes one immutable decision.

```bash
docker compose --env-file deploy/production.env \
  -p gfs-production-validation -f deploy/compose.yaml \
  --profile validation run --rm production-validation \
  --finalize --workspace /app --restore-scratch /validation-scratch \
  --authorization I_AUTHORIZE_GFS_100_MATCH_PRODUCTION_VALIDATION
```

The runner and Web service acquire the same workspace Web lease. A production
validation writer therefore cannot run beside Studio or beside another
validator. Progress is persisted before match execution and includes the active
task identity and a privacy-safe SHA-256 of its idempotency key. Finalization
requires the same task ID and digest to complete at exactly the next attempt,
making the forced-termination observation replayable rather than timing-based
guesswork.

The gate passes only when all 100 logical tasks and 100 unique match reports are
complete, every report accepts its integrity checks, provider calls remain zero,
at least one running task was recovered, the deployment attestation matches the
progress journal, and an integrity-checked backup restores an identical session
and match inventory into a clean staging root. Failed and incomplete outcomes
remain valid negative evidence; the runner does not silently retry failed tasks.

This protocol has been registered but not executed. Do not claim the production
operations gate until its generated decision exists and passes the unified
release control plane.
