# Authenticated GFS Studio deployment

This deployment contract places the Python Web process on an internal Compose
network. Only Caddy publishes host ports. Caddy obtains and renews the public
TLS certificate, forwards the exact public Host and trusted
`X-Forwarded-Proto=https`, and the application adds its own Host allowlist,
product-token login, CSRF, secure session Cookie, and browser security headers.

## Prepare

1. Point the public DNS name to the deployment host and allow inbound TCP 80,
   TCP 443, and optionally UDP 443.
2. Copy `deploy/.env.deploy.example` to an ignored file such as
   `deploy/.env.deploy`.
3. Create the ignored file named by `GFS_WEB_ACCESS_TOKEN_FILE`, containing an
   independent product token with at least 32 random characters. Compose mounts
   it as a read-only secret instead of putting its value in container metadata.
   Do not reuse an LLM/provider API key.
4. Back up the Studio and verify the bundle before replacing a running release.

## Start

```bash
docker compose --env-file deploy/.env.deploy -f deploy/compose.yaml up -d --build
```

Check both services and then open `https://<GFS_WEB_HOST>`:

```bash
docker compose --env-file deploy/.env.deploy -f deploy/compose.yaml ps
docker compose --env-file deploy/.env.deploy -f deploy/compose.yaml logs --no-log-prefix gfs gateway
```

Never print or commit the environment file. Application logs do not need the
access token or provider credentials.

The current deployment intentionally runs one application process. Sessions
and login-rate state are bounded and process-local; do not scale the `gfs`
service horizontally until a shared session/rate store with equivalent expiry,
revocation, privacy, and failure semantics is implemented and tested.

The Web recovery center writes its bounded managed catalog to the `gfs_backups`
volume. Copying bundles off-host, retention beyond 50 entries, encryption, and
timed RPO/RTO drills remain deployment-operator responsibilities.

Before a real deployment exists, the cross-platform local process drill checks
the same application lease and startup reconciliation paths without a match:

```bash
python scripts/verify_process_recovery.py --out data/evaluation/process_recovery_verification_v1.json
```

Its timing is not a deployment-volume RTO and does not validate Docker,
orchestrator restart policy, storage latency, or kill-during-write behavior.

## Stop and recover

```bash
docker compose --env-file deploy/.env.deploy -f deploy/compose.yaml down
```

Named volumes retain reports, session/task state, backups, and Caddy
certificates. Removing volumes is not part of the normal shutdown procedure.
The deployment is not considered production-verified until the image builds,
health checks pass, HTTPS is observed from an external client, backup/restore
is timed on these volumes, and security/accessibility reviews are complete.
