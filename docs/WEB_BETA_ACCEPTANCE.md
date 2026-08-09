# GFS Studio Web Beta Acceptance

The Web Beta is a local product adapter over the canonical
`ProductWorkspace` and `ProductControlPlane`. It does not fork simulation,
readiness, evidence, transaction, or reporting rules into a second system.

## Accepted code-level scope

- one responsive page for Studio creation, readiness, match execution, and
  report review;
- versioned JSON routes for Studio status and match execution;
- `/healthz` liveness and `/readyz` evidence-aware readiness;
- real HTTP operation on an ephemeral loopback port;
- loopback Host validation against DNS rebinding, CSRF protection for
  mutations, a 64 KiB request limit, fail-fast concurrent
  mutations, structured errors, path-confined HTML artifacts, CSP, and no CORS;
- keyboard focus styles, semantic labels and landmarks, live status regions,
  responsive layout, and reduced-motion support;
- no external provider call during status, health, readiness, or acceptance.
- persistent idempotent match tasks, refresh-safe polling, bounded terminal
  history, single-server ownership, and visible interrupted state after restart.
- persistent privacy-bounded operational telemetry, aggregate authenticated
  operations metrics, bounded rotation, and readiness-visible telemetry damage.
- one integrated recovery center with a bounded managed catalog, create and
  verify controls, exact-ID destructive confirmation, active-task blocking,
  transactional restore, and task-history reset.

Run the machine-readable verification without a match, training, or network
provider:

```powershell
python scripts\verify_product_web.py --out data\evaluation\web_beta_verification_v1.json
```

The authoritative result is
`data/evaluation/web_beta_verification_v1.json`. A passing local result proves
the HTTP adapter and its safety boundaries; it does not prove production
deployment, match soak reliability, external WCAG conformance, or user value.

## Authenticated boundary

The default Beta is loopback-only. Explicit remote mode additionally requires
an exact Host allowlist, a separate high-entropy product token, an HTTPS proxy
marker, and a secure session Cookie. A real socket verifier covers login and
authorization. Sessions are independent, bounded, absolutely and idly expiring,
and individually revocable. Client/global login limits return 429 with
`Retry-After`. The Caddy/Compose topology is only statically verified.

## Deliberately unclaimed

Match requests execute in a persistent
single-worker queue while the threaded server keeps health endpoints
responsive. Backup download/export, container deployment, telemetry export to
an external monitoring backend, the deterministic 100-match soak,
external accessibility review, and target-user validation remain Stage 1/3
work. The local `/api/v1/operations` endpoint exposes aggregates only, never raw
events.
