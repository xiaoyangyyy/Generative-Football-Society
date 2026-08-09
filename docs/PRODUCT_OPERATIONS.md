# Product operations and telemetry

GFS Studio uses one operational evidence stream at
`data/persistence/product_telemetry.jsonl`. Web requests, authentication
outcomes, persistent task transitions, worker lifecycle, and successful
backup/restore operations all flow through this component; it is not a
detached analytics subsystem.

## Privacy and security contract

The schema has a closed event and field allowlist. It records normalized route
templates, status classes, bounded request/task IDs, durations, transition
counts, and exception class names. It never accepts request bodies, headers,
Cookies, tokens, IP addresses, Host values, query strings, fixture/team names,
idempotency keys, artifact paths, backup paths, or exception messages.

The current log is limited to 2 MiB with one rotated segment. Writes are
process-safe, flushed and fsynced. A telemetry I/O failure increments an
in-memory degradation counter and never rolls back a successful product
transaction. Corrupt records and write degradation appear in readiness.

Authenticated clients receive aggregate metrics through
`GET /api/v1/operations`; the Studio status envelope includes the same
snapshot. Raw events are never returned by the Web API.

## Code-only operations verification

Run the control-plane soak without a match, training, formal experiment, or
external call:

```powershell
python scripts\verify_product_operations.py --tasks 100 --out data\evaluation\product_operations_verification_v1.json
```

The verifier makes 200 concurrent submission attempts for 100 logical tasks,
requires exactly-once idempotent persistence, completes 95 tasks, records three
controlled failures, leaves two running across a simulated restart, and
requires exact recovery and telemetry counts.

This is not the Stage 1 100-match soak. It does not execute the match engine,
measure long-duration memory/resource behavior, exercise containers or TLS, or
support any scientific model-quality claim. Those gates remain open until
separately executed and evidenced.
