# GFS Studio Backup and Recovery

Studio recovery protects the mutable user workspace without changing the
frozen simulator release, research evidence, checkpoints, training runs, or
provider credentials. A bundle contains only the persisted Studio session and
the report, dashboard, cognitive log, and ball log files referenced by its
completed match journal.

## Operator workflow

Create a new archive while no match transaction is active:

```powershell
python gfs.py studio backup --out backups\studio-2026-08-09.zip
```

Verify it without writing to the workspace:

```powershell
python gfs.py studio verify-backup backups\studio-2026-08-09.zip
```

Restore into an empty workspace. An existing Studio is never replaced
implicitly:

```powershell
python gfs.py studio restore backups\studio-2026-08-09.zip
python gfs.py studio restore backups\studio-2026-08-09.zip --replace
```

`--replace` is an explicit destructive boundary. Every archive member is
staged and fsynced first; referenced artifacts switch before the session file.
If any switch fails, already changed files are rolled back to their exact prior
bytes.

## Integrity and trust boundary

The manifest records each relative path, byte size, and SHA-256. Verification
rejects duplicate or untracked members, path traversal, symbolic links,
oversized expansion, missing match artifacts, invalid Studio configuration,
and hash/size mismatch. Restore never calls `ZipFile.extract` and never writes
outside the workspace whitelist.

SHA-256 makes corruption and accidental modification observable; it does not
authenticate a bundle supplied by an attacker. Store backups in access-
controlled storage and add a signed/encrypted storage layer before production.
API keys and environment configuration are deliberately excluded.
The operational Web task queue is also excluded from the portable bundle.
Operational telemetry is likewise excluded: it remains append-only deployment
evidence on the persistence volume, and records only aggregate-safe backup and
successful restore events without bundle paths or Studio names.
Restore refuses to proceed while any task is queued or running, then resets
terminal task history inside the same rollback transaction so tasks from one
Studio can never appear in another restored Studio.

Run the isolated recovery acceptance without a match, training, or external
call:

```powershell
python scripts\verify_product_recovery.py --out data\evaluation\product_recovery_verification_v1.json
```

Take a backup before each product study or release-candidate change and verify
it immediately. The following automated drill force-kills an owned local Web
child, starts a fresh process, and proves OS-lease release, orphaned-task
reconciliation, committed Studio/backup preservation, and timed health
recovery without running a match:

```powershell
python scripts\verify_process_recovery.py --out data\evaluation\process_recovery_verification_v1.json
```

This is a temporary-local-filesystem process test. Production RPO/RTO remain
unclaimed until timed failure and restore drills run against the actual
deployment volumes, including faults during filesystem replacement.

The Studio Web recovery center manages `backups/studio/` with generated backup
IDs and a 50-entry capacity. Its API never accepts or returns an absolute path.
Restore requires CSRF, `replace=true`, and an exact typed backup ID, and remains
blocked while match tasks are queued or running. Run its local socket verifier:

```powershell
python scripts\verify_web_recovery.py --out data\evaluation\web_recovery_verification_v1.json
```
