# All known exposed credential closure

Two incidents are currently registered: the exposed DeepSeek API credential
and the exposed GitHub Classic personal access token. Both must be revoked in
their respective provider consoles. Closing only one incident cannot pass the
gate. Never copy an old or replacement credential into the repository, an
attestation, logs, screenshots, issue text, command arguments, or a remote URL.

After both revocations, place one distinct redacted provider receipt per
incident under `data/evaluation/security_closure_v2/`. Each receipt may be
JSON, PDF, or PNG, must be non-empty and no larger than 10 MiB, must not show
any credential value, and is addressed only by SHA-256. Different filenames
with identical bytes do not count as distinct receipts. Create the single
multi-incident attestation described by
`data/evaluation/security_closure_protocol_v2.json`, tied to the current Git
commit, then verify it:

```bash
python scripts/verify_security_closure.py \
  --attestation data/evaluation/security_closure_v2/attestation.json \
  --out data/evaluation/security_closure_verification_v2.json
```

The verifier scans repository files for OpenAI-compatible keys, GitHub Classic
PATs and GitHub fine-grained PATs. It reports only counts by credential family,
never values. It also requires the exact registered incident set, distinct
content-addressed receipts, current-commit scan identity and the fixed signed
attestation. The runtime check also rejects credentials embedded in the
HTTPS origin URL without printing that URL. A replacement may be marked
`not_generated`; if generated, its
only permitted recorded location is `local_environment_only`.
Every revocation timestamp must be timezone-aware, no later than the signature,
and neither the revocation nor signature may be in the future.

The repository scan cannot prove provider-side revocation. The credential
owner must perform revocation in the official consoles and supply genuinely
redacted receipts. Templates are available through the evidence kit and are
not evidence until completed and accepted by the verifier.
