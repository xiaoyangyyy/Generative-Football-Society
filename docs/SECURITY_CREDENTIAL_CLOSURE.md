# Exposed provider credential closure

The exposed DeepSeek credential must be revoked in the provider console. Never
copy the old or replacement credential into the repository, an attestation,
logs, screenshots, issue text, or command arguments.

After revocation, place a redacted provider receipt under
`data/evaluation/security_closure_v1/`. It may be JSON, PDF, or PNG, must not
show any credential value, and is addressed only by SHA-256. Create the
attestation described by
`data/evaluation/security_closure_protocol_v1.json`, tied to the current Git
commit, then verify it:

```bash
python scripts/verify_security_closure.py \
  --attestation data/evaluation/security_closure_v1/attestation.json \
  --out data/evaluation/security_closure_verification_v1.json
```

The verifier scans repository files using a boundary-aware provider-key
pattern but reports only the count. It never prints, stores, or calls a
credential. A replacement may be marked `not_generated`; if generated, its
only permitted recorded location is `local_environment_only`.
