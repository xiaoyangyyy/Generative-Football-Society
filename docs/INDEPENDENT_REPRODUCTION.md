# Independent reproduction handoff

The authoritative contract is
`data/evaluation/independent_reproduction_protocol_v1.json`. It is registered
but cannot start until the frozen confirmatory decision, its complete
result-contingent mechanism-replication decision, and successful
container-runtime evidence all exist. No independent reproduction has been
performed.

The reviewer must be independent of implementation, must not have contributed
to the frozen commit, and must disclose conflicts. A public identity or ORCID,
affiliation, signing time, and exact attestation are required. The reviewer
works from a clean checkout, records the commit and environment, reruns the
exact 30-pair/60-run confirmatory protocol and then the selected 72-run,
three-arm mechanism-replication protocol without training or provider calls.
The full independent budget is therefore 132 simulations. Each stage is
analyzed only after its own completeness checks pass.

Every deviation is classified as `none`, `nonmaterial`, or `material`.
Any material deviation forces an `inconclusive` conclusion. Protocol,
Both decisions, result-selected branch, promotion or nonpromotion outcome,
pair counts, bootstrap settings, changed-pair counts, validity gates,
secondary metrics, and execution identities must match exactly. The frozen
scalar results in both stages use an absolute tolerance of `1e-12`. A failed
or negative mechanism result must reproduce as that same result; it cannot be
relabelled as success.

Read-only protocol audit:

```bash
python scripts/verify_independent_reproduction.py
```

After an external reviewer supplies the signed review:

```bash
python scripts/verify_independent_reproduction.py \
  --review data/evaluation/independent_reproduction_v1/review.json
```

The verifier reads evidence only. It never executes either study, trains a
model, starts a match, or calls a provider. A structurally valid review is not
automatically a successful reproduction: both stages' identity, completeness,
comparison, deviation, independence, and artifact gates must pass.
