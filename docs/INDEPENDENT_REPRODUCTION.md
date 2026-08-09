# Independent reproduction handoff

The authoritative contract is
`data/evaluation/independent_reproduction_protocol_v1.json`. It is registered
but cannot start until the frozen confirmatory decision and successful
container-runtime evidence both exist. No independent reproduction has been
performed.

The reviewer must be independent of implementation, must not have contributed
to the frozen commit, and must disclose conflicts. A public identity or ORCID,
affiliation, signing time, and exact attestation are required. The reviewer
works from a clean checkout, records the commit and environment, reruns the
exact 30-pair/60-run protocol without training or provider calls, and analyzes
only after completeness checks pass.

Every deviation is classified as `none`, `nonmaterial`, or `material`.
Any material deviation forces an `inconclusive` conclusion. Protocol,
decision, promotion, pair count, bootstrap settings, changed-pair count,
promotion gates, secondary metrics, and execution identity must match exactly.
The five frozen scalar results use an absolute tolerance of `1e-12`.

Read-only protocol audit:

```bash
python scripts/verify_independent_reproduction.py
```

After an external reviewer supplies the signed review:

```bash
python scripts/verify_independent_reproduction.py \
  --review data/evaluation/independent_reproduction_v1/review.json
```

The verifier reads evidence only. It never executes the confirmatory study,
trains a model, starts a match, or calls a provider. A structurally valid
review is not automatically a successful reproduction: all identity,
completeness, comparison, deviation, independence, and artifact gates must
pass.
