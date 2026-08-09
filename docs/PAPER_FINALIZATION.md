# Evidence-locked paper finalization

The final manuscript is a separate artifact from the registered-report draft.
It cannot be verified until the 60-run confirmatory decision, the selected
72-run mechanism-replication decision, and a signed full-study independent
review all exist. Negative, failed, and inconclusive outcomes are valid
results and must remain unchanged.

Read-only protocol status:

```bash
python scripts/finalize_paper.py
```

After all evidence exists, build the deterministic result ledger explicitly:

```bash
python scripts/finalize_paper.py --build-ledger \
  --authorization I_AUTHORIZE_GFS_PAPER_RESULT_LEDGER_V1
```

The author then writes `docs/PAPER_FINAL.md`, changes all four completion
markers, and embeds the exact canonical ledger block and its SHA-256 marker.
The verifier rejects stale source hashes, altered numbers, branch drift,
missing sections, or residual pre-execution claims:

```bash
python scripts/finalize_paper.py --verify-final \
  --out data/evaluation/paper_finalization_v1/verification.json
```

These commands do not train a model, run a simulation, or call a provider.
