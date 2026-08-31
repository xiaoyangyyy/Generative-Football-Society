# Evidence-locked paper finalization

The authoritative V2 contract is
`data/evaluation/action_paper_finalization_protocol_v2.json`. The final
manuscript is a separate artifact from the completed-results draft. It cannot
be verified until the existing 60-run action-outcome decision, the current
72-run V2 external-replication decision, and a signed V2 full-study independent
review all exist. Negative, failed, and inconclusive outcomes are valid results
and must remain unchanged. The earlier V1 finalization contract remains
historical and is not authoritative for the repaired action-policy candidate.

Read-only protocol status:

```bash
python scripts/finalize_paper.py
```

After all evidence exists, build the deterministic result ledger explicitly:

```bash
python scripts/finalize_paper.py --build-ledger \
  --authorization I_AUTHORIZE_GFS_ACTION_PAPER_LEDGER_V2
```

The author then writes `docs/PAPER_FINAL.md`, changes all four completion
markers, and embeds the exact canonical ledger block and its SHA-256 marker.
The verifier rejects stale source hashes, altered numbers, branch drift,
missing sections, or residual pre-execution claims:

```bash
python scripts/finalize_paper.py --verify-final \
  --out data/evaluation/action_paper_finalization_v2/verification.json
```

These commands do not train a model, run a simulation, or call a provider.
