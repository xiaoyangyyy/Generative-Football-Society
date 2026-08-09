# Result-contingent mechanism and replication study

This prospective Stage 4 protocol starts only after the frozen confirmatory
decision exists. The decision selects exactly one branch; replication outcomes
cannot select or change it:

- promotion candidate: replicate improvement and isolate a positive planning
  contribution;
- no meaningful difference: replicate equivalence and test whether enabling
  planning remains practically indistinguishable from prediction-only M1;
- inconclusive: run the same fixed design and report only a bounded variance
  diagnosis.

All branches execute the same 72 full-length simulations. Six fixtures use
twelve teams absent from the confirmatory fixtures, four independent samples
are used per fixture, and three matched arms are evaluated in a frozen order:
M0, M1 with the same checkpoint but planning disabled, and full M1. This makes
the planning mechanism identifiable without introducing a new trained model or
an external provider.

External scale is evaluated separately against the seven-match licensed IDSSE
manifest. Source match totals are divided by two teams. Full M1 must place both
mean passes per team match in the observed 364–492 range and mean shots per team
match in the observed 10.5–13.5 range. These are deliberately literal observed
ranges rather than tuned tolerances. A failure is retained and blocks a positive
external-validity claim.

The default command is status-only and runs no simulation:

```bash
python scripts/academic_replication_study.py
```

Execution requires the completed, identity-valid confirmatory decision and the
exact authorization token frozen in the protocol. It never authorizes training
or provider calls. Progress is atomic and resumable only when the protocol,
confirmatory decision, checkpoint, and code hashes remain identical. Analysis
is only available after all 72 fixed runs are complete.

This protocol is currently waiting for the confirmatory decision. Its presence
does not satisfy the mechanism/replication or final release gate.
