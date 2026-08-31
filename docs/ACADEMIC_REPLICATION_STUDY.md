# Action-policy external replication study

The active V2 protocol is
`data/evaluation/academic_action_replication_protocol_v2.json`. It was
registered after the completed action-policy primary result and before any V2
replication run. It therefore does not pretend to be a pre-primary
preregistration. The already observed
`inconclusive_keep_research_only` decision fixes the variance-diagnosis
branch; replication outcomes cannot select or change that branch.

The repository retains the earlier V1 protocol and its 72-run failed result as
historical negative evidence. V1 produced zero behavior-changing pairs and
failed external shot-scale validity, but it predates the action-adoption code
path and its execution identity is now stale. It cannot be used as replication
evidence for the current candidate and is not overwritten by V2.

The branch vocabulary remains frozen:

- promotion candidate: replicate improvement and isolate a positive planning
  contribution;
- no meaningful difference: replicate equivalence and test whether enabling
  planning remains practically indistinguishable from prediction-only M1;
- inconclusive: run the same fixed design and report only a bounded variance
  diagnosis.

All branches execute the same 72 full-length simulations. Six fixtures use
twelve teams absent from the confirmatory fixtures, four independent samples
are used per fixture, and three matched arms are evaluated in a frozen order:
M0, M1 with the same checkpoint but action planning disabled, and the
mechanism-confirmed M1 action policy. This makes the action-policy contribution
identifiable without introducing a new trained model or an external provider.

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

Execution requires the completed, identity-valid action-outcome decision and
the exact V2 authorization token frozen in the protocol. It never authorizes
training or provider calls. Progress is written to
`academic_action_replication_v2` and cannot collide with V1 evidence. Resume
is allowed only when the protocol, primary decision, checkpoint, action-policy
code and data hashes remain identical. Analysis is available only after all 72
fixed runs are complete.

V2 is currently `ready_not_started`: 0/72 runs, no training, and no provider
calls. Registration alone does not satisfy the external-replication or final
release gate.
