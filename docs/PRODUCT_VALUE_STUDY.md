# Comparative product-value study

This preregistered study asks a narrower and more useful question than a generic
satisfaction survey: does GFS Studio help target users reach evidence-correct
football-analysis decisions faster than their current manual workflow without
reducing correctness or creating critical errors?

The design is a randomized, within-participant crossover with two matched case
packs. Sequence `AB` uses GFS Studio on alpha and the manual baseline on beta;
sequence `BA` reverses the condition order and case assignment. Each participant
receives a non-measured training task first. The frozen minimum is 24 valid
participants: at least eight in each target role and at least twelve in each
sequence. Optional stopping and performance-based exclusion are forbidden.

Incorrect or incomplete work receives the full 900-second penalty. The primary
endpoint is the geometric mean of each participant's GFS-to-baseline penalized
time ratio. Its deterministic participant-bootstrap 95% upper bound must be at
most 0.80. In addition, the lower bootstrap bound for the paired accuracy
difference must be at least -0.05 and GFS must produce zero critical errors.
These constraints prevent a superficially faster but less trustworthy workflow
from passing.

Analysis records are JSON Lines with an exact allowlist. Participant and
moderator IDs are pseudonymous; raw text and direct identifiers are forbidden.
Each completed condition requires an evidence SHA-256. Exclusions are limited to
withdrawn consent or infrastructure failure before the first measured condition,
and excluded records contain no behavioral measurements.

Audit the unexecuted protocol with:

```bash
python scripts/product_value_study.py
```

After independently collecting the frozen records, analyze them with:

```bash
python scripts/product_value_study.py --analyze \
  --records data/evaluation/product_value_validation_v1/participant_records.jsonl \
  --out data/evaluation/product_value_validation_v1/decision.json
```

The analyzer makes no provider call, runs no match, observes no participant, and
does not overwrite an existing decision unless `--overwrite` is explicit. A
failed study is a valid result and does not satisfy the unified release gate.
