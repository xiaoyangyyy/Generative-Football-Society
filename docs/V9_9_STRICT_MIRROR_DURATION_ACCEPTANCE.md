# V9.9 Strict Mirror and Duration Acceptance

## Decision

V9.9 validates the reception shadow under genuinely matched home-away seeds and rejects an unnecessary model correction. Deployment remains unchanged.

V9.8 used mirrored fixtures but derived different random streams for each orientation. Its broad stability and performance evidence remains useful, while its mirror-bias interpretation is superseded by V9.9.

## Strict Mirror Protocol

The seed is derived from the sorted unordered fixture and seed index. A-B and B-A therefore share the same random stream identity.

The 120-second run covers 14 teams, 42 oriented fixtures, 8 seeds, and 336 matched pairs. All eleven gates pass.

- Home-possession delta: -0.0068, 95% cluster CI [-0.0129, -0.0007].
- Completion delta: 0.0049, 95% cluster CI [-0.0012, 0.0109].
- xG delta: 0.0023, 95% cluster CI [-0.0032, 0.0083].

Strict seeds halve the earlier possession estimate but leave a small short-window signal.

## Duration Diagnostic

The 600-second diagnostic covers the same 14 teams and 42 fixtures with 4 seeds, for 168 matched pairs.

- Home-possession delta: 0.00005, 95% cluster CI [-0.0128, 0.0123].
- Completion delta: -0.0029, 95% cluster CI [-0.0086, 0.0028].
- xG delta: -0.0121, 95% cluster CI [-0.0373, 0.0170].
- 600-second throughput: 746.5 matched pairs/hour with four workers.

The only false gate is the intentionally unmet eight-seed acceptance gate. All behavioral, reconciliation, performance, and stability gates pass.

## Interpretation

The short-window possession signal does not accumulate. It converges to zero as the initial possession and finite-tick transient lose weight. Reception residual displacement therefore should not receive a compensating home-away correction.

The next justified step is a limited full-match strict-mirror audit, not another architecture change.

## Verification

    python scripts/verify_strict_mirror_v99.py
    python -m pytest -q
