# V9.8 Large-Sample Shadow Acceptance

## Decision

V9.8 passes performance convergence and large-sample shadow. It remains a research shadow and does not change deployment. V9.7 retains the full 90-minute production-cadence evidence.

## Protocol

- 14 canonical national teams spanning multiple confederations and strength bands.
- 21 unordered fixtures, each mirrored home and away.
- 8 deterministic seeds per orientation.
- 336 matched baseline/treatment pairs at 120 seconds and the production 5.5-second macro cadence.
- Cluster bootstrap by unordered fixture preserves dependence across mirrors and seeds.

The 120-second window is used for high-powered regression and performance estimation. It is not presented as a replacement for the v9.7 full-match shadow.

## Results

- 5,568 reception events scheduled and applied; zero cancelled.
- Treatment overhead ratio: 1.0349.
- Four-worker observed throughput: 4,238 matched pairs/hour.
- Pass delta: 0.0089, 95% cluster CI [-0.0298, 0.0476].
- Completion delta: 0.0048, 95% cluster CI [0.0007, 0.0091].
- xG delta: -0.0020, 95% cluster CI [-0.0075, 0.0034].
- Home-possession delta: -0.0140, 95% cluster CI [-0.0211, -0.0067].

All eleven predeclared gates pass. The small signed possession shift is statistically visible but remains inside the 0.05 stability boundary and should remain a monitored metric in any full-match expansion.

## Performance Changes

Temporal calibration assets and pass-triplet weights are now integrity-checked and loaded once per worker. The shadow runner builds one world per worker, keeps match writeback disabled, writes atomic checkpoints, validates resume protocol compatibility, and supports a serial fallback.

## Verification

    python scripts/verify_large_shadow_v98.py
    python -m pytest -q
