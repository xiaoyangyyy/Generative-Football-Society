# Formal Research Validation — 2026-08

This report closes the frozen evidence-first plan while keeping deployment at
v7.0.0. Completion means every planned experiment and decision was executed;
it does not mean that rejected research layers were promoted.

## Formal baseline

M0 was evaluated on six World Cup fixtures, three matched seeds per fixture,
and full 5,400-second matches. All 18 samples passed the frozen hard and soft
external-observable contract. The calibration planner therefore selected
`no_op_freeze_parameters`: changing passing, shooting, discipline, or spatial
parameters after a passing formal baseline would have been unsupported.

## Full matched-seed ablation

Seven variants used the same 18-match protocol and a paired match bootstrap:

| Variant | Observed loss delta | Gate result | Decision |
|---|---:|---|---|
| no affective dynamics | +25.9540 | failed | retain core |
| no spatial intelligence | +28.6695 | failed | retain core |
| no pass-logit blend | -0.1345 | passed | optional, default off |
| no discipline tick | +82.4630 | failed | retain core |
| no tactical bias | +26.3289 | failed | retain core |
| M1 world model | approximately 0 | passed | research-only, default off |
| C1 cognition | -0.8656 | passed | research-only, default off |

M1 and C1 did not have paired 95% intervals wholly below zero, so neither has
a statistically resolved external-loss improvement.

## Retrained world-model candidate

A new group-safe dataset contains 72 simulated matches: 48 regular and 24
shot-rich. The candidate used a three-member GRU ensemble, 50 epochs, a
two-step curriculum, and learned semantic event/path heads.

The v9 candidate was rejected by strict validation. Transition quality was
0.795 and the semantic heads were useful for three events at one and two
steps, but two-step state prediction was 10.1% worse than persistence and the
shot planner quality was only 0.0054. The stable checkpoint was not replaced.

## Mechanisms and LLM boundary

The mechanism-chain, stress-test, active-probe, sequential-policy, fusion, and
online-calibration contract suite passed 79 tests. No real cognitive or fusion
logs were present, so this proves implementation correctness only. It does not
support a live-provider LLM benefit claim.

## Final verification

- Full suite: 539 passed, 2 skipped.
- Frozen v7: 16 artifacts verified; reversible Git LF/CRLF conversion accepted,
  any other byte change rejected.
- Release boundary: no private raw data, checkpoint residue, or oversized file.
- Stable deployment remains v7.0.0 with v6.0.0 rollback.

Reproduce the final audit with:

```bash
python scripts/audit_staged_completion.py
```

Machine-readable evidence is in `data/evaluation/staged_completion_v1.json`.
