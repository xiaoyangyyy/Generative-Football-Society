# V7 Candidate Acceptance

Status: **evidence-ready candidate**
Frozen parent: `6.0.0`

V7 is layered on the frozen, hash-verified v6 release. It does not overwrite a v6 checkpoint, sealed split, metric, or release manifest.

## Imported evidence

- Metrica: 2 continuous-tracking matches and 19,393 pass candidates.
- Sportec IDSSE: 7 continuous-tracking matches, 1,002,644 frames, 6,062 passes, and 171 shots.
- StatsBomb 360: 425 valid event-context matches, 373,070 passes, and 10,483 shots. Match `3845506` is excluded because the official upstream JSON is malformed.
- SkillCorner: 10 complete tracking matches with 642,687 frames, plus metadata, phases, dynamic events, 8,585 passes, and 24,374 receiver candidates.

The reproducible inventory and derived-file hashes are in `data/releases/v7-data-candidate.json`.

## Held-out results

| Candidate | Independent evidence | Result | Gate |
|---|---:|---:|---:|
| Joint pass completion | 55,404 samples, 66 matches | BA 0.7362, AUC 0.7891, Brier 0.1260, ECE 0.0211 | pass |
| Receiver ranking | 1,950 events | Top-1 0.3810, Top-3 0.8400, MRR 0.6193 | pass |
| Shot probability | 1,590 shots, 194 goals, 64 matches | AUC 0.7484, Brier 0.0922 vs 0.1072 baseline | pass |
| Probabilistic transitions | 78,856 events, 85 matches | log loss 0.4381 vs 0.5710; pass-only 0.4501 vs 0.4878 | pass |
| SkillCorner tracking | 10 matches, 642,687 frames | 75.2% adapter-complete sampled frames; 68.0% ball availability | pass |
| SkillCorner strict LODO | 8,585 passes; 6,828 receiver events | completion AUC 0.881, ECE 0.019; receiver Top-3 0.960, MRR 0.687 | pass |
| Metrica strict receiver LODO | 1,763 receiver events | Top-1 0.314, Top-3 0.682, MRR 0.532 | pass |
| Sportec strict receiver LODO | 4,751 receiver events | Top-1 0.290, Top-3 0.646, MRR 0.508 | pass |

Splits are provider-stratified match holdouts. Calibration is fitted only on dev matches. `reports/acceptance/v7_readiness.json` is the machine-readable aggregate decision.

## Scope boundary

The probabilistic transition checkpoint is an event-level calibrated model, not a claim of learned frame-level football physics. StatsBomb 360 is event freeze-frame context, not continuous tracking. Broadcast tracking has real occlusion: SkillCorner averages 14.95 represented players and 8.57 directly detected players per sampled frame. Full-provider receiver LODO is accepted using event-internal percentile features; Sportec completion LODO remains evidence-insufficient because only four failed passes carry intended-receiver labels.

## Reproduce

```bash
python scripts/build_v7_data_manifest.py
python scripts/train_joint_pass_v7.py
python scripts/train_shot_v7.py
python scripts/train_probabilistic_world_v7.py
python scripts/assess_skillcorner_tracking_v7.py
python scripts/evaluate_joint_pass_lodo_v7.py
python scripts/assess_v7_readiness.py
python scripts/verify_release_manifest.py data/releases/v6.0.0.json
python -m pytest -q
```
