# V8.5 Pass Triplet Acceptance

Date: 2026-07-25

## Decision

V8.5 passes joint ball, receiver, and interceptor transition gates for Metrica and SkillCorner under strict bidirectional provider leave-one-out. Sportec uses exact fallback because its event direction is not validated in global tracking coordinates. The artifacts are evaluated research candidates and are not deployed.

## Data Correction

The audit found that Metrica's `Ball` CSV header had been assigned both to a player slot and the canonical ball slot. This produced a false opponent located exactly on the ball. The v8.5 corrected Metrica frames exclude `Ball` from player identity mapping and are stored separately, preserving registered v8 evidence. Median nearest-interceptor lane distance changed from about 0.01 m to 2.61-2.71 m.

The event dataset contains 16,170 aligned passes and 14,088 valid ball/receiver/defender triplets. The defender is the visible opponent nearest to the ball-receiver segment at the pass frame.

## Joint Model

The transition head jointly predicts bounded residuals for:

- the ball;
- the identified receiver;
- the nearest interception candidate.

Ball and player residuals have separate physical caps. Inputs include pairwise metric geometry, lane distance, velocity, and one-frame velocity change. Unsupported inference is exactly the constant-velocity fallback.

## Strict Provider LODO

| Held provider | Conditioned joint | Zero intervention | Result |
| --- | ---: | ---: | --- |
| Metrica | 1.770 m | 2.394 m | pass |
| SkillCorner | 1.005 m | 1.177 m | pass |
| Sportec | 0.798 m | 0.798 m | exact fallback |

Metrica is trained only from SkillCorner and SkillCorner only from Metrica. The held provider is absent from training and development selection. Sportec is excluded from strong intervention training because its direction contract remains unavailable.

## Identity Falsification

The identity tests preserve the original players, absolute coordinates, kinematic baseline, and targets. Only receiver-conditioned or defender-conditioned feature columns are reassigned within the batch.

| Held provider | Correct | Receiver shuffled | Defender shuffled |
| --- | ---: | ---: | ---: |
| Metrica | 1.770 m | 2.405 m | 2.164 m |
| SkillCorner | 1.005 m | 2.149 m | 1.011 m |

Both identity gates pass. The smaller SkillCorner defender margin is retained as a boundary rather than overstated.

Canonical evidence:

- `data/frame_world/v85_tracking/manifest.json`
- `data/frame_world/v85_pass_triplets/manifest.json`
- `reports/acceptance/frame_pass_triplet_v85.json`
