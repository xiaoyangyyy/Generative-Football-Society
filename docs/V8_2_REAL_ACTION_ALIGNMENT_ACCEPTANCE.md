# V8.2 Real Action Alignment Acceptance

Date: 2026-07-25

## Decision

V8.2 passes the research gates for real event-to-frame alignment and direction-conditioned pass/shot transitions. Pressure passes a separate SkillCorner match-held-out recognition gate. Artifacts remain evaluated candidates; V7 remains deployed.

## Dataset

- 16,170 strong pass events.
- 445 strong shot events.
- 7,063 strong pressure events from SkillCorner.
- 19 matches and three tracking providers on the unified 10 Hz clock.
- Alignment P95: 0.39 ms for SkillCorner, about 40 ms for Metrica, and 53-55 ms for Sportec, within the 61 ms tolerance.

Labels are multi-hot because pressure and an on-ball action may overlap. Every aligned event retains source time, frame index, alignment error, actor/team identifiers, confidence, and supervision strength.

Sportec TacklingGame events are retained as weak pressure proxies but excluded from strong pressure targets. Metrica pressure is unavailable, not treated as a negative label.

## Direction Validation

Metrica endpoint directions have mean cosine agreement 0.984 with observed 0.5-second ball displacement. SkillCorner requires a 180-degree conversion for right-to-left event coordinates; after correction, mean cosine agreement is 0.871 and 96.5% of labels have positive agreement.

Sportec PlayAngle is not validated in the tracking global coordinate system. Its direction is therefore marked unknown and inference falls back exactly to the kinematic baseline.

## Strict Provider LODO

At the 0.5-second event horizon:

| Held provider | Direction evidence | Event-ball conditioned | Direction removed | Result |
| --- | --- | ---: | ---: | --- |
| Metrica | yes | 4.304 m | 6.115 m | pass |
| SkillCorner | yes | 3.007 m | 3.029 m | pass |
| Sportec | no | baseline fallback | baseline fallback | pass |

The held provider is excluded from training and development selection. Correct action direction improves held-out event-ball prediction by 29.6% on Metrica and 0.7% on SkillCorner. The small SkillCorner LODO gain is retained as a boundary, not overstated.

## Pressure

Strong pressure supervision is evaluated using eight SkillCorner training matches, one development match, and one held-out test match. Test pressure F1 is 0.659. Pressure is currently an auxiliary recognition target only: it does not drive entity transitions until actor-to-slot mapping is independently validated.

## Boundaries

- Pass/shot type without a validated direction cannot determine a ball trajectory and is not allowed to move the ball.
- Sportec pressure remains weak evidence.
- Shot recognition remains data-limited at 445 events and is reported, not promoted as a standalone head.
- Local defensive response to pressure still requires actor-slot and target-player alignment.

Evidence: `data/frame_world/v82_actions/manifest.json`, `reports/acceptance/frame_world_v82.json`, and `reports/acceptance/frame_pressure_v82.json`.
