# V9.4 Official Metrica Game 3 Acceptance

V9.4 imports the third match published in the official Metrica Sports sample-data repository. Game 3 uses FIFA EPTS tracking, XML metadata, and JSON events. PASS events contain explicit start/end frame and time, passer, and receiver identifiers.

The adapter adds 1,118 pass receptions and 1,101 observed next actions. Metrica now contributes 3 matches, 2,879 events, and 2,808 observed times. Its combined median delay is 1.08 seconds and 90th percentile is 3.00 seconds, consistent with SkillCorner and StatsBomb 360.

Strict three-way LOPO passes all point and coverage gates. Held-Metrica MAE improves from 0.9096 seconds in v9.2 to 0.8796 seconds. Held-Metrica 90% coverage is 92.17% and integrated Brier is 0.1406.

All official Game 3 raw files are pinned by SHA-256. V9.4 is evaluated only; active production remains v7 with v6 rollback.