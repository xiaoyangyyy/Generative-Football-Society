# GFS Research Data Card

## Scope

This card describes the evidence represented in the repository. It does not
grant access or redistribution rights to any upstream dataset.

| Source | Repository coverage | Primary use | Known limitation |
|---|---:|---|---|
| StatsBomb World Cup 2022 benchmark | 128 team-matches | 15 frozen aggregate observables | event-data selection and competition scope |
| StatsBomb 360 | 425 matches; 373,070 passes; 10,483 shots | freeze-frame pass/shot supervision | not continuous tracking |
| SkillCorner Open Data | 10 matches; 8,585 passes; 3,314 shots | tracking-derived pass and state evidence | 1 Hz sampled tracking in derived manifest |
| Sportec IDSSE | 7 matches; 6,062 passes; 171 shots; 1,002,644 frames | independent tracking/event provider | completion labels have four negatives and are insufficient |
| Metrica Sports sample data | 3 temporal-provider matches; two receiver-choice games | receiver and temporal semantics | small sample-game collection |

## Provenance and integrity

Derived manifests record provider names, match identifiers, counts, and many
source/derived SHA-256 values. Metrica receiver manifests include the canonical
sample-data repository URL. Sportec records declare CC BY 4.0 locally.
SkillCorner and StatsBomb derived manifests name their sources but do not
contain a complete license decision in the same artifact.

Hashes detect identity drift; they are not licenses and do not prove that a
dataset is representative, unbiased, or suitable for a downstream decision.

## Processing and separation

Provider-specific loaders produce event, tracking, pass, shot, and temporal
artifacts. Holdout and leave-one-provider-out evaluations are used where
recorded. StatsBomb 360 freeze frames are never described as continuous
tracking. Metrica receiver-choice data is not used as completion labels.

The confirmatory M0/M1 experiment runs simulation against a frozen external
observable contract. It does not retrain on the confirmatory outputs.

## Licensing and availability

`data/evaluation/data_license_registry_v1.json` covers the five known external
source families with source evidence, verification status, and an archive
decision. Its policy is default deny. StatsBomb has custom source terms,
Metrica has acknowledgement guidance without a detected license, SkillCorner
exposes MIT repository metadata but still needs a data-scope review, Sportec
has only a local CC BY 4.0 assertion, and the Transfermarkt mirror has no
recorded redistribution terms. Consequently every external source remains
excluded from a release archive. This is a complete decision registry for
known repository sources, not a completed legal review or a blanket
archival-data claim.

## Bias and validity limits

Competition, geography, team quality, tracking technology, annotation
practice, class balance, and missingness differ across providers. Aggregate
calibration can hide match- or provider-level failures. Sportec pass
completion is explicitly insufficient because its observed negative class is
too small. No dataset in this card supports causal claims about real people or
organizations.
