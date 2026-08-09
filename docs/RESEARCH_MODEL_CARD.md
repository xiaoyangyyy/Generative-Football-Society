# GFS Research Model Card

## Model family and authority

GFS contains a stable simulator pipeline (M0) and a sealed calibrated
world-model planner candidate (M1). M0 is the active research baseline. M1 is
default-off, research-only, and cannot change the stable release pointer.
Language-model cognition is a separate optional layer and has no demonstrated
prospective benefit.

## Intended use

- deterministic football-simulation research;
- evaluation of simulator-internal calibration and behavior;
- mechanism studies under frozen fixtures and seeds; and
- software/research workflow demonstrations.

## Out-of-scope use

- real player, team, coaching, scouting, betting, medical, employment, or
  disciplinary decisions;
- causal claims about real football or society;
- autonomous production promotion; and
- claims that simulated agent dialogue represents human cognition.

## Inputs and outputs

M0 consumes fixture, seed, team state, tactics, and simulator state. M1 also
consumes the frozen world-model state representation and checkpoint. Outputs
are simulator actions, event traces, aggregate match observables, integrity
records, and calibration reports.

## Evaluation status

The frozen M0 evaluation contains 18 full-length simulations and passes the
repository external observable contract. A prior sealed 18-match M1 review
found no resolved loss improvement and no behavior change above tolerance;
M1 remains `research_only_default_off`. The new 30-pair confirmatory
comparison is preregistered and not executed.

## Risks and limitations

External observations are incomplete proxies with provider-specific clocks,
labels, sampling, and selection. Calibration against aggregate football
metrics does not establish realistic trajectories, decisions, psychology, or
causality. M1 has material compute cost, and previous evidence did not show
action-path lift. LLM outputs can be invalid or nondeterministic and are
excluded from stable authority.

## Governance

Promotion requires frozen identity, complete paired evidence, external
validity gates, behavior change, full regression, and separate release review.
Checkpoint existence, code execution, or a favorable secondary metric cannot
authorize promotion.
