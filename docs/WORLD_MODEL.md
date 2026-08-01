# World Model v7

The world model learns action-conditioned match transitions for counterfactual
pass and shot planning. It is an optional advisory layer: the deterministic
match engine remains authoritative.

## Core design

```text
full observation + candidate action
              |
 CNN grid + player attention + context encoder
              |
 bootstrap GRU/Transformer ensemble
              |
        residual state delta
              |
 mean next observation + trajectory disagreement + outcome ensemble
              |
 quality/OOD gate -> bounded planner advantage
```

The 307-dimensional observation contains spatial fields, ball state, score and
clock, all 22 player positions and velocities, tactics, and attack direction.
Actions contain type, target, pass/shot kind, receiver slot, pre-action
estimates, and a normalized transition horizon.

Training-only outcome slots contain the actual pass result, goal result, and
on-target result. These labels are extracted and zeroed before the action enters
the model, preventing outcome leakage.

## Important design properties

- Predicts a bounded residual over the current observation instead of rebuilding
  the whole state from one latent vector.
- Uses real pass completion and goal outcomes, not pre-action probability or xG
  as fake labels.
- Weights ball, score, player, and tactical features by semantic importance so
  sparse grids cannot dominate the loss.
- Uses one-step counterfactual planning. Repeating one action for several latent
  steps is no longer the default.
- Stores holdout metrics and a `planner_quality` score in every checkpoint.
- Scales planner bonuses by checkpoint quality and observation coverage.
- Loads v2/v3 checkpoints for replay compatibility, but gives them zero planning
  authority under the default quality gate.
- Splits train and validation data by complete match rather than adjacent rows.
- Excludes aggregate backfill traces when raw ball logs are loaded, preventing
  duplicate training examples.
- Uses independent pass and shot quality gates; a weak branch contributes zero.
- Uses a legal hold action as the counterfactual baseline.
- Uses ensemble disagreement and observation coverage to reduce OOD influence.
- Trains each dynamics member with independent Bayesian-bootstrap sample
  weights and rolls members forward separately at inference.
- Territorial progress is exposed only as `progress_delta`; the misleading
  historical xG alias has been removed.

## Environment flags

| Variable | Default | Meaning |
|---|---:|---|
| `MATCH_WORLD_MODEL` | `0` | Load the world-model runtime |
| `MATCH_WM_PLAN` | `1` | Enable quality-gated planning |
| `MATCH_WM_RECORD` | `0` | Record full-state transitions |
| `MATCH_WM_CHECKPOINT` | `data/world_model/latent_wm.pt` | Checkpoint path |
| `MATCH_WM_TRANSITION` | `gru` | `gru` or `transformer` |
| `MATCH_WM_TRANSITION_ENSEMBLE_SIZE` | `3` | Independently bootstrapped latent dynamics members; minimum `2` |
| `MATCH_WM_PLANNER_BLEND` | `0.30` | Maximum pass advantage blend |
| `MATCH_WM_SHOT_BLEND` | `0.25` | Maximum shot advantage blend |
| `MATCH_WM_IMAGINATION_STEPS` | `1` | Counterfactual rollout depth |
| `MATCH_WM_MIN_QUALITY` | `0.15` | Minimum branch confidence after coverage discount |

## Train and validate

```bash
python scripts/ensure_world_model.py --force --collect-pairs 48 --epochs 50
python scripts/validate_world_model.py
```

Manual workflow:

```bash
python scripts/collect_world_model_traces.py --pairs 48
python scripts/backfill_wm_from_ball_log.py
python scripts/train_world_model.py --epochs 50 --use-ball-log \
  --transition gru --transition-ensemble-size 3
python scripts/validate_world_model.py
```

Only validated v7 checkpoints are allowed to influence planning. A missing,
legacy, low-quality, or sparse-input model contributes a zero bonus.

## LLM decision fusion

When both world-model planning and the cognitive layer are enabled, the model
is used as evidence rather than as an unbounded controller:

- In-match coach triggers compare `hold`, `pass`, `cross`, and `shot` and send
  risk-adjusted values, event probabilities, confidence, and uncertainty to the
  coach LLM.
- Before a match, seven tactical presets are translated into explicit action
  mixtures and ranked with the same short-horizon evidence. This is a tactical
  proxy, not a full-match win probability.
- The LLM may disagree with the recommendation, but must select an evaluated
  candidate and provide a rationale. Out-of-set choices are constrained and
  recorded.
- The opponent's tactical intent is represented by a persistent posterior over
  seven interpretable archetypes. Every action is re-evaluated under every
  archetype, then ranked by posterior mean, disagreement, and tail risk.
- The LLM may submit a structured opponent hypothesis, but only with a listed
  archetype and named observable tactical features. Numeric likelihood gates
  the update and caps its posterior influence at `0.20`.
- Missing checkpoints, closed quality gates, or disabled planning preserve the
  original deterministic fallback path.

Outcome-linked records are appended to
`data/persistence/world_model_fusion.jsonl`. Generate the observational report
after enough matches have accumulated:

```bash
python scripts/evaluate_fusion_policy.py --min-records 20
```

The report compares agreement and disagreement groups but deliberately marks
them as non-causal. To estimate a tactical effect inside the simulator, use
paired seeds while holding the opponent policy fixed:

```bash
python scripts/evaluate_tactical_counterfactual.py \
  --team Brazil --opponent Scotland \
  --baseline balanced --treatment gegenpress \
  --samples 16 --match-seconds 900
```

Matched-seed estimates are causal only for the configured simulator; they do
not establish real-world football validity.

### Opponent-adaptive belief loop

The four observable opponent controls do not reveal the opponent's complete
intent. At each coach trigger, a sticky Markov prior is combined with a
tempered likelihood over `balanced`, `gegenpress`, `possession_control`,
`counter_attack`, `low_block`, `wing_play`, and `direct_vertical`. The packet
exposes the complete posterior, normalized entropy, tactical-switch
probability, observed feature vector, and observation likelihood of each
hypothesis.

Before the first live update, compatible cognitive logs provide a weak
cross-match meta-prior for the opponent. Compatibility requires both the exact
world-model checkpoint signature and the complete policy-environment
fingerprint. All decisions from one match are collapsed into one match-level
distribution, preventing repeated coach triggers from inflating sample size.
When raw four-control observations are present, the compiler reconstructs an
observation-only posterior and discards the prior/LLM-contaminated posterior;
legacy posterior fallback is permitted with lower trust. Meta-prior trust is
bounded at `0.45` and reduced under between-match instability, `watch`, or
`quarantined` recent-style drift.

For each hypothesis, the world model intervenes only on the opponent's four
tactical observation slots and re-runs `hold`, `pass`, `cross`, and `shot`.
Action ranking uses the posterior-weighted expected value with explicit
penalties for between-hypothesis standard deviation and the posterior-weighted
10% lower tail. This makes a high-entropy belief favor robust actions instead
of silently pretending the MAP opponent tactic is certain.

The coach LLM can challenge the numeric posterior through
`opponent_hypothesis`, which contains a listed tactical preset, confidence,
named evidence features, and a short rationale. The executor strips invented
features and multiplies the LLM confidence by feature grounding and the world
model's relative observation likelihood. The resulting pseudo-evidence has a
hard maximum influence of `0.20`; unsupported claims have zero or negligible
effect. The fused posterior then reweights the already-computed
counterfactuals before the selected action is reconciled.

The live detector separates gradual control drift from a tactical regime
change. It combines normalized feature displacement, Jensen-Shannon divergence
between predictive and updated beliefs, and posterior predictive surprise. A
moderate shift enters `watch` and needs a second consistent candidate; only an
extreme single observation can confirm immediately. Confirmation increments a
regime identifier, resets the within-match run length, and releases the sticky
old posterior: the fresh belief retains only a `0.15` meta-prior anchor.

The LLM may return `opponent_change_claim` with explicit from/to presets,
confidence, cited features, and a falsifiable rationale. Acceptance requires
alignment with the detector, observed feature changes in the claimed direction,
and likelihood support. This explanation is audit-only and always carries
`can_trigger_change_point=false`; it cannot create, confirm, cancel, or reset a
numeric change point.

Every adopted decision stores the posterior, MAP hypothesis, entropy, LLM
hypothesis audit, meta-prior provenance, change-point state, any non-controlling
change explanation, selected-action sensitivity, tail value, and all
hypothesis-conditioned values. Online reports summarize grounding acceptance,
LLM influence, belief entropy, inferred regimes, and decision sensitivity.
Because hidden intent has no direct truth label, these are explicitly
descriptive diagnostics rather than tactical-classification accuracy.

Successful full-fidelity counterfactual runs are also registered in
`data/persistence/tactical_counterfactuals.jsonl`. Before the next match, the
fusion layer combines this registry with the outcome-linked audit log to build:

- a team- and opponent-scoped strategy-memory summary;
- an evidence tier (`insufficient_history`, `observational_only`, or
  `matched_seed_simulator`);
- a conservative adjusted recommendation trust value.

Observational results can only lower the recommendation to advisory status.
Only matched-seed runs of at least ten simulated minutes, with fast mode off,
can provide simulator-level directional support. Historical evidence never
reopens a closed checkpoint/coverage quality gate. Use `--no-register` for
diagnostic counterfactual runs that must not influence later decisions.

## Online same-target calibration

During a micro match, every executed action can form an exact calibration tuple:

```text
observation before action + executed action -> predicted next observation
                                      compare with actual next observation
```

The online calibrator tracks weighted transition MSE, skill against persistence,
uncertainty/error correlation, and separate pass/shot trust factors. A branch
needs at least eight live transitions before it can affect planning. Online
evidence can only reduce checkpoint-derived confidence; it cannot create quality
for a branch whose offline gate is closed.

Coach action selections are also tracked prospectively. A selection is counted
as adopted only when the same team performs the matching high-level action after
the decision and within its horizon. This is temporal adoption evidence, not
proof that the LLM caused the action.

When the evidence gate is open, the selected evaluated action can also create a
one-shot policy intervention for that team's next feasible action. Its strength
is the configured maximum multiplied by the geometric mean of LLM confidence
and the selected candidate's uncertainty-adjusted world-model confidence. The
intervention adds at most `0.35` to one action logit, expires with the decision
horizon, and is consumed by the next action sample. It never forces an action or
bypasses role eligibility, shot distance, cooldown, normal sampling, or action
physics. The audit distinguishes the recommendation, LLM selection, applied
intervention, sampled action, and whether they matched.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MATCH_WM_LLM_ACTION_BRIDGE` | `1` | Enable the bounded one-shot policy bridge |
| `MATCH_WM_LLM_ACTION_BIAS_MAX` | `0.35` | Maximum action-logit bias, hard-clipped to `[0, 0.5]` |
| `MATCH_WM_LLM_ACTION_CONTROL_RATE` | `0.20` | Deterministic randomized share assigned to a zero-bias control arm, clipped to `[0, 0.5]` |
| `MATCH_WM_LLM_ACTION_MIN_ARM_SAMPLES` | `2` | Minimum samples per arm before within-match reliability feedback can activate |
| `MATCH_WM_LLM_OUTCOME_HORIZONS` | `0,60,180` | Comma-separated causal credit horizons in seconds; transition (`0`) is always included |
| `MATCH_WM_LLM_RESIDUAL_MIN_SAMPLES` | `6` | Realized forecast residuals required before prediction trust can reduce bridge strength |
| `MATCH_WM_ACTIVE_LEARNING` | `1` | Expose bounded information-gain opportunities to the coach LLM |
| `MATCH_WM_EXPLORATION_BUDGET` | `0.25` | Maximum rolling share of decisions marked for exploration, clipped to `[0, 0.5]` |
| `MATCH_WM_EXPLORATION_MAX_REGRET` | `0.08` | Largest predicted utility gap allowed inside the exploration safe set |
| `MATCH_WM_EXPLORATION_MIN_INFORMATION` | `0.45` | Minimum composite information value before exploration becomes eligible |
| `MATCH_WM_EXPLORATION_STRENGTH_SCALE` | `0.50` | Additional multiplier on the bounded action bias for exploration decisions |

Eligible decisions are deterministically randomized from the match seed and
decision identity. Treatment opportunities receive the bounded bias; control
opportunities go through exactly the same action path with a zero bias and do
not consume an extra LLM call. The report estimates the treatment effect on the
probability of executing the LLM-selected action, with standard error and a 95%
confidence interval. Once both arms meet the configured sample floor, adverse
evidence can only reduce later bias strength (`1.0`, `0.85`, `0.60`, or `0.25`);
the experiment can never increase it beyond the confidence-derived bound.

This randomized estimate supports a causal claim only about action selection in
the configured simulator. The next transition also records attack-oriented ball
progress, possession retention, net xG change, and score-difference change. A
declared short-horizon utility combines them as:

```text
goal_diff + 0.35*xg_net + 0.15*progress + 0.05*retention_edge
```

Outcome effects use stabilized inverse-propensity weighting, so reports remain
valid if treatment propensities differ, and persisted multi-match reports use
match-clustered standard errors. Strict policy readiness requires both arms and
requires each arm to appear across at least two match clusters. Single-match
estimates remain explicitly exploratory. Outcome evidence takes priority over
adoption evidence when reducing future bias: making the LLM-selected action more
frequent is not treated as proof that the action is better.

Credit is tracked at `transition`, `60s`, and `180s` by default. A later same-team
coach decision censors the older decision's remaining isolated-action windows,
preventing two interventions from being presented as one direct effect. In
parallel, the audit retains an intention-to-treat policy-regime view that includes
natural downstream coach decisions. The isolated view answers “what survived
without another intervention”; the regime view answers “what happened under the
deployed decision system.” Reliability feedback uses the longest ready regime
horizon, so a superficially good next action cannot hide a worse medium-term
trajectory.

For every evaluated action, the world model now predicts the same declared
policy utility at each credit horizon. Horizons beyond the 60-second action
encoding limit use an autoregressive action-persistence rollout (for example,
three 60-second segments for `180s`) and compound ensemble uncertainty with
rollout depth. Once the action is executed, the corresponding forecast is joined
to the realized outcome and produces raw and uncertainty-standardized residuals.

Residual calibration is reported globally, by horizon, and by action+horizon:
mean bias, MAE, RMSE, 90% uncertainty coverage, skill against a zero forecast,
and a bounded trust factor. The calibration summary is included in subsequent
LLM evidence packets. After the minimum sample count, the longest available
action-specific calibration can reduce the policy bridge; sparse history remains
neutral and calibration can never increase confidence or reopen a quality gate.

Completed cognitive match logs also form a cross-match contextual residual
memory. Every record carries a SHA-256 checkpoint signature and policy-utility
schema version; observations from another checkpoint are rejected. Context uses
team, opponent, executed action, horizon, pitch zone, score state, and match
phase, with conservative backoff in this order:

```text
team+opponent+context → team+action → action+context → action+horizon → horizon
```

Each group is fitted chronologically with at least eight samples: the first half
estimates a median residual bias, then separate validation and calibration
quarters decide whether the correction improves MSE and construct a 90%
split-conformal interval. Harmful corrections are reset to zero. At the next
match, same-checkpoint memory is loaded before the first LLM
coach trigger; corrected forecasts, intervals, evidence scope, and trust are
included in the decision packet. A confidence-weighted forecast adjustment is
hard-bounded to `±0.10` in candidate ranking, and memory trust can only reduce
the execution bridge. Conformal coverage relies on historical exchangeability
and should be re-audited under simulator, checkpoint, or policy drift.

Checkpoint identity alone is not enough: every decision now also carries a
SHA-256 policy-environment fingerprint over the complete micro-match and
cognitive configurations. Residuals are compiled only when both checkpoint and
environment fingerprints match, so a changed action temperature, physics
constant, cognitive cadence, experiment rate, or horizon cannot silently borrow
corrections from an incompatible regime.

The residual stream is continuously checked using two adjacent rolling windows
(up to 16 observations each). The monitor compares standardized mean shift,
residual-scale and RMSE ratios, a two-sample KS distance, and total-variation
drift across zone, score state, and match phase. It exposes three operational
states:

- `stable`: apply the validated correction and conformal interval normally.
- `watch`: halve the point correction, widen the interval by 50%, and cap
  cross-match memory trust at `0.75`.
- `quarantined`: remove the point correction entirely and cap memory trust at
  `0.50`; the raw checkpoint forecast remains visible to the LLM.

This state is part of the LLM evidence packet, the adopted-decision audit, and
the aggregate online report. Quarantine is not permanent: as new observations
arrive, the two rolling windows move forward; once both describe the new stable
regime, memory returns to `stable`. These thresholds are simulator safety
defaults rather than claims of real-football statistical validity and must be
recalibrated for materially different data rates or utility scales.

### Reducible and irreducible uncertainty

World-model forecasts expose three separate quantities. `epistemic_uncertainty`
is estimated from disagreement among independently bootstrapped dynamics and
outcome heads, plus the checkpoint's held-out quality gap; it represents model
ignorance that additional representative data may reduce. `aleatoric_uncertainty` uses the
within-head Bernoulli variance of pass/shot events and the held-out progress
residual scale; it represents match randomness that a larger dataset should not
be expected to remove. `uncertainty` composes both monotonically and remains the
quantity used for risk penalties and interval calibration.

The decomposition follows total-variance semantics but remains a simulator
proxy. Dynamics members have independent GRU/Transformer parameters and
bootstrap supervision while sharing the observation encoder and decoder; their
decoded ball, player, and whole-state trajectory disagreement is audited
separately. New checkpoints persist `progress_rmse`, ensemble mean/primary MSE,
and disagreement-error correlation from held-out data. Long-horizon rollouts
advance every member separately, then compound epistemic and aleatoric terms
before recomposition. Every forecast also carries its source and component
audit. A v6 checkpoint is expanded by exact copies of its primary transition,
so it remains replayable but reports zero dynamics disagreement and cannot pass
the v7 readiness gate.

The LLM is explicitly told that epistemic uncertainty may justify bounded data
acquisition, while aleatoric uncertainty can only increase caution. The online
report audits realized-error correlations and verifies the composition identity.
Deployments may make this mandatory with
`--require-uncertainty-decomposition` and
`--require-transition-ensemble`; legacy forecasts remain readable but cannot
satisfy those gates.

### Risk-constrained active learning

The coach packet now distinguishes exploitation from data acquisition. For each
action, the active-learning layer combines epistemic uncertainty,
multi-horizon forecast disagreement, global action novelty, and novelty in the
current team/opponent/zone/score/phase context. It also includes a bounded
opponent-hypothesis discrimination term: posterior entropy multiplied by how
strongly the action's predicted value varies across opponent hypotheses. This
is a probe-value proxy, not guaranteed information gain. Together these produce an auditable
`information_value`; it is not treated as match value.

An action can become an exploration candidate only when its predicted regret
from the greedy action is below the configured bound, its turnover probability
is close to the greedy action, its normal quality gate is open, and the rolling
exploration budget has capacity. The LLM then receives both actions and may
choose `exploit`, `explore`, or `decline`. An `explore` response is accepted only
for the packet's declared exploration action; otherwise the executor converts it
to exploitation. Valid exploration receives an additional intervention-strength
reduction and still passes through the existing randomized zero-bias control arm
and stochastic simulator action sampler.

Every registered decision stores the acquisition intent, expected information
value, predicted regret, and strength scale. Match and aggregate reports measure
exploration execution, action/context coverage, and the change in uncertainty at
the next prediction for the same action and context. A non-positive uncertainty
change is preserved rather than relabelled as learning. Because the coach's
choice to explore is not randomized and the runtime does not update neural
weights in place, this metric is descriptive acquisition evidence, not a causal
claim or proof of online model improvement. The collected outcomes feed the
checkpoint/environment-scoped residual memory and provide prioritized evidence
for later retraining.

These estimates still do not establish long-horizon match improvement or real
football validity. Aggregate experiments should keep the checkpoint, control
rate, tactics, and simulator configuration fixed.

### Action-conditioned opponent response and two-ply planning

Sequential coach-decision records now provide a second-order belief transition:
`P(next opponent tactic belief | current belief, our executed action)`. Evidence
is isolated by world-model checkpoint and complete policy-environment
fingerprint. Each match contributes equal total weight, so repeated triggers in
one match cannot imitate independent sample size. Training and validation are
chronological; an action-specific transition matrix receives non-zero authority
only when it improves match-clustered held-out Brier score over a transparent
sticky structural prior. Even after validation, learned authority is capped at
`0.50` and the transition remains explicitly observational rather than causal.

The decision packet uses this response belief for a two-ply policy proxy. It
combines the immediate robust action value with one continuation action chosen
against the complete predicted response posterior. It never chooses a separate
continuation using hidden opponent truth. To keep live inference bounded, the
second ply reuses the current-state hypothesis-conditioned payoff matrix; this
is not presented as a trajectory rollout or match-value forecast. The active
learning score includes a bounded response-information term, allowing safe
experiments to distinguish both current intent and possible reactions.

The LLM may submit one conditional `opponent_response_hypothesis` for an
evaluated first action. The engine validates its schema and model support, caps
its branch influence at `0.15` (with half authority when only the structural
prior exists), and records a non-persistent audit. LLM hypotheses can neither
write response memory nor make causal claims. Online evaluation scores realized
response forecasts against the next observed opponent belief with match-level
clustering and can require learned response evidence via
`--require-opponent-response-model`.

All diagnostics are stored in the per-match cognitive log. Aggregate them with:

```bash
python scripts/evaluate_online_world_model.py \
  --min-transitions 50 --min-policy-arm 8 --min-residual-samples 20 \
  --required-policy-horizon 60 --require-policy-effect \
  --require-outcome-calibration --require-uncertainty-decomposition \
  --require-transition-ensemble --require-opponent-belief \
  --require-opponent-meta-belief --require-opponent-change-detection \
  --require-opponent-response-model
```
