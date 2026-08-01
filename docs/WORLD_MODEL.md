# World Model v8

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
| `MATCH_WM_LLM_CONTRASTIVE_REPAIR` | `0` | Allow one explanation-only LLM revision after model counterevidence |
| `MATCH_WM_LLM_CONTRASTIVE_REPAIR_PATH_BUDGET` | `128` | Cumulative initial+revision member-trajectory ceiling (`16..128`) |

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
  --transition gru --transition-ensemble-size 3 \
  --semantic-event-loss-weight 0.20
python scripts/validate_world_model.py --require-semantic-event-heads
```

Only validated v7+ checkpoints are allowed to influence transition planning;
learned semantic-event blending additionally requires v8 evidence. A missing,
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
the transition-ensemble readiness gate. A v7 checkpoint loads into v8 with
neutral, explicitly untrained semantic heads and retains its prior transition
behavior.

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

The decision packet uses this response belief for a two-ply policy. It combines
the immediate robust action value with one continuation action chosen against
the complete predicted response posterior. It never chooses a separate
continuation using hidden opponent truth. A changing-action rollout interface
now advances every dynamics member through action one and re-encodes action two
from the predicted ball state. The resulting continuation values propagate
first- and second-step uncertainty and blend with the transparent current-state
proxy; they do not replace it outright.

Predicted-state search has its own authority gate. Training constructs
state-aligned adjacent transition pairs separately inside training and grouped
validation matches. Training pairs supervise a genuinely autoregressive
changing-action rollout: the first decoded prediction is re-encoded as the
second-step state, rather than replacing it with the observed intermediate
state. A one-step warmup followed by a linear curriculum limits early exposure
bias, while independent Bayesian-bootstrap weights keep dynamics members from
receiving identical trajectory supervision. Configure the final loss with
`--multi-step-loss-weight` (default `0.25`) and the warmup with
`--multi-step-warmup-fraction` (default `0.20`).

Validation compares that changing-action rollout against persistence and stores
ensemble-versus-member gain plus disagreement/error correlation. The live
planner requires an explicit autoregressive training contract, at least 32
training and validation pairs across four match groups in each split, a trained
transition ensemble, recorded non-zero optimization steps, non-negative
ensemble/calibration evidence, positive error
reduction, and at least `0.02` skill. Thus an older checkpoint that merely ran a
two-step validation cannot acquire planning authority. Authority is capped at
`0.50`, scaled by uncertainty calibration, and reduced again by path
uncertainty. A maximum of 48
continuation/hypothesis evaluations is allowed per decision (hard ceiling 112),
and unevaluated low-posterior hypotheses retain proxy values. Legacy checkpoints
without this evidence execute no speculative rollout and remain replayable. The
active-learning score still includes a bounded response-information term.
The live budget is configurable with `MATCH_WM_TRAJECTORY_BRANCH_BUDGET`
(`16..112`); it is included in the policy-environment fingerprint, so evidence
from different search budgets cannot silently share scoped memories.

The LLM may submit one conditional `opponent_response_hypothesis` for an
evaluated first action. The engine validates its schema and model support, caps
its branch influence at `0.15` (with half authority when only the structural
prior exists), and records a non-persistent audit. LLM hypotheses can neither
write response memory nor make causal claims. Online evaluation scores realized
response forecasts against the next observed opponent belief with match-level
clustering and can require learned response evidence via
`--require-opponent-response-model`. It separately audits rollout budgets,
validation provenance and deployment coverage via
`--require-two-step-trajectory-planning`.

### Empirically gated LLM semantic residual critic

The coach LLM can now make a falsifiable `world_model_critique` for the action
it actually selects and one declared forecast horizon. The claim must say that
the model is overestimating, underestimating, or neutral; it carries bounded
confidence and may cite only semantic evidence that is present in the packet
(match zone, score state, phase, opponent belief/response, trajectory audit,
uncertainty, or observed tactical shape). A critique for another action, an
unevaluated horizon, or absent evidence is rejected.

Every accepted critique initially runs in shadow. It proposes at most `0.08`
policy-utility correction but cannot alter the stored neural forecast, update
model weights, or write its own reliability memory. After the selected action
is actually executed, the proposal is joined to the same-horizon realized
policy utility. Cross-match memory is isolated by checkpoint and complete policy
environment and by a hash of the LLM model plus critic prompt contract. Changing
the language model or critic contract therefore resets authority to shadow mode.
The compiler gives each match equal weight, fits a non-negative correction scale
on the chronological first half, and activates only when the held-out second
half improves MSE by at least two percent. At least two training and two
validation matches are required.

Even after validation, reliability is capped at `0.35`; the resulting live
ranking adjustment is capped at `0.04` and the original world-model forecast
remains immutable for calibration. Thus semantic LLM judgment can complement a
numeric model only after demonstrating repeatable residual skill, while harmful
or merely eloquent criticism retains zero authority. Online diagnostics report
match-clustered baseline/corrected MSE and can require this path with
`--require-llm-semantic-critic`.

Execution authority is asymmetric: a validated `overestimate` critique may
reduce the selected action's policy-bridge strength by up to 50 percent, while
an `underestimate` critique cannot increase bridge strength. This gives the
semantic channel a learned safety-brake role without allowing it to manufacture
additional control authority.

### Multi-scale state semantics and paired LLM event forecasts

Each multi-horizon neural trajectory now carries an auditable `state_scales`
projection instead of exposing only one policy-utility number. The short scale
summarizes oriented ball progress, speed and possession; the tactical scale
summarizes final-third entry, territorial gain, pressure around the ball and
tactical-shape movement; the strategic scale summarizes score-difference and
clock evolution. These values are deterministic projections of every dynamics
member, not extra observed facts or an independently validated causal model.
Binary event probabilities use Jeffreys-smoothed member frequencies, so a small
ensemble cannot report unjustified probability zero or one.
Legacy-expanded or otherwise untrained transition ensembles emit neutral `0.5`
event probabilities instead of presenting identical members as confidence.

V8 adds one semantic-event head per independently bootstrapped dynamics member.
Each head sees the current latent, its own member-consistent predicted future
latent, and the action context. Labels are derived only from realized future
states inside the same training split; the LLM never supplies supervision.
Training uses unweighted binary cross-entropy as a proper scoring rule at both
one-step and changing-action autoregressive two-step depths. The existing
one-step warmup and curriculum also ramps the event objective, configurable via
`--semantic-event-loss-weight` (default `0.20`).

Grouped validation gives every match equal weight and compares the learned
Brier score against both the training-split event prevalence and the transparent
member-frequency projection. Runtime blending is authorized separately for
each event and exact rollout depth only after at least 32 samples, four matches,
four positive and negative examples, positive skill, non-negative ensemble and
disagreement evidence, and calibration error at most `0.20`. Learned authority
is capped at `0.50`; three-step and longer horizons remain projection-only until
they receive their own training and validation evidence. Strict offline
validation requires at least two active events at both supported depths with
`--require-semantic-event-heads`. Online deployment evidence can independently
require at least four matches where bounded learned/fused probabilities both
beat the transparent projection via `--require-learned-semantic-events`.

The coach may submit one `world_model_event_hypothesis` for its selected action
and one evaluated horizon. It must choose an event already exposed by the
numeric trajectory (`retain_possession`, `enter_final_third`,
`positive_territorial_shift`, or `improve_scoreline`), declare `occur` or
`not_occur`, provide confidence in `[0.5, 1]`, and cite only short, tactical, or
strategic scales actually present in that forecast. Invalid actions, horizons,
events, absent scales, and non-finite model probabilities are rejected.

Accepted hypotheses are strictly shadow-only. The LLM probability and frozen
neural probability are persisted before the outcome and resolved from the same
simulator-native future record at exactly the declared horizon. The log stores
paired Brier errors and their difference; neither prediction can mutate the
other, change action ranking, write model weights, or claim causal authority.
Aggregate diagnostics weight matches equally, reject malformed persisted rows,
report calibration error and event/horizon profiles, and require at least four
matches for strict readiness. Evidence is isolated by a hash of the LLM model
and event-contract version plus the world-model checkpoint and complete policy
environment; mixed or unspecified provenance closes the gate. Audit this path with
`--require-llm-semantic-events`. Readiness means that the paired shadow
evaluation pipeline is trustworthy; it does not grant the semantic event
channel control authority.

### Shadow event-conditioned continuation options

The coach may also propose one `world_model_event_option`: its already selected
first action, one supported event and horizon, and two different continuation
actions for event occurrence versus absence. The contract is deliberately a
binary, depth-limited strategy tree rather than a free-form plan. The first
action must equal the coach selection, the horizon must already exist in that
candidate's packet, and only validated one- or two-step rollout depths are
eligible.

The world model evaluates only those two proposed continuations from every
member-predicted state. Transparent member event indicators are blended with a
learned semantic head only when that exact event/depth gate is open, with
learned authority capped at `0.50`. Jeffreys branch shrinkage prevents an empty
or tiny event branch from claiming extreme value. A decision defaults to at
most 16 continuation evaluations, has a hard ceiling of 32, and also counts the
nested ensemble work inside each continuation under a 144 member-trajectory
hard ceiling. It fails closed if both branches cannot be evaluated for every
member. The audit records member
support, conditional and fixed-branch values, model calls, budget, checkpoint,
environment, and LLM option-contract signature.

This tree is shadow-only: it cannot schedule or execute either continuation,
change the selected first action, mutate policy, write world-model weights, or
acquire causal authority. If the simulator later executes the proposed first
action, the declared event is resolved at the exact horizon. The next same-team
action within 120 seconds is then attached as a descriptive follow-up and
compared with the corresponding proposed continuation. Cross-match readiness
requires at least four compatible matches, enough resolved events and observed
follow-ups, zero malformed rows, respected budgets, complete provenance, and
all non-control safety flags. Use `--require-llm-event-options` to require that
audit path. Continuation agreement is neither a causal effect nor evidence that
the conditional option improved match value.

V2 closes the remaining temporal-value loop. When the runtime exposes the
multi-horizon policy-utility head, each proposed continuation is scored on every
event-conditioned member state using the same utility definition later measured
by the simulator. If and only if the proposed first action occurred, the event
resolved, the next same-team action naturally matched the corresponding branch,
and a pre-action baseline was captured, that branch prediction receives a new
delayed outcome window. The realized continuation utility, residual, absolute
error, and squared error are then attached to the immutable option audit.
Mismatched actions receive no counterfactual label; missing baselines and expired
follow-up windows fail closed. A newer same-team coach decision terminates an
unobserved continuation and censors an already anchored value window, preventing
one plan from borrowing another plan's behavior or outcome.

Aggregate value calibration gives every match equal weight and reports bias,
MAE, MSE, and skill against a zero-utility prediction. Strict readiness requires
enough realized matching branches, at least four compatible matches, zero
malformed value rows, and positive match-clustered skill via
`--require-llm-event-option-values`. This establishes calibration of the
world-model value estimate under naturally selected matching branches. It does
not estimate what would have happened under an unobserved continuation and does
not authorize the LLM option to control play.

### Model-checked contrastive explanations

The coach may submit one `world_model_contrastive_claim` that explains its
selected action relative to a different evaluated action. The claim must name
one evaluated one- or two-step horizon and exactly one schema-locatable context:
score, match phase, own tactics, opponent tactics, or crowd state. It must also
state whether that factor supports or opposes the selected action's margin.
Free-text latent causes such as momentum are rejected because the world model
cannot intervene on them reproducibly.

The checker keeps the observed state fixed except for the declared context,
which is replaced by its simulator-schema neutral reference: level 0-0 score,
mid-match clock, balanced tactical controls, or neutral crowd. It evaluates the
selected and alternative actions on both observed and neutralized states using
the same runtime policy-utility head and uncertainty penalty. Cross-match
residual corrections are deliberately excluded from both sides so historical
memory cannot manufacture local factor sensitivity. The difference between
the two selected-versus-alternative margins is compared with the LLM's claimed
direction. A factor already at its neutral reference is rejected rather than
inventing sensitivity. Two-step probes require the validated trajectory gate;
all four predictions are limited by a hard 64 member-trajectory budget.

The audit records changed feature indices, intervention magnitude, both margins,
uncertainties, quality factors, directional effect, checkpoint/environment and
LLM-contract provenance. It is always shadow-only: faithfulness cannot change
the selected action, rewrite either forecast, or create policy authority.
Cross-match diagnostics give every match equal weight and require at least four
compatible matches, zero malformed audits, bounded computation, a directional
faithfulness rate of at least `0.60`, and non-trivial mean factor effect. Enable
that readiness check with `--require-llm-contrastive-faithfulness`.

This is a model-faithfulness test, not a real-football causal explanation. It
answers whether the stated reason agrees with the configured model's local
context sensitivity; it cannot prove that neutralizing the factor in reality
would cause the same action-margin change.

### One-shot counterevidence repair

When `MATCH_WM_LLM_CONTRASTIVE_REPAIR=1`, an accepted but directionally
unfaithful contrastive claim may receive exactly one explanation-only revision.
The second prompt contains the frozen selected action, the two measured margins,
the signed factor effect, the neutral reference, and a finite contract listing
evaluated actions, validated horizons and allowed factors. It contains no
permission to revise tactics, controls, action confidence or model output.

The revision is schema-validated and then sent through the same world-model
probe. An attempted action change is rejected before evaluation. The initial and
revision probes share a cumulative member-trajectory budget (default and hard
maximum `128`), and an exhausted budget prevents the additional LLM call rather
than soliciting an unverifiable answer. A repair succeeds only when the revised
claim is accepted, becomes directionally faithful and improves signed effect;
otherwise the initial claim remains the effective audit explanation. No result
from this loop can alter the already selected action or policy bridge.

Online diagnostics report attempts, accepted revisions, match-clustered repair
success, directional-effect gain, single-call compliance, cumulative budgets,
action immutability and provenance. Strict readiness requires at least four
compatible matches, a `0.50` repair-success rate and positive mean effect gain
with `--require-llm-contrastive-repair`. This measures whether model feedback
helps the LLM correct its own model-relative explanation; it remains unrelated
to real-world causal validity or match-value improvement. Both the enable flag
and cumulative path budget are part of the policy-environment fingerprint, so
repair evidence cannot pool across different cost or deliberation regimes.

### Transparent trajectory modes and chance constraints

Every multiscale prediction now exposes `trajectory_modes`, a transparent
partition of trained transition-ensemble members. Members are grouped by exact
terminal and pathwise downside signatures: loss of possession, negative
territorial shift, worsening scoreline, and failure to enter the final third.
For multi-step forecasts the model returns every intermediate member state from
one autoregressive rollout; it never constructs a path by joining independent
horizon calls. The endpoint must exactly match the final member forecast or the
path is rejected. The three largest signatures are reported explicitly;
smaller signatures are combined into a residual mixed tail. Each mode retains
its member indices, probability, terminal and interval downside rates, progress,
possession retention and score-difference movement. Normalized mode entropy
describes disagreement between these alternative futures. These modes are a
readable decomposition of the existing ensemble, not an additional model or
source of truth. Untrained ensembles expose neutral `0.5` probabilities and
untrusted counts, so they cannot issue certificates.

The coach may declare one `world_model_risk_constraint` for its selected action,
one evaluated one- or two-step horizon, one downside event, and a maximum
violation probability. The audit compares the Jeffreys-smoothed member frequency
with that threshold and separately computes a one-sided 90% Wilson upper bound
from the raw member count. A constraint is conservatively certified only when
the upper bound—not merely the point estimate—is below the declared limit.
Two-step certificates require the same held-out planning gate as predicted-state
search; deeper horizons and untrained modes fail closed.

`risk_scope=terminal` evaluates the declared event only at the horizon endpoint.
`risk_scope=within_horizon` instead evaluates member-aligned intermediate states
and is available only for an exactly two-step validated rollout. Loss of
possession, negative territorial shift and scoreline worsening use ANY-step
semantics; failure to enter the final third uses NEVER-entered semantics, so an
early midfield state does not become a false violation after a later successful
entry. The live simulator mirrors those definitions by monitoring the realized
interval from the executed action to the declared horizon. It records the first
and last observation, maximum sampling gap, and whether a downside occurred at
any point. A negative label is accepted only with at least two observations and
coverage whose start and maximum gap are within the explicit tolerance; an
eligible due certificate with incomplete monitoring remains visibly unscored.

Certification is always shadow-only. It cannot veto, authorize, strengthen, or
change the selected action. If the simulator later executes that exact action,
the declared horizon is paired with its naturally realized outcome and scored
with Brier loss; certified violations are explicitly counted as false-safe
certificates. Cross-match diagnostics are provenance-isolated and match
clustered. Strict readiness requires enough realized and certified observations
across at least four matches, zero malformed rows, Brier at most `0.25`, and a
certified violation rate no greater than the mean declared threshold. Enable it
with `--require-llm-risk-certificates`. This validates probabilistic calibration
inside the configured simulator; it is neither a causal safety guarantee nor a
license for LLM control.

### Distributional policy utility and model-checked risk preference

Each trained transition member now carries its own value through the exact
policy-utility definition later used by the simulator: score-difference change,
xG-net movement, territorial progress and possession retention. The runtime
preserves the transition-member axis through the outcome heads and reports the
full member-value vector, mean, standard deviation, `q10/q25/q50/q75/q90`,
worst-quartile CVaR, best-quartile mean, and Jeffreys-smoothed upside/downside
probabilities. Untrained or single-member dynamics fail closed rather than
presenting a fake distribution. This first distribution has
`distribution_scope=epistemic_member_only`: it measures disagreement among
learned transition members and is not presented as complete outcome randomness.

Contextual residual memory now retains signed `q10/q25/q50/q75/q90` errors from
its held-out calibration split in addition to the conformal interval radius.
When at least four such calibration outcomes are available and drift is not
quarantined, the runtime crosses centered transition-member values with these
residual scenarios. The resulting `calibrated_predictive` lattice is used for
decision-distribution summaries and proper scoring. It keeps the original
member values and residual offsets separately and reports epistemic-member,
residual-outcome, and total lattice variance. The additive variance identity is
recomputed from the two frozen axes; it is a property of the equal-weight lattice,
not a claim that the uncertainties are causally or statistically independent.
Insufficient residual history leaves the useful epistemic distribution visible
but closes predictive calibration readiness.

For every evaluated horizon, `distributional_action_frontiers` identifies the
actions that are nondominated across three deliberately distinct objectives:
mean utility, lower-tail `CVaR_25`, and probability of positive utility. It also
lists each criterion's leaders. This exposes real tradeoffs—for example, a shot
may lead on upside while holding leads on the lower tail—instead of hiding all
risk preferences inside one scalar rank. The frontier is evidence only and does
not modify the existing policy recommendation.

The coach may provide one `world_model_distributional_claim` comparing its
selected action with one evaluated alternative at the same horizon. It must name
exactly one criterion, the exact `distribution_scope`, and state whether the
selected action is better, worse or approximately equal. Both actions must have
the same scope. The engine recomputes that relation with an explicit
tolerance and records directional faithfulness. The claim cannot change either
distribution, the selected action, or policy authority.

If the exact selected action is naturally executed, the frozen distribution is
paired with the same-horizon realized simulator utility. Evaluation reports
empirical CRPS, pinball loss at `q10/q50/q90`, central-80% coverage, median
calibration, point MAE/MSE and claim faithfulness. Cross-match aggregation
recomputes every proper score from the stored scenario vector and, for predictive
claims, reconstructs that vector from the frozen member and residual axes. It
gives each match equal weight, rejects provenance mixing, and exposes claims that reached
their horizon without a score. Strict readiness requires sufficient evidence
over at least four matches, directional faithfulness of at least `0.60`, central
80% coverage between `0.55` and `0.98`, median frequency between `0.25` and
`0.75`, CRPS no worse than mean-point MAE, and every scored distribution to be a
valid held-out-residual `calibrated_predictive` lattice. Enable it with
`--require-llm-distributional-decisions`. These are simulator-distribution
calibration claims, not causal estimates of choosing one action over another.

Each action also exposes `temporal_utility_paths` when at least two ordered
horizons share a valid distribution scope. Epistemic paths connect the same
transition-member index through time. When the same checkpoint, policy
environment, context, and decision record provide enough multi-horizon realized
residual paths, predictive paths additionally use residual-rank templates learned
from a reference/held-out split. Templates preserve observed cross-horizon rank
patterns and their repeated empirical frequency, and are shared across candidate
actions so pathwise comparisons retain one exogenous scenario identity. Drift in
`watch` or `quarantined` state disables this memory. If compatible history is
insufficient, the report names and uses a transparent comonotonic residual-rank
fallback. Neither empirical templates nor the fallback constitute a calibrated
temporal copula or calibrated joint probability. Reports include path minimum
utility, lower-tail path-minimum CVaR, maximum drawdown, ever-downside, downside
recovery, and positive-to-negative reversal scenario rates. Every horizon is a
cumulative forecast from the same origin, so horizon values are not treated as
independent incremental rewards.

The executed action's complete path forecast is frozen prospectively as
`world_model_temporal_path_forecast`. When every declared horizon has resolved,
one record-level `world_model_temporal_path_evaluation` scores the realized path
exactly once. Binary path events (ever-downside, terminal-upside, recovery, and
positive-to-negative reversal) use Brier scores; path-minimum utility and maximum
drawdown use empirical CRPS. Whenever empirical rank templates are active, the
certificate also freezes a counterfactual *coupling* benchmark: the same marginal
member and residual distributions joined with transparent comonotonic ranks.
Thus the evaluation asks whether learning temporal dependence improves path
forecasts without confusing that comparison with an action counterfactual.

Cross-match diagnostics give each match equal weight and require at least eight
empirically coupled realized paths spanning at least four matches. Empirical
coupling is `validated` only when its event Brier, path-minimum CRPS, and
maximum-drawdown CRPS do not trail the same-marginal benchmark by more than
`0.02`, while mean event calibration gap is at most `0.25`. A
`degraded` result closes temporal-rank memory lookup; insufficient evidence keeps
the feature explicitly exploratory. Use
`--require-temporal-path-calibration` to make this a strict online readiness
gate. Forecasts, scores, and their benchmark remain shadow-only, non-causal, and
make no calibrated joint-probability claim.

### Model-checked opponent information queries

The coach may submit one `opponent_information_query` for its selected action.
It chooses only an observable tactical feature, an already evaluated future
horizon, whether the question is intended to reduce opponent uncertainty or
resolve an action choice, and one shadow action for each high/low observation.
The LLM cannot choose a threshold, likelihood model, posterior update,
information-gain value, or action value. The engine fixes the
binary observation at `feature >= 0.5` with a bounded observation-noise model and
evaluates all four tactical features, not only the feature selected by the LLM.

For every feature, the world model reports the forecast observation rate,
posterior under high/low observations, normalized information gain, best action
inside each observation branch, and expected decision value of information. The
prior is the selected action's predicted opponent-response posterior when
available, otherwise the current opponent belief. Action values come entirely
from the existing opponent-hypothesis counterfactual grid. The leaderboard ranks
information gain first for uncertainty-reduction queries and adaptive decision
value first for action-resolution queries. It therefore exposes whether the LLM
found the model's best question for its declared purpose or merely supplied a
plausible narrative. Rank, objective regret, and normalized objective efficiency
make that comparison explicit rather than relying on explanation quality.

The fixed Gaussian observation model remains the cold-start benchmark, but it
is no longer assumed to be perfectly calibrated. Same-checkpoint and
same-environment realized query scores are grouped by horizon and feature, then
split chronologically into an earlier reference half and a later validation
half. The reference half learns only one bounded logit-intercept correction.
Only the first valid query for a horizon-feature group in each match is retained,
so repeated within-match decisions cannot manufacture confidence. That correction
becomes active only with at least eight separate matches in each half
and strictly better held-out Brier score than the raw forecast; otherwise lookup
falls back to the untouched Gaussian probabilities. A feature-only group can be
used when an exact horizon group has insufficient evidence.

When active, the same offset is applied to every opponent hypothesis likelihood,
not merely to the aggregate displayed probability. High/low posteriors,
information gain, adaptive value, and contingent-action regret are consequently
recomputed from one coherent Bayesian branch model. Each audit freezes both raw
and calibrated likelihoods and probabilities, and each realized score preserves
both Brier benchmarks. The online gate rejects an active calibration whose
match-clustered Brier score becomes worse than raw. Version-2 query history is
not recycled as training data because it cannot prove that its stored forecast
was an uncalibrated baseline.

The same audit evaluates the proposed high/low actions against the best action in
each posterior branch. It reports branch regrets, expected policy regret, worst
branch regret, and a model-owned consistency certificate (`expected <= 0.03`,
`worst branch <= 0.05`). When the observation later resolves, the score records
the applicable proposed action and its branch regret. This is a test of conditional
reasoning only: neither action is queued or automatically executed.

At the declared horizon, the simulator records all four observable opponent
tactical controls and computes Brier scores for the frozen feature forecasts.
Cross-match diagnostics require at least four scored matches, compatible model
provenance, no missing or malformed scores, useful query-purpose agreement, and
frequent agreement with the model-owned query ranking. Enable the strict gate
with `--require-opponent-information-queries`. This evaluates forecasted
observations—not latent tactical truth. Queries are shadow-only: they cannot
change the current action, schedule a future action, or update opponent memory.

### Model-checked multi-horizon risk preferences

The coach may additionally declare one `world_model_risk_preference`. This is
not a free-form score: it contains the selected action, an exact distribution
scope, weights over two to four evaluated horizons that must sum to one, bounded
loss aversion in `[1, 4]`, diminishing sensitivity in `[0.5, 1]`, and a maximum
acceptable preference regret. The utility reference is fixed at zero, so the
coach cannot move the gain/loss boundary after seeing the action distributions.

The world model applies that same prospect-value transform to every scenario of
all four actions at every declared horizon. It independently reports expected
preference value, preferred actions, selected-action regret at each horizon,
weighted aggregate regret, and whether both the local and aggregate regrets fit
the declaration. The LLM cannot submit its own transformed values, change a
forecast, or acquire action authority. Mixed distribution scopes and incomplete
action/horizon evidence fail closed.

The preference audit then compares all four actions *inside each aligned temporal
scenario* before aggregating across scenarios. It reports mean, q90 and worst
pathwise regret, the rate within declared regret, action-rank switches over time,
and the worst/failed paths. This ordering matters: comparing expected action
values first can hide a choice that performs poorly in many member-consistent
futures. The model-owned minimum pathwise robustness rate is `0.75`; this is a
sensitivity threshold over coupled scenarios, not a probability guarantee.

To prevent post-hoc parameter tuning from becoming a sophisticated form of
rationalization, a second model-owned audit perturbs loss aversion by `±0.5` and
diminishing sensitivity by `±0.1`, clipped only to the contract bounds. It
crosses this parameter grid with a nominal evidence case, every aligned
leave-one-transition-member-out case, and—for predictive lattices—every aligned
leave-one-residual-quantile-out case. The same omitted axis index is used for all
actions and horizons. Misaligned residual quantile levels fail closed. The audit
reports the fraction of joint cases within declared regret, worst aggregate and
per-horizon regret, and rationalization fragility. A nominally coherent choice
therefore remains distinguishable from one that is genuinely robust to nearby
preferences and world-model evidence.

When the selected action naturally occurs, each declared horizon is paired only
with its same-horizon realized utility. The evaluator transforms both the frozen
scenario vector and realized value under the original preference, then reports
CRPS, point error, and central-80% coverage in preference-value space. Aggregate
diagnostics revalidate the complete prospective audit digest, scenario transform,
regret calculation, and realized score. No realized regret is claimed because
the unchosen actions do not have observed counterfactual outcomes. Strict
readiness additionally requires calibrated predictive lattices, four matches,
zero malformed or missing eligible scores, at least `0.60` prospective
consistency, fully certified local preference robustness, and pathwise temporal
robustness, central-80% coverage in
`[0.55, 0.98]`, and CRPS no worse than the
mean forecast's absolute error. Enable it with
`--require-llm-risk-preferences`.

All diagnostics are stored in the per-match cognitive log. Aggregate them with:

```bash
python scripts/evaluate_online_world_model.py \
  --min-transitions 50 --min-policy-arm 8 --min-residual-samples 20 \
  --required-policy-horizon 60 --require-policy-effect \
  --require-outcome-calibration --require-uncertainty-decomposition \
  --require-transition-ensemble --require-opponent-belief \
  --require-opponent-meta-belief --require-opponent-change-detection \
  --require-opponent-response-model --require-two-step-trajectory-planning \
  --require-llm-semantic-critic --require-llm-semantic-events \
  --require-learned-semantic-events --require-llm-event-options \
  --require-llm-event-option-values \
  --require-llm-contrastive-faithfulness \
  --require-llm-contrastive-repair \
  --require-llm-risk-certificates \
  --require-llm-distributional-decisions \
  --require-llm-risk-preferences
```
