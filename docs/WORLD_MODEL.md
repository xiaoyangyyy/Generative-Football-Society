# World Model v6

The world model learns action-conditioned match transitions for counterfactual
pass and shot planning. It is an optional advisory layer: the deterministic
match engine remains authoritative.

## Core design

```text
full observation + candidate action
              |
 CNN grid + player attention + context encoder
              |
         GRU/Transformer
              |
        residual state delta
              |
 next observation + bootstrap outcome ensemble
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
python scripts/train_world_model.py --epochs 50 --use-ball-log --transition gru
python scripts/validate_world_model.py
```

Only validated v6 checkpoints are allowed to influence planning. A missing,
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

The intervention adoption rate is still not a causal effect estimate. Measuring
that requires matched seeds or randomized no-bias controls under the same model,
checkpoint, tactics, and simulator configuration.

Both diagnostics are stored in the per-match cognitive log. Aggregate them with:

```bash
python scripts/evaluate_online_world_model.py --min-transitions 50
```
