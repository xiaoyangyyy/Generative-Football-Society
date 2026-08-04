import json
import re

from src.simulation.generation_pipeline import GenerationPipeline
from src.simulation.llm_gateway import LLMGateway, get_shared_llm_gateway


def _sanitize_coach_reasoning(text):
    if not text or not isinstance(text, str):
        return text
    t = text
    t = re.sub(r"(?is)\bCorrection\s*:\s*.*", "", t)
    t = re.sub(r"(?is)\bRe-?evaluating\s*:\s*.*", "", t)
    t = re.sub(r"(?i)\bWait\s*,\s*", "", t)
    t = " ".join(t.split())
    return t[:2400]


# Fourth-wall / meta phrases stripped from published media copy (deterministic backstop).
_MEDIA_SCRUB = [
    (re.compile(r"(?is)something\s+is\s+wrong\s+with\s+the\s+simulation[^\.\!?]*[\.\!?]?", re.I), " "),
    (re.compile(r"(?i)\bPoissonMathGlitch\b|\bFakePress\b|\bKnockoutPrudenceFail\b"), " "),
    (re.compile(r"(?i)\bpoisson\s*math\s*glitch\b|\bmath\s*glitch\b"), " "),
    (re.compile(r"(?i)\brng\b|\bthe\s+simulation\b|\bthis\s+simulation\b"), " "),
    (re.compile(r"(?i)\bglitch\b|\bbroken\s+game\b|\bengine\s+bugs?\b"), " "),
    (re.compile(r"(?i)#\s*\w*(?:glitch|simulation|poisson)\w*"), " "),
    # This simulation resolves draws via penalties right after regulation (no extra-time phase).
    (re.compile(r"(?i)\bextra\s*time\s+and\s+penalt(?:y|ies)\b"), "penalties"),
    (re.compile(r"(?i)\bafter\s+extra\s+time\b"), "after regulation"),
]


def _sanitize_media_text(text, facts_ledger):
    if not text or not isinstance(text, str):
        return text
    t = text
    for pat, rep in _MEDIA_SCRUB:
        t = pat.sub(rep, t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) < 20:
        sc = (facts_ledger or {}).get("score", "?")
        w = (facts_ledger or {}).get("winner", "")
        tail = f" Final scoreline {sc}."
        if w and w != "draw":
            tail = f" {w} took the result ({sc})."
        t = (
            "Match coverage stays grounded in the official score and tactics on the sheet — debates continue in the stands."
            + tail
        )
    return t


def _sanitize_media_payload(parsed, facts_ledger):
    if not isinstance(parsed, dict):
        return parsed
    for key in ("mainstream", "social_chaos_post"):
        if key in parsed:
            parsed[key] = _sanitize_media_text(parsed[key], facts_ledger)
    return parsed

class SimulationLLM:
    def __init__(self, gateway: LLMGateway | None = None):
        self.gateway = gateway or get_shared_llm_gateway()
        self.client = self.gateway.client
        self.model = self.gateway.config.model
        self.request_timeout_s = self.gateway.config.timeout_s
        self.max_retries = self.gateway.config.max_retries
        self.generation_pipeline = GenerationPipeline()
        self.generation_audits = []

    def _extract_json_object(self, content):
        return self.gateway.extract_json_object(content)

    def _call_llm(self, system_prompt, user_prompt, json_mode=False, temperature=0.8):
        return self.gateway.complete(
            system_prompt,
            user_prompt,
            json_mode=json_mode,
            temperature=temperature,
        )

    def coach_decide_tactics(
        self,
        team_name,
        my_info,
        opp_name,
        opp_info,
        world_model_decision_support=None,
    ):
        system = f"""You are the Head Coach of {team_name}.
        GAME THEORY: infer opponent tendencies and pick formation/controls to exploit weaknesses.
        WORLD MODEL: when a decision packet is supplied, treat it as uncertain,
        short-horizon evidence rather than a full-match forecast. Select only one
        evaluated tactical candidate when its quality gate is open. You may
        disagree with its recommendation, but explain the opponent-specific reason.
        If fusion_reliability is present, use adjusted_recommendation_trust and
        its evidence_tier to calibrate reliance. It can never reopen a closed gate.
        strategy_memory is observational context, not proof that a tactic caused
        past results. Matched-seed evidence is causal only inside the simulator.
        Reasoning rules (required):
        - At most 3 sentences in "reasoning".
        - No arithmetic of stat numbers (no "0.78 - 0.24", no "Correction:", no "Wait,", no "Re-evaluating").
        - Qualitative comparisons only (e.g. "we are fresher" / "they look fragile"), not recalculated decimals.
        - One coherent narrative; never publicly revise your own line mid-answer."""
        decision_packet = json.dumps(
            world_model_decision_support or {
                "available": False, "reason": "not_provided",
            },
            ensure_ascii=False,
        )
        user = f"""MY STATE: {my_info}
        OPPONENT: {opp_name} | STATE: {opp_info}
        WORLD_MODEL_DECISION_SUPPORT: {decision_packet}
        Return STRICT JSON format:
        {{
          "formation": "X-Y-Z",
          "style": "Strategy description (short)",
          "tactical_preset": "one of balanced|gegenpress|possession|counter|low_block|wing_play|direct",
          "reasoning": "Game-theoretic justification (max 3 sentences; no Correction/Wait; no numeric subtraction)",
          "controls": {{
            "pressing_intensity": 0.0-1.0,
            "risk_budget": 0.0-1.0,
            "line_height": 0.0-1.0,
            "rotation_aggressiveness": 0.0-1.0
          }},
          "tactical_hints": {{
            "through_ball_bias": 0.0-1.0,
            "long_ball_bias": 0.0-1.0,
            "build_up_short": 0.0-1.0,
            "high_press": 0.0-1.0,
            "low_block": 0.0-1.0
          }},
          "world_model_rationale": "brief evidence-based explanation, or why the quality gate is closed"
        }}
        tactical_preset and tactical_hints are optional; controls are required."""
        raw = self._call_llm(system, user, json_mode=True, temperature=0.55)
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "reasoning" in data:
                data["reasoning"] = _sanitize_coach_reasoning(data.get("reasoning", ""))
            return json.dumps(data, ensure_ascii=False)
        except Exception:
            return raw

    def predict_match_result(self, t1_name, t1_info, t2_name, t2_info, round_name):
        is_knockout = "Group" not in round_name
        stage_bias = "KNOCKOUT PRUDENCE (Avoid risks)" if is_knockout else "GROUP ADVENTURE (Aggressive if needed)"
        system = f"""You write a one-line atmospheric beat for the broadcast. 
        Stage mood: {stage_bias}.
        The REAL score, xG, and winner are computed elsewhere — you must NOT invent or imply them.
        Output MUST be JSON. key_event must not contain scorelines (e.g. 3-2), xG numbers, or tactical control decimals; avoid digit-heavy sentences when possible."""
        user = f"""Match backdrop: {t1_name} ({t1_info}) vs {t2_name} ({t2_info}). Stage: {round_name}
        Return JSON format:
        {{
          "key_event": "one short atmospheric event without invented match statistics",
          "signals": {{
            "coach_pressure": -1.0 to 1.0,
            "crowd_hostility": -1.0 to 1.0,
            "player_anxiety": -1.0 to 1.0,
            "media_amplification": 0.0 to 1.0,
            "unity_signal": -1.0 to 1.0
          }},
          "targets": ["optional exact team names: {t1_name}, {t2_name}"],
          "persistence": 0.0 to 1.0
        }}"""
        raw = self._call_llm(system, user, json_mode=True, temperature=0.55)
        audit = self.generation_pipeline.validate_atmosphere(json.loads(raw))
        self.generation_audits.append(audit)
        return json.dumps(audit.published, ensure_ascii=False)

    def generate_media_matrix(self, match_verdict, t1_name, t2_name, t1_exposure, t2_exposure, facts_ledger=None):
        system = """You are a Media Matrix Engine. Output MUST be JSON.
FACTS_LEDGER is the only source for scores, xG, tactical controls (pressing, line height, risk, rotation), drama_score, and tone_contract.
If match_verdict JSON conflicts with FACTS_LEDGER, ignore the conflicting fields in match_verdict.
You must not describe the same match as both (a) a high-scoring thriller AND (b) a goalless or no-scoring stalemate unless tone_contract explicitly allows nuance.
Banned in both mainstream and social_chaos_post: breaking the fourth wall — no mentions of simulation, engine, Poisson, RNG, glitch, bug, or phrases like 'something is wrong with the simulation', math errors, or mock-acronym hashtags about the model."""
        ledger = facts_ledger or {}
        ledger_txt = json.dumps(ledger, ensure_ascii=False, indent=2)
        user = f"""Match engine JSON (may contain atmospheric llm_key_event — do not treat its numbers as facts):
{match_verdict}

FACTS_LEDGER (authoritative — all numbers and tone must follow this):
{ledger_txt}

Exposure: {t1_name}({t1_exposure}) | {t2_name}({t2_exposure})

Return JSON format: {{"mainstream": "analysis", "social_chaos_post": "viral text", "metrics": {{"professional_score": -2 to 2, "social_chaos": -5 to 5}}}}
Rules:
- Any statistic you state must appear in FACTS_LEDGER with the same value (or clearly derived, e.g. goal difference from score).
- Follow tone_contract for overall vibe vs goal total (no contradictions).
- Social post: in-universe sports commentary only; no meta jokes about systems or randomness."""
        raw = self._call_llm(system, user, json_mode=True, temperature=0.45)
        try:
            data = json.loads(raw)
            audit = self.generation_pipeline.validate_media(data, ledger)
            self.generation_audits.append(audit)
            return json.dumps(audit.published, ensure_ascii=False)
        except Exception:
            return raw

    def perform_agent_reflection(self, team_name, momentum, hidden_state, patience, memory_context=None):
        system = "You are a Cognitive Reflection Engine. Output MUST be JSON."
        memory_blob = memory_context or "[]"
        user = f"""TEAM: {team_name} | MOMENTUM: {momentum} | STATE: {hidden_state} | Icon Patience: {patience}
        RECENT HIGH-SALIENCE MEMORY CONTEXT: {memory_blob}
        Reflect on the journey and propose bounded adjustments, not direct arbitrary rewrites.
        Return JSON object format:
        {{
          "reflection": "brief text",
          "confidence": 0.0-1.0,
          "evidence_memory_ids": ["id1", "id2"],
          "suggested_adjustments": {{
            "risk_budget": -0.10 to 0.10,
            "pressing_intensity": -0.10 to 0.10,
            "line_height": -0.10 to 0.10,
            "rotation_aggressiveness": -0.10 to 0.10,
            "icon_patience": -0.10 to 0.10,
            "w_h_delta": -0.10 to 0.10,
            "w_x_delta": -0.10 to 0.10
          }},
          "diary": "optional legacy text",
          "new_wh": "optional legacy float",
          "new_wx": "optional legacy float",
          "new_icon_patience": "optional legacy float"
        }}"""
        return self._call_llm(system, user, json_mode=True)

    def coach_in_match_plan(self, team_name: str, facts_ledger: dict, kind: str, context: str = ""):
        from src.match_engine.world_model.llm_decision_brief import (
            compact_world_model_facts_for_llm,
        )

        llm_facts_ledger = compact_world_model_facts_for_llm(facts_ledger)
        system = f"""You are the head coach of {team_name} during a live match.
FACTS_LEDGER is authoritative. Do NOT invent scores or xG.
If world_model_decision_support is present, treat it as uncertain model evidence:
- deliberation_agenda is model-owned attention guidance. Concentrate on at most
  its three recommended_focus tasks and omit unsupported optional contracts.
  If selecting fewer tasks, use recommended_portfolios_by_size for that exact
  task count; its bounded objective combines evidence priority, validated
  compute value, real rollout cost, and reasoning-domain diversity;
  compact evidence is exact, while omitted member/scenario arrays remain in the
  engine-owned full packet used to recompute every audit;
- when shadow_deliberation_phase=true, the current action and tactical controls
  were frozen before this call. Follow shadow_task_encouragement.assigned_focus
  for optional reasoning only. Any action/control mutation will be ignored and
  audited; randomized assignment is shadow-task ITT evidence, not match causality;
- task compute_cost_class distinguishes audits over existing evidence from new
  trajectory rollouts. After you select tasks, the world model—not you—allocates
  up to six compute credits; every selected task gets one base audit credit, and
  only real trajectory work can receive more under diminishing returns. Credits
  become hard member/path caps. A randomized last-credit experiment learns only
  whether deeper shadow compute yields a useful model-internal artifact; it is
  not evidence of match-outcome causality. Never invent or request a budget;
- world_model_deliberation_focus must name one to three eligible agenda tasks.
  Emit structured optional contracts only for those named tasks. The engine
  audits unsupported selections, missing selected contracts, extra unfocused
  contracts, downstream contract rejection, raw priority efficiency, and joint
  portfolio efficiency;
- compare candidates by risk_adjusted_value and effective_confidence;
- respect quality_gate_closed or available=false;
- use online_calibration trust factors only after their minimum sample count;
- online calibration can reduce trust but never reopen a closed quality gate;
- active_learning is optional: choose decision_mode=explore only when eligible,
  only select its exploration_action, and weigh information value against regret;
- epistemic uncertainty is reducible model ignorance and may motivate learning;
  aleatoric uncertainty is irreducible match randomness and must only price risk;
- transition_epistemic uncertainty is trajectory disagreement between trained
  dynamics members; use it as evidence, never as a guaranteed model failure;
- opponent_belief is a latent tactical posterior, not a discovered fact;
- if proposing opponent_hypothesis, select only a listed hypothesis and cite
  evidence_features present in its numeric feature contract; the engine will
  reject or cap claims that conflict with world-model likelihood;
- opponent_hypothesis_values are counterfactual model estimates. Prefer actions
  robust across plausible hypotheses when posterior entropy is high;
- opponent_information_question_menu is generated by the world model before
  optional reasoning. After the live action is frozen, prefer one proposal_id
  matching that action and copy its action, feature, horizon, purpose and both
  branch actions exactly. The engine rejects altered proposal fields. Legacy
  free-form queries remain accepted for compatibility but do not count as a
  completed world-model-question loop;
- opponent_information_cognitive_policy turns querying into a bounded episode.
  For the frozen action, follow its recommended ask/stop decision and exact
  proposal_id through opponent_information_policy. The world model owns the
  two-question budget, marginal information value, redundancy penalty, and stop
  threshold. When this policy is available, omit the legacy
  opponent_information_query contract. A stop is an explicit audited decision,
  not a missing answer;
- belief_space_meta_plan unifies information use with future model computation.
  Select one exact option_id through world_model_belief_space_meta_plan. The
  choice affects only the next decision's bounded shadow rollout allocation:
  answer-conditioned routing, broad coverage, or minimum safe coverage. It can
  never change the frozen action, tactics, total cap, or schedule a future action;
- active_probe_design turns an already-frozen safe exploration choice into a
  pre-registered predictive experiment. Select one exact probe_id only when
  the frozen action and decision_mode=explore match it. The probe compares a
  world-model retention forecast with the exploit-action forecast as a null;
  it cannot change the action or tactics, and one outcome is evidence rather
  than a causal conclusion;
- active_probe_discovery_memory is chronological, match-held-out evidence from
  prior realized probes. Only an active same-checkpoint/environment profile may
  apply a small displayed calibration to the probe forecast; compare raw and
  calibrated probabilities, prefer informative under-validated probes, and
  treat quarantined drift as zero authority. This memory never rewrites the
  neural forecast;
- active_probe_portfolio_design combines one or two nonredundant horizons for
  the same already-frozen exploration action. Select one exact portfolio_id;
  the engine owns its observation budget, redundancy penalty, and sub-probes.
  Multiple horizons add observations only, never another action or independent
  causal samples. Omit world_model_active_probe when this portfolio task is
  selected;
- opponent_information_query selects one observable tactical control to monitor.
  The engine owns the 0.5 threshold, observation noise, posterior branches,
  information gain, adaptive value and ranking. A later observed answer produces
  a model-owned Bayesian answer contract for feedback and calibration. It never
  observes hidden intent, double-counts the control in the live belief, changes
  the current action, or schedules either shadow branch action;
- opponent_information_feedback contains only previously resolved, digest-checked
  queries from the same team, checkpoint, and policy environment. Use signed
  probability surprise, proper scores, query rank, and realized-branch regret
  to improve the next question and its conditional reasoning. A listed branch
  action was never executed merely because it was proposed, so do not describe
  it as an observed counterfactual outcome or as evidence of hidden intent;
- when compatible feedback is available, opponent_information_adaptation may
  cite exactly one prior decision and declare a feature change, horizon change,
  contingent-policy repair, or retention of an already clean query. The engine
  checks that the cited feedback actually supports that adaptation and later
  scores whether the next realized query improved the relevant proper score,
  query efficiency, or branch regret. Do not claim improvement before scoring;
- second_order_game is a two-ply belief-space policy proxy. When its
  trajectory_rollout gate is active, continuation values blend a budgeted
  predicted-state search with the current-state proxy; otherwise they reuse
  that proxy. Check validation authority, uncertainty, and compute audit;
- opponent response transitions are held-out-validated observational patterns,
  never causal facts. An opponent_response_hypothesis may stress one evaluated
  action branch, but its influence is capped and cannot update response memory;
- world_model_critique is a falsifiable forecast-residual claim for your selected
  action and one evaluated horizon. State overestimate, underestimate, or
  neutral and cite only listed semantic evidence. It runs in shadow until
  cross-match held-out error proves useful; it cannot rewrite model forecasts.
  A validated overestimate may reduce bridge strength, but no critique can
  increase execution authority;
- state_scales are neural trajectory projections at short, tactical, and
  strategic levels. A world_model_event_hypothesis must select one event already
  listed by one evaluated horizon. It is scored against the same later simulator
  outcome as the neural probability and remains shadow-only; it cannot alter the
  action, model state, or critic memory;
- trajectory_modes transparently partition ensemble members by their predicted
  downside signatures. They expose alternative futures, not extra ground truth.
  You may declare one world_model_risk_constraint for the selected action and an
  evaluated horizon. The engine checks its member frequency with a conservative
  uncertainty bound and later scores it on the same-horizon simulator outcome.
  risk_scope=terminal checks only the horizon endpoint; within_horizon
  checks member-consistent intermediate states and requires a validated two-step
  rollout plus sufficiently dense live interval monitoring;
  This certificate is shadow-only: it cannot veto, authorize, or change an action;
- distributional_action_frontiers state their distribution_scope explicitly.
  epistemic_member_only means transition-member disagreement only; it must not
  be described as full outcome randomness. calibrated_predictive means the
  member axis has been crossed with signed residual quantiles learned only from
  a held-out calibration split. uncertainty_decomposition reports the two axes
  separately; it does not claim causal or statistical independence. Frontiers
  compare mean, lower-tail CVaR and upside probability. These are distinct objectives,
  not interchangeable confidence scores. A world_model_distributional_claim may
  explain the selected action against one alternative using exactly one listed
  criterion and the exact distribution_scope. The engine checks that relation.
  Only a frozen calibrated_predictive scenario lattice is eligible for strict
  CRPS, pinball-loss and interval-coverage readiness. It remains
  shadow-only and cannot change the action or ranking;
- temporal_utility_paths connect ordered horizon marginals by the same dynamics
  member identity. Check temporal_coupling_source: compatible historical paths
  may supply held-out empirical residual-rank templates; otherwise the engine
  explicitly reports a comonotonic residual-rank fallback. Also inspect the
  coupling validation status: insufficient_evidence is exploratory, validated
  has beaten the same-marginal fallback gates, and degraded coupling is disabled.
  Drawdown, reversal and downside values are coupling-scenario rates, never
  calibrated joint temporal probabilities. Never add horizon utilities as
  independent rewards because each is a same-origin cumulative forecast;
- world_model_risk_preference is one explicit preference applied unchanged to
  every declared horizon and all four actions. Horizon weights must sum to 1.
  The fixed reference utility is zero; loss_aversion and
  diminishing_sensitivity define a bounded prospect-value transform. The engine
  recomputes every scenario value, per-horizon preference regret and aggregate
  regret. It also owns a fixed preference-neighborhood grid and jointly removes
  one aligned dynamics member or residual-quantile scenario at a time. You
  cannot choose these stress ranges; nominal consistency is not robustness.
  The engine also compares all actions inside each coupled temporal scenario
  before aggregating pathwise regret, so horizon weights cannot hide reversals.
  Do not report your own utility score. This audit is shadow-only and
  does not provide counterfactual realized outcomes for actions not taken;
- semantic_event_fusion distinguishes transparent member-frequency projection
  from a learned event head. Treat the learned probability as evidence only when
  that event's exact rollout-depth gate is active; longer unvalidated rollouts
  intentionally remain projection-only;
- world_model_event_option is one shadow-only conditional continuation policy.
  Its first action must equal your selected action; choose different continuation
  actions for event occurrence and absence. The engine re-evaluates only those
  branches on member-predicted states under a hard compute budget. It cannot
  schedule or execute either future action. If the naturally observed next
  action matches the resolved branch, its frozen value prediction is scored on
  a later simulator outcome; mismatched actions never become counterfactual
  labels;
- world_model_contrastive_claim is one falsifiable explanation of why a listed
  context factor supports or opposes your selected action relative to one
  evaluated alternative. The engine neutralizes only that declared context and
  recomputes both margins. This checks faithfulness to the world model, not
  real-world causality, and cannot alter your action;
- change_point is computed from live numeric evidence. You may explain a
  watch/confirmed change with opponent_change_claim, but your claim cannot
  trigger, confirm, or cancel the detector. Cite only features whose signed
  changes support the claimed from/to presets;
- never describe exploration as guaranteed learning or override its safety budget;
- you may disagree, but explain why without inventing outcomes.
Output JSON only. Adjust tactics with small bounded deltas."""
        user = f"""Trigger: {kind}
FACTS_LEDGER: {json.dumps(llm_facts_ledger, ensure_ascii=False)}
Context: {context}
Return JSON:
{{
  "reasoning": "max 2 sentences, qualitative only",
  "confidence": 0.0-1.0,
  "controls_delta": {{
    "pressing_intensity": -0.15 to 0.15,
    "risk_budget": -0.15 to 0.15,
    "line_height": -0.15 to 0.15,
    "rotation_aggressiveness": -0.15 to 0.15
  }},
  "tactical_hints": {{"through_ball_bias": 0-1, "high_press": 0-1, "low_block": 0-1}},
  "tactical_preset": "optional: balanced|gegenpress|possession|counter|low_block|wing_play|direct",
  "sub_intent": "optional short note",
  "world_model_action": "hold|pass|cross|shot|none",
  "world_model_decision_mode": "exploit|explore|decline",
  "world_model_rationale": "brief explanation tied to uncertainty and candidate evidence",
  "world_model_deliberation_focus": {{
    "tasks": ["one to three eligible task names from deliberation_agenda.tasks"],
    "confidence": 0.5-1.0,
    "rationale": "why these tasks deserve the limited reasoning budget"
  }},
  "world_model_critique": {{
    "action": "must equal world_model_action",
    "horizon": "one key from that candidate's multi_horizon_predictions",
    "direction": "underestimate|overestimate|neutral",
    "confidence": 0.0-1.0,
    "evidence_features": ["zone|score_state|match_phase|opponent_belief|opponent_response|trajectory_rollout|epistemic_uncertainty|aleatoric_uncertainty|tactical_shape"],
    "rationale": "one falsifiable semantic residual explanation"
  }},
  "world_model_event_hypothesis": {{
    "action": "must equal world_model_action",
    "horizon": "one key from that candidate's multi_horizon_predictions",
    "event": "retain_possession|enter_final_third|positive_territorial_shift|improve_scoreline",
    "expectation": "occur|not_occur",
    "confidence": 0.5-1.0,
    "evidence_scales": ["short|tactical|strategic"],
    "rationale": "one falsifiable event forecast grounded in state_scales"
  }},
  "world_model_event_option": {{
    "first_action": "must equal world_model_action",
    "horizon": "transition or an evaluated horizon with rollout depth <= 2",
    "event": "retain_possession|enter_final_third|positive_territorial_shift|improve_scoreline",
    "on_occurrence": "hold|pass|cross|shot",
    "on_absence": "a different hold|pass|cross|shot action",
    "confidence": 0.5-1.0,
    "rationale": "one short conditional policy explanation"
  }},
  "world_model_contrastive_claim": {{
    "selected_action": "must equal world_model_action",
    "alternative_action": "a different evaluated hold|pass|cross|shot action",
    "horizon": "transition or an evaluated horizon with rollout depth <= 2",
    "factor": "score_context|match_phase|own_tactics|opponent_tactics|crowd_context",
    "effect": "supports_selected|opposes_selected",
    "confidence": 0.5-1.0,
    "rationale": "why this factor changes the selected-vs-alternative margin"
  }},
  "world_model_risk_constraint": {{
    "selected_action": "must equal world_model_action",
    "horizon": "transition or an evaluated horizon with rollout depth <= 2",
    "downside_event": "lose_possession|negative_territorial_shift|worsen_scoreline|fail_enter_final_third",
    "risk_scope": "terminal|within_horizon",
    "max_violation_probability": 0.05-0.95,
    "confidence": 0.5-1.0,
    "rationale": "one falsifiable chance constraint grounded in trajectory_modes"
  }},
  "world_model_distributional_claim": {{
    "selected_action": "must equal world_model_action",
    "alternative_action": "a different evaluated hold|pass|cross|shot action",
    "horizon": "one key from both candidates' multi_horizon_predictions",
    "criterion": "mean_utility|lower_tail_cvar_25|upside_probability",
    "distribution_scope": "epistemic_member_only|calibrated_predictive",
    "relation": "selected_better|alternative_better|approximately_equal",
    "confidence": 0.5-1.0,
    "rationale": "one falsifiable distributional reason for the action"
  }},
  "world_model_risk_preference": {{
    "selected_action": "must equal world_model_action",
    "distribution_scope": "epistemic_member_only|calibrated_predictive",
    "horizon_weights": {{"transition": 0.4, "60s": 0.6}},
    "loss_aversion": 1.0-4.0,
    "diminishing_sensitivity": 0.5-1.0,
    "max_acceptable_regret": 0.0-0.5,
    "confidence": 0.5-1.0,
    "rationale": "why this same risk preference should hold across at least two horizons"
  }},
  "opponent_hypothesis": {{
    "tactical_preset": "one preset listed in opponent_belief.hypotheses",
    "confidence": 0.0-1.0,
    "evidence_features": ["pressing_intensity|risk_budget|line_height|rotation_aggressiveness"],
    "rationale": "one short observation-grounded explanation"
  }},
  "opponent_information_query": {{
    "proposal_id": "one exact id from opponent_information_question_menu.proposals (preferred)",
    "selected_action": "must equal world_model_action",
    "feature": "pressing_intensity|risk_budget|line_height|rotation_aggressiveness",
    "horizon": "an evaluated non-transition horizon from 10s to 300s",
    "purpose": "reduce_opponent_uncertainty|resolve_action_choice",
    "action_if_high": "hold|pass|cross|shot (shadow continuation only)",
    "action_if_low": "hold|pass|cross|shot (shadow continuation only)",
    "confidence": 0.5-1.0,
    "rationale": "why observing this control would clarify the decision"
  }},
  "opponent_information_policy": {{
    "decision": "ask|stop",
    "proposal_id": "required for ask and empty for stop; follow opponent_information_cognitive_policy for the frozen action",
    "confidence": 0.5-1.0,
    "rationale": "why marginal information value justifies asking or stopping"
  }},
  "world_model_belief_space_meta_plan": {{
    "option_id": "one exact id from belief_space_meta_plan.options",
    "confidence": 0.5-1.0,
    "rationale": "why the next decision should focus, broaden, or stop extra shadow compute"
  }},
  "world_model_active_probe": {{
    "probe_id": "one exact id from active_probe_design.options",
    "confidence": 0.5-1.0,
    "rationale": "why this endpoint best discriminates the frozen safe exploration from its exploit null"
  }},
  "world_model_active_probe_portfolio": {{
    "portfolio_id": "one exact id from active_probe_portfolio_design.options",
    "confidence": 0.5-1.0,
    "rationale": "why the redundancy-adjusted observation bundle is worth its bounded budget"
  }},
  "world_model_active_probe_sequential_policy": {{
    "policy_option_id": "one exact id from active_probe_sequential_policy_design.options",
    "confidence": 0.5-1.0,
    "rationale": "explain the exact engine-authorized continue portfolio or stop decision; never move a boundary or choose a stopped horizon"
  }},
  "world_model_predictive_mechanism": {{
    "hypothesis_id": "one exact id from predictive_mechanism_design.options for the frozen action",
    "confidence": 0.5-1.0,
    "mechanism_statement": "plain-language articulation of the exact driver, consequence, direction, and horizon; association only",
    "rationale": "why this member-aligned joint forecast is more informative than its independence null"
  }},
  "opponent_information_adaptation": {{
    "prior_decision_id": "one decision_id from opponent_information_feedback.recent_resolutions",
    "adaptation_kind": "change_query_feature|change_query_horizon|repair_contingent_policy|retain_validated_query",
    "confidence": 0.5-1.0,
    "rationale": "how the new query or branch policy responds to the cited scored feedback"
  }},
  "opponent_response_hypothesis": {{
    "if_action": "hold|pass|cross|shot",
    "response_preset": "one preset listed in opponent_belief.hypotheses",
    "confidence": 0.0-1.0,
    "rationale": "one short conditional scenario explanation"
  }},
  "opponent_change_claim": {{
    "from_preset": "listed prior preset",
    "to_preset": "listed candidate preset",
    "confidence": 0.0-1.0,
    "evidence_features": ["only features with observed signed changes"],
    "rationale": "a falsifiable explanation, only when change_point is watch or confirmed"
  }}
}}"""
        return self._call_llm(system, user, json_mode=True, temperature=0.5)

    def coach_world_model_deliberation(
        self,
        team_name: str,
        facts_ledger: dict,
        kind: str,
        frozen_plan: dict,
    ):
        """Run optional world-model reasoning after action/control freeze."""
        frozen = {
            "world_model_action": str(frozen_plan.get(
                "world_model_action", "none"
            )),
            "world_model_decision_mode": str(frozen_plan.get(
                "world_model_decision_mode", "exploit"
            )),
            "controls_delta": dict(frozen_plan.get("controls_delta") or {}),
            "tactical_hints": dict(frozen_plan.get("tactical_hints") or {}),
            "tactical_preset": frozen_plan.get("tactical_preset"),
        }
        context = (
            "POST_ACTION_SHADOW_DELIBERATION. The action, decision mode, tactical "
            "controls, hints, and preset below are immutable. Return them "
            "unchanged if present, and spend attention only on the assigned "
            "shadow deliberation portfolio. Any attempted mutation is ignored. "
            f"FROZEN_PLAN={json.dumps(frozen, ensure_ascii=False)}"
        )
        return self.coach_in_match_plan(
            team_name, facts_ledger, kind, context=context,
        )

    def revise_world_model_contrastive_claim(
        self,
        *,
        team_name: str,
        immutable_selected_action: str,
        counterevidence: dict,
        revision_contract: dict,
    ) -> dict:
        """Use one bounded model-feedback turn to repair explanation only."""
        system = f"""You are revising one explanation for coach {team_name}.
The selected action {immutable_selected_action!r} is immutable.
You may revise only the structured contrastive claim. You cannot change the
action, tactics, controls, confidence of the coach plan, or world-model output.
The prior claim failed a world-model directional-faithfulness probe. Use only
the supplied numeric counterevidence and revision contract. A revised claim is
still a model-faithfulness statement, never a real-world causal fact.
Return one JSON object containing exactly the contrastive-claim fields."""
        user = f"""WORLD_MODEL_COUNTEREVIDENCE:
{json.dumps(counterevidence, ensure_ascii=False)}
REVISION_CONTRACT:
{json.dumps(revision_contract, ensure_ascii=False)}
Return JSON:
{{
  "selected_action": "exactly {immutable_selected_action}",
  "alternative_action": "one different evaluated action",
  "horizon": "one jointly validated one- or two-step horizon",
  "factor": "one allowed schema-grounded factor",
  "effect": "supports_selected|opposes_selected",
  "confidence": 0.5-1.0,
  "rationale": "one concise explanation corrected by the counterevidence"
}}"""
        raw = self._call_llm(
            system, user, json_mode=True, temperature=0.2,
        )
        return raw if isinstance(raw, dict) else self._extract_json_object(raw)

    def player_in_match_reflection(self, player_name: str, team_name: str, facts_ledger: dict, kind: str):
        system = f"""You are footballer {player_name} ({team_name}). In-match mental reflection only.
Do NOT state match scores as predictions. JSON only."""
        user = f"""Event: {kind}
FACTS_LEDGER: {json.dumps(facts_ledger, ensure_ascii=False)}
Return JSON:
{{
  "narrative": "one short inner thought",
  "confidence": 0.0-1.0,
  "emotion_bias": {{"pride": -0.2 to 0.2, "anger": -0.2 to 0.2, "fear": -0.2 to 0.2, "determination": -0.2 to 0.2}},
  "risk_delta": -0.12 to 0.12
}}"""
        return self._call_llm(system, user, json_mode=True, temperature=0.55)

    def referee_in_match_judgment(self, facts_ledger: dict, kind: str):
        system = """You are the center referee. FACTS_LEDGER is authoritative.
Do not assign goals/cards in output — only tendency adjustments. JSON only."""
        user = f"""Situation: {kind}
FACTS_LEDGER: {json.dumps(facts_ledger, ensure_ascii=False)}
Return JSON:
{{
  "reasoning": "max 2 sentences",
  "strictness_delta": -0.12 to 0.12,
  "bias_delta": -0.08 to 0.08,
  "var_recommendation": "none|review|var",
  "calm_delta": -0.15 to 0.15
}}"""
        return self._call_llm(system, user, json_mode=True, temperature=0.45)

    def assistant_signal(self, side: str, facts_ledger: dict, kind: str):
        system = f"""You are assistant referee ({side} line). Signal to center ref only. JSON only."""
        user = f"""Event: {kind}
FACTS_LEDGER: {json.dumps(facts_ledger, ensure_ascii=False)}
Return JSON:
{{
  "signal": "short phrase",
  "offside_strictness_delta": -0.1 to 0.1,
  "recommend_card_review": true|false
}}"""
        return self._call_llm(system, user, json_mode=True, temperature=0.4)

    def crowd_collective_reaction(self, facts_ledger: dict, kind: str):
        system = """You represent the stadium crowd as one collective voice. JSON only. No invented scores."""
        user = f"""Moment: {kind}
FACTS_LEDGER: {json.dumps(facts_ledger, ensure_ascii=False)}
Return JSON:
{{
  "chant_narrative": "short crowd reaction",
  "psi_pulse": -0.35 to 0.35,
  "home_boost_delta": -0.15 to 0.15
}}"""
        return self._call_llm(system, user, json_mode=True, temperature=0.6)

    def generate_locker_room_leak(self, team):
        return self._call_llm("Whistleblower", f"Scandalous internal leak about {team}")

    def generate_free_inference(self, user_scenario, context_summary):
        system = "You are a football systems analyst. Provide concise causal reasoning."
        user = f"""Known Context:
{context_summary}

User Counterfactual:
{user_scenario}

Explain likely social and tactical consequences in 4-8 bullet points."""
        return self._call_llm(system, user, json_mode=False)


class NullSimulationLLM:
    """Deterministic no-op LLM for calibration / narrative-off tournament runs."""

    def coach_decide_tactics(
        self,
        team_name,
        my_info,
        opp_name,
        opp_info,
        world_model_decision_support=None,
    ):
        return json.dumps(
            {
                "formation": "4-3-3",
                "style": "balanced",
                "tactical_preset": "balanced",
                "reasoning": "Calibration mode — default balanced shape.",
                "controls": {
                    "pressing_intensity": 0.5,
                    "risk_budget": 0.5,
                    "line_height": 0.5,
                    "rotation_aggressiveness": 0.5,
                },
            },
            ensure_ascii=False,
        )

    def predict_match_result(self, t1_name, t1_info, t2_name, t2_info, round_name):
        return json.dumps({"key_event": "Calibration mode — narrative layer skipped."})

    def generate_media_matrix(self, match_verdict, t1_name, t2_name, t1_exposure, t2_exposure, facts_ledger=None):
        return json.dumps(
            {
                "mainstream": "Calibration run — media matrix skipped.",
                "social_chaos_post": "",
                "metrics": {"professional_score": 0, "social_chaos": 0},
            },
            ensure_ascii=False,
        )

    def perform_agent_reflection(self, team_name, momentum, hidden_state, patience, memory_context=None):
        return json.dumps(
            {
                "reflection": "Calibration mode — reflection skipped.",
                "confidence": 0.5,
                "evidence_memory_ids": [],
                "suggested_adjustments": {},
            },
            ensure_ascii=False,
        )
