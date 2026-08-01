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
        system = f"""You are the head coach of {team_name} during a live match.
FACTS_LEDGER is authoritative. Do NOT invent scores or xG.
If world_model_decision_support is present, treat it as uncertain model evidence:
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
- never describe exploration as guaranteed learning or override its safety budget;
- you may disagree, but explain why without inventing outcomes.
Output JSON only. Adjust tactics with small bounded deltas."""
        user = f"""Trigger: {kind}
FACTS_LEDGER: {json.dumps(facts_ledger, ensure_ascii=False)}
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
  "world_model_rationale": "brief explanation tied to uncertainty and candidate evidence"
}}"""
        return self._call_llm(system, user, json_mode=True, temperature=0.5)

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
