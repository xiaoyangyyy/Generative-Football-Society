import re
import numpy as np


class SocialDialogueEngine:
    """
    Modular social dialogue layer:
    1) speech-act based dialogue generation
    2) meme market topic competition
    3) text -> structured signal extraction
    4) social memory compression hooks via agent APIs
    """

    def __init__(self, feed, turns_per_match=4, *, rng=None):
        self.feed = feed
        self.turns_per_match = max(2, int(turns_per_match))
        self.rng = rng or np.random.default_rng()
        self.market_step = 0
        self.topic_market = {}
        self.speech_acts = [
            "defend",
            "attack",
            "unify",
            "shift_blame",
            "calm",
            "provoke",
        ]

    def _clip01(self, x):
        return float(np.clip(float(x), 0.0, 1.0))

    def _softmax_choice(self, keys, logits, rng):
        arr = np.array([float(logits[k]) for k in keys], dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite logits in social dialogue softmax: {arr}")
        arr = arr - np.max(arr)
        probs = np.exp(arr)
        total = float(probs.sum())
        if (not np.isfinite(total)) or total <= 0.0:
            raise ValueError(f"Invalid probability mass in social dialogue softmax: sum={total}, logits={arr}")
        probs = probs / total
        if not np.all(np.isfinite(probs)):
            raise ValueError(f"Non-finite probabilities in social dialogue softmax: {probs}")
        idx = int(rng.choice(np.arange(len(keys)), p=probs))
        return keys[idx]

    def _seed_topics(self, t1_name, t2_name, stage_name, context, rng):
        winner = context.get("winner") or "draw"
        score = context.get("score", "0-0")
        key_event = str(context.get("key_event", "")).lower()
        drama = float(context.get("drama_score", 0.3))
        strictness = float(context.get("ref_strictness", 0.5))

        topics = [
            f"{(stage_name or '').lower()}_narrative",
            "tactical_identity",
            "discipline_and_refereeing",
            "winner_legitimacy",
            "momentum_shift",
        ]
        if winner != "draw":
            topics.append(f"{winner.lower()}_dominance")
        if strictness > 0.60:
            topics.append("referee_fairness")
        sn = (stage_name or "").strip().lower()
        if sn == "final":
            topics.extend(
                [
                    "cinderella_run",
                    "penalty_resilience",
                    "favorite_underperformance_pressure",
                    "brazil_underperformance",
                ]
            )
        if "pens" in key_event or score.startswith("0-0"):
            topics.append("mental_strength")
        if drama > 0.65:
            topics.append("high_drama")

        for topic in topics:
            if topic not in self.topic_market:
                self.topic_market[topic] = {
                    "heat": 0.18 + 0.25 * rng.random(),
                    "novelty": 0.60 + 0.25 * rng.random(),
                    "controversy": 0.30 + 0.30 * rng.random(),
                    "last_step": self.market_step,
                }

        return topics

    def _decay_market(self):
        for topic, rec in self.topic_market.items():
            age = max(0, self.market_step - int(rec.get("last_step", self.market_step)))
            rec["heat"] = float(np.clip(rec["heat"] * np.exp(-0.08 * age), 0.01, 1.85))
            rec["novelty"] = float(np.clip(rec["novelty"] * np.exp(-0.10 * age), 0.01, 1.25))
            rec["controversy"] = float(np.clip(rec["controversy"] * np.exp(-0.045 * age), 0.01, 1.2))

    def _stage_topic_bias(self, topic, stage_name):
        """Prefer the current stage narrative; damp carry-over from earlier rounds."""
        stage_name = stage_name or ""
        sn = stage_name.lower().strip()
        t = topic.lower()
        bias = 0.0
        if t == f"{sn}_narrative":
            bias += 0.88 + (0.30 if sn == "final" else 0.0)
        final_stage_topics = frozenset(
            {
                "cinderella_run",
                "penalty_resilience",
                "favorite_underperformance_pressure",
                "brazil_underperformance",
                "winner_legitimacy",
            }
        )
        if sn == "final":
            if t in final_stage_topics:
                bias += 0.68
            if t == "discipline_and_refereeing":
                bias -= 0.52
        is_group = sn.startswith("group")
        if is_group and ("semi-finals" in t or ("final" in t and "semi" not in t)):
            bias -= 0.55
        if (not is_group) and t.startswith("group ") and "narrative" in t:
            bias -= 0.45
        if "semi-finals" in t and ("final" in sn and "semi" not in sn):
            bias -= 0.55
        if "quarter-finals" in t and sn in ("semi-finals", "final"):
            bias -= 0.35
        return bias

    def _pick_topic(self, agent, candidate_topics, rng, stage_name=""):
        if not candidate_topics:
            candidate_topics = list(self.topic_market.keys()) or ["match_discourse"]
        logits = {}
        stage_l = (stage_name or "").strip().lower()
        for topic in candidate_topics:
            rec = self.topic_market.get(topic, {"heat": 0.2, "novelty": 0.5, "controversy": 0.3})
            referee_bias = 0.35 if ("referee" in topic and agent.referee_grievance > 0.25) else 0.0
            if stage_l == "final" and (
                "referee" in topic.lower() or "discipline_and_referee" in topic.lower()
            ):
                referee_bias *= 0.30
            conflict_bias = 0.25 * np.tanh(agent.conflict_heat)
            stage_bias = self._stage_topic_bias(topic, stage_name)
            logits[topic] = (
                1.05 * rec["heat"]
                + 0.48 * rec["novelty"]
                + 0.62 * rec["controversy"]
                + referee_bias
                + conflict_bias
                + stage_bias
                + rng.normal(0.0, 0.05)
            )
        return self._softmax_choice(list(logits.keys()), logits, rng)

    def _pick_speech_act(self, speaker, opponent, match_context, rng):
        score_diff = float(match_context.get("score_diff_for_speaker", 0.0))
        stage_pressure = float(np.clip(match_context.get("stage_pressure", 0.3), 0.0, 1.0))
        grievance = float(np.clip(speaker.referee_grievance, 0.0, 0.95))
        arrogance = float(np.clip(speaker.personality.get("arrogance", 0.5), 0.0, 1.0))
        cohesion = float(np.clip(speaker.team_cohesion, 0.0, 1.0))
        conflict = float(np.clip(speaker.conflict_heat, 0.0, 1.05))

        logits = {
            "defend": 0.55 + 0.90 * max(0.0, -score_diff) + 0.45 * grievance,
            "attack": 0.50 + 0.85 * max(0.0, score_diff) + 0.50 * arrogance + 0.20 * conflict,
            "unify": 0.45 + 0.80 * max(0.0, -score_diff) + 0.35 * (1.0 - cohesion),
            "shift_blame": 0.30 + 0.70 * grievance + 0.55 * conflict + 0.25 * stage_pressure,
            "calm": 0.35 + 0.55 * stage_pressure + 0.40 * max(0.0, speaker.hidden_state[2]),
            "provoke": 0.25 + 0.65 * arrogance + 0.50 * max(0.0, score_diff),
        }
        return self._softmax_choice(self.speech_acts, logits, rng)

    def _compose_utterance(
        self, speaker_name, target_name, topic, act, context, rng,
    ):
        score = context.get("score", "0-0")
        stage_name = context.get("stage_name", "Stage")
        templates = {
            "defend": [
                f"{stage_name}: we respect the {score} result but the narrative on {topic} is still ours to define.",
                f"We absorb the noise and stay aligned. {topic} will look different next match.",
                f"Our setup on {topic} held under pressure — finer margins decided {score}.",
                f"We stay disciplined on {topic}; the table talk does not change our structure.",
                f"Narrow margins in {stage_name}; we keep our line on {topic} and move forward.",
            ],
            "attack": [
                f"We imposed our rhythm. Anyone questioning this result ignores the details around {topic}.",
                f"The scoreboard is fair. Our preparation exposed their limits on {topic}.",
            ],
            "unify": [
                f"No fractures inside this squad. We move as one and reset the narrative on {topic}.",
                f"This group responds together. We protect each other and improve on {topic}.",
            ],
            "shift_blame": [
                f"Context matters: officiating and game breaks changed the flow around {topic}.",
                f"Hard to evaluate the match without acknowledging external distortion on {topic}.",
            ],
            "calm": [
                f"Noise is temporary; process is permanent. We keep calm and keep working on {topic}.",
                f"No panic cycle. We trust the long arc and stabilize the conversation around {topic}.",
            ],
            "provoke": [
                f"@{target_name} talk less, solve {topic} on the pitch next time.",
                f"Pressure reveals identity. @{target_name}, your story on {topic} is collapsing.",
            ],
        }
        bucket = templates.get(act, [f"We respond on {topic}."])
        text = str(rng.choice(bucket))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_signal(self, text, act, speaker, target, topic):
        lower = text.lower()
        exclam = min(3, text.count("!"))
        caps_ratio = 0.0
        if text:
            caps_ratio = sum(1 for c in text if c.isupper()) / max(1, sum(1 for c in text if c.isalpha()))

        act_sentiment = {
            "defend": -0.08,
            "attack": 0.16,
            "unify": 0.22,
            "shift_blame": -0.18,
            "calm": 0.12,
            "provoke": -0.24,
        }.get(act, 0.0)
        unity_signal = {
            "defend": 0.14,
            "attack": 0.02,
            "unify": 0.32,
            "shift_blame": -0.20,
            "calm": 0.24,
            "provoke": -0.28,
        }.get(act, 0.0)
        provocation = {
            "defend": 0.28,
            "attack": 0.52,
            "unify": 0.10,
            "shift_blame": 0.48,
            "calm": 0.08,
            "provoke": 0.72,
        }.get(act, 0.25)

        provocation = float(np.clip(provocation + 0.06 * exclam + 0.12 * caps_ratio, 0.0, 1.0))
        credibility = float(np.clip(
            0.35 + 0.55 * speaker.personality.get("discipline", 0.6) - 0.30 * provocation + 0.10 * (act == "calm"),
            0.05,
            1.0,
        ))
        sentiment = float(np.clip(
            act_sentiment + 0.12 * (credibility - 0.5) - 0.08 * max(0.0, speaker.referee_grievance - 0.2),
            -1.0,
            1.0,
        ))

        blame_target = "none"
        if act == "shift_blame":
            if "ref" in lower or "officiat" in lower or "whistle" in lower:
                blame_target = "referee"
            else:
                blame_target = "external"
        elif act in {"attack", "provoke"}:
            blame_target = target.team_name

        return {
            "sentiment": sentiment,
            "unity_signal": float(np.clip(unity_signal, -1.0, 1.0)),
            "provocation_level": provocation,
            "credibility": credibility,
            "blame_target": blame_target,
            "topic": topic,
            "speech_act": act,
        }

    def _update_topic_market(self, topic, signal, rng, stage_name=""):
        rec = self.topic_market.setdefault(
            topic,
            {"heat": 0.2, "novelty": 0.5, "controversy": 0.3, "last_step": self.market_step},
        )
        sentiment_abs = abs(float(signal.get("sentiment", 0.0)))
        prov = float(signal.get("provocation_level", 0.2))
        cred = float(signal.get("credibility", 0.5))

        stage = (stage_name or "").strip().lower()
        topic_l = str(topic).lower()
        stage_local_boost = 0.0
        if stage == "final":
            if topic_l in {"final_narrative", "cinderella_run", "penalty_resilience", "winner_legitimacy"}:
                stage_local_boost = 0.22
            elif topic_l == "tactical_identity":
                stage_local_boost = -0.06

        # Calibrated coupled dynamics:
        # - slower heat accumulation
        # - rising controversy drags novelty down
        # - high current heat naturally dampens further growth (without hard clipping logic)
        h0 = float(rec.get("heat", 0.2))
        c0 = float(rec.get("controversy", 0.3))
        n0 = float(rec.get("novelty", 0.5))
        high_heat_damp = 1.0 - 0.22 * np.tanh(max(0.0, h0 - 1.25) * 2.0)
        stage_drive = 0.16 * stage_local_boost
        drive_h = (0.10 + 0.22 * prov + 0.12 * sentiment_abs + 0.05 * cred + stage_drive) * high_heat_damp
        drive_c = 0.12 + 0.30 * prov + 0.13 * (signal.get("blame_target") == "referee")

        h1 = 0.90 * h0 + 0.06 * c0 + 0.03 * n0 + drive_h + rng.normal(0.0, 0.01)
        c1 = 0.90 * c0 + 0.04 * h0 + drive_c + rng.normal(0.0, 0.01)
        n1 = 0.86 * n0 - 0.06 * c0 - 0.03 * h0 + 0.03 * cred + rng.normal(0.0, 0.008)

        rec["heat"] = float(np.clip(h1, 0.01, 1.95))
        rec["controversy"] = float(np.clip(c1, 0.01, 1.3))
        rec["novelty"] = float(np.clip(n1, 0.01, 1.5))
        rec["last_step"] = self.market_step

    def run_post_match_dialogue(
        self, a1, a2, stage_name, context, *, rng=None,
    ):
        rng = rng or self.rng
        self.market_step += 1
        self._decay_market()

        context = dict(context or {})
        context["stage_name"] = stage_name
        topics = self._seed_topics(
            a1.team_name, a2.team_name, stage_name, context, rng,
        )

        aggregates = {
            a1.team_name: {"sentiment": 0.0, "unity_signal": 0.0, "provocation_level": 0.0, "credibility": 0.0, "blame_referee": 0.0, "n": 0},
            a2.team_name: {"sentiment": 0.0, "unity_signal": 0.0, "provocation_level": 0.0, "credibility": 0.0, "blame_referee": 0.0, "n": 0},
        }
        topic_traces = []

        actors = [a1, a2]
        for turn_idx in range(self.turns_per_match):
            speaker = actors[turn_idx % 2]
            target = a2 if speaker is a1 else a1
            speaker_diff = float(context.get("score_diff", 0.0)) if speaker is a1 else -float(context.get("score_diff", 0.0))
            local_ctx = dict(context)
            local_ctx["score_diff_for_speaker"] = speaker_diff
            local_ctx["stage_pressure"] = float(context.get("stage_pressure", 0.3))

            topic = self._pick_topic(speaker, topics, rng, stage_name)
            act = self._pick_speech_act(speaker, target, local_ctx, rng)
            text = self._compose_utterance(
                speaker.team_name, target.team_name, topic, act,
                local_ctx, rng,
            )
            signal = self._extract_signal(text, act, speaker, target, topic)
            self._update_topic_market(
                topic, signal, rng, stage_name=stage_name,
            )
            topic_traces.append((topic, signal["provocation_level"]))

            self.feed.publish(
                speaker.team_name,
                text,
                tags=["social_dialogue", stage_name, topic, act],
            )
            speaker.record_social_dialogue_event(
                content=f"[{act}] {text}",
                stage_name=stage_name,
                topic=topic,
                signal=signal,
                target=target.team_name,
            )

            agg = aggregates[speaker.team_name]
            agg["sentiment"] += float(signal["sentiment"])
            agg["unity_signal"] += float(signal["unity_signal"])
            agg["provocation_level"] += float(signal["provocation_level"])
            agg["credibility"] += float(signal["credibility"])
            agg["blame_referee"] += 1.0 if signal.get("blame_target") == "referee" else 0.0
            agg["n"] += 1

        out = {}
        for team_name, agg in aggregates.items():
            n = max(1, agg.pop("n", 1))
            out[team_name] = {
                "sentiment": agg["sentiment"] / n,
                "unity_signal": agg["unity_signal"] / n,
                "provocation_level": agg["provocation_level"] / n,
                "credibility": agg["credibility"] / n,
                "blame_referee": agg["blame_referee"] / n,
            }

        market_chaos = 0.0
        if topic_traces:
            for topic, prov in topic_traces:
                rec = self.topic_market.get(topic, {})
                market_chaos += 0.13 * float(rec.get("heat", 0.0)) + 0.085 * float(rec.get("controversy", 0.0)) + 0.18 * float(prov)
            market_chaos /= max(1, len(topic_traces))
        market_chaos = float(np.clip(market_chaos, 0.0, 0.92))

        ranked_topics = sorted(
            self.topic_market.items(),
            key=lambda kv: (
                kv[1].get("heat", 0.0) + self._stage_topic_bias(kv[0], context.get("stage_name", "")),
                kv[1].get("controversy", 0.0),
            ),
            reverse=True,
        )[:5]
        topic_snapshot = [{"topic": k, "heat": round(v["heat"], 3), "controversy": round(v["controversy"], 3)} for k, v in ranked_topics]
        return {
            "signals": out,
            "market_chaos": market_chaos,
            "topic_snapshot": topic_snapshot,
        }
