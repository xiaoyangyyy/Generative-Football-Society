"""Memory vectorization, similarity search, and context retrieval."""

import numpy as np


class AgentMemoryRetrievalMixin:
    def retrieve_similar_decision_memories(self, controls, top_k=4, current_stage=None, current_opponent_style=None):
        if not self.decision_memory:
            return []
        target = self._control_vector(controls)
        stage_target = current_stage or (self.decision_memory[-1].get("stage") if self.decision_memory else None)
        style_target = current_opponent_style or (self.decision_memory[-1].get("opponent_style") if self.decision_memory else None)
        scored = []
        for rec in self.decision_memory:
            vec = self._control_vector(rec.get("controls", {}))
            dist = float(np.linalg.norm(target - vec))
            out = rec.get("outcomes", {})
            utility = (
                0.9 * float(out.get("goal_diff", 0.0))
                + 0.25 * float(out.get("xg_for", 0.0) - out.get("xg_against", 0.0))
                - 0.7 * float(out.get("fatigue_delta", 0.0))
                - 0.15 * abs(float(out.get("chaos", 0.0)))
            )
            # Stage and style context weighting for better transfer.
            stage_bonus = 0.18 if stage_target and rec.get("stage") == stage_target else 0.0
            style_bonus = 0.20 if style_target and rec.get("opponent_style") == style_target else 0.0
            score = utility - 1.25 * dist + stage_bonus + style_bonus
            scored.append((score, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[: max(1, top_k)]]


    def _memory_vector(self, rec):
        app = rec.get("appraisal", {})
        emo = rec.get("emotion_profile", {})
        cop = rec.get("coping", {})
        return np.array(
            [
                float(app.get("impact", 0.0)),
                float(app.get("novelty", 0.5)),
                float(app.get("control", 0.5)),
                float(app.get("certainty", 0.5)),
                float(app.get("norm_violation", 0.0)),
                float(app.get("agency", 0.0)),
                float(emo.get("pride", 0.0)),
                float(emo.get("anger", 0.0)),
                float(emo.get("shame", 0.0)),
                float(emo.get("fear", 0.0)),
                float(emo.get("determination", 0.0)),
                float(cop.get("planning", 0.0)),
                float(cop.get("self_correction", 0.0)),
                float(cop.get("external_blame", 0.0)),
                float(cop.get("risk_shift", 0.0)),
            ],
            dtype=float,
        )

    def _cosine_similarity(self, a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= self.eps or nb <= self.eps:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _query_vector(self, opponent=None, stage_pressure=0.3):
        app = dict(self.appraisal_state)
        app["novelty"] = float(np.clip(0.6 * app.get("novelty", 0.5) + 0.4 * stage_pressure, 0.0, 1.0))
        if opponent is not None:
            app["agency"] = float(np.tanh(app.get("agency", 0.0) + 0.15 * self.rivalry_database.get(opponent, 0.0)))
        rec = {
            "appraisal": app,
            "emotion_profile": dict(self.emotion_profile),
            "coping": dict(self.coping_profile),
        }
        return self._memory_vector(rec)

    def retrieve_memory_context(self, top_k=None, opponent=None):
        if not self.episodic_memory:
            return []
        q = self._query_vector(opponent=opponent, stage_pressure=0.3)
        logits = []
        recs = []
        for rec in self.episodic_memory:
            sim = self._cosine_similarity(q, self._memory_vector(rec))
            mw = float(max(self.eps, rec.get("memory_weight", rec.get("salience", 0.0))))
            logit = self.gamma_retrieval * sim + self.eta_retrieval * np.log(mw + self.eps)
            if opponent and rec.get("opponent") == opponent:
                logit += 0.15
            logits.append(logit)
            recs.append(rec)
        arr = np.array(logits, dtype=float)
        arr = arr - np.max(arr)
        w = np.exp(arr)
        w = w / max(self.eps, float(np.sum(w)))
        weighted = []
        for ww, rec in zip(w.tolist(), recs):
            rc = dict(rec)
            rc["retrieval_weight"] = float(ww)
            weighted.append(rc)
        weighted.sort(key=lambda r: r.get("retrieval_weight", 0.0), reverse=True)
        selected = weighted if top_k is None else weighted[: max(1, top_k)]
        selected_ids = {record.get("id") for record in selected}
        for record in self.episodic_memory:
            if record.get("id") in selected_ids:
                record["retrieval_count"] = int(record.get("retrieval_count", 0)) + 1
        if top_k is None:
            return selected
        # top_k is only for display/export convenience, not for core weighting math.
        return selected


    def retrieve_memory_context_display(self, top_k=10, opponent=None):
        return self.retrieve_memory_context(top_k=max(1, int(top_k)), opponent=opponent)

    def retrieve_memory_by_tags(self, tags, top_k=8):
        tags = set(tags or [])
        if not tags:
            return self.retrieve_memory_context_display(top_k=top_k)
        base = self.retrieve_memory_context(top_k=None)
        for rec in base:
            overlap = len(tags & set(rec.get("tags", [])))
            rec["retrieval_weight"] = float(rec.get("retrieval_weight", 0.0) * (1.0 + 0.35 * overlap))
        base.sort(key=lambda r: r.get("retrieval_weight", 0.0), reverse=True)
        return base[: max(1, top_k)]

