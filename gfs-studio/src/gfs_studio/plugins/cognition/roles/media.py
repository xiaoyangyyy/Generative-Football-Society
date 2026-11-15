class MediaRole:
    def review(self, summary) -> dict:
        score_delta = summary.goals_micro_home - summary.goals_micro_away
        xg_delta = summary.micro_xg_home - summary.micro_xg_away
        sentiment = max(-1.0, min(1.0, 0.4 * score_delta + 0.2 * xg_delta))
        return {
            "headline": (
                f"{summary.home_team} and {summary.away_team}: "
                "a match defined by fine margins"
            ),
            "sentiment": sentiment,
            "media_pressure_delta": 0.05 * sentiment,
            "causal_claim": False,
        }
