from __future__ import annotations

from gfs_core.contracts.status import StatusSnapshot


class JsonReportService:
    def status(
        self, snapshot: StatusSnapshot, limit: int,
    ) -> list[dict[str, object]]:
        return snapshot.ranked(limit).reset_index().to_dict(orient="records")

    def monte_carlo(
        self, rows: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        return [dict(row) for row in rows]

    @staticmethod
    def micro_match(summary) -> dict[str, object]:
        return {
            "home": summary.home_team,
            "away": summary.away_team,
            "score": [summary.goals_micro_home, summary.goals_micro_away],
            "xg": [summary.micro_xg_home, summary.micro_xg_away],
            "passes": [summary.passes_home, summary.passes_away],
            "pass_completion": [
                summary.pass_completion_home,
                summary.pass_completion_away,
            ],
            "shots": [summary.shots_home, summary.shots_away],
            "possession_home": summary.possession_home,
            "ticks": summary.ticks,
        }
