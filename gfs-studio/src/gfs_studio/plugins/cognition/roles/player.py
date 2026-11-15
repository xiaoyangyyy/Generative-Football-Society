class PlayerRole:
    def review(self, summary) -> dict:
        stats = summary.player_stats.get(summary.home_team, {})
        high_minutes = [
            player_id for player_id, row in stats.items()
            if float(row.get("minutes", 0.0)) >= 60.0
        ]
        return {
            "high_workload_players": high_minutes,
            "rotation_recommended": bool(high_minutes),
            "suspension_or_medical_inference": False,
        }
