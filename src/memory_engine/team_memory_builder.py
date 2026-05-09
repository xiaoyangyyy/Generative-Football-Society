import pandas as pd
from src.memory_engine.tournament_weights import get_tournament_weight

def build_team_memory(team: str, stats: pd.DataFrame, team_matches: pd.DataFrame, goalscorers_df: pd.DataFrame = None) -> dict:
    if team not in stats.index:
        raise ValueError(f"Team {team} not found in stats.")
        
    team_stat = stats.loc[team]
    matches = team_matches[team_matches['team'] == team]
    
    historical_stats = {
        "matches": int(team_stat['total_matches']),
        "wins": int((matches['result'] == 'win').sum()),
        "draws": int((matches['result'] == 'draw').sum()),
        "losses": int((matches['result'] == 'loss').sum()),
        "weighted_win_rate": round(team_stat['c1_win_rate'] / 100.0, 2)
    }
    
    t_counts = matches.groupby('tournament').size()
    main_tournaments = []
    for t_name, count in t_counts.nlargest(8).items():
        main_tournaments.append({
            "name": t_name,
            "matches": int(count),
            "weight": get_tournament_weight(t_name)
        })
        
    r_counts = matches.groupby('opponent').size()
    main_rivals = []
    for opp, count in r_counts.nlargest(5).items():
        main_rivals.append({"opponent": opp, "matches": int(count)})
        
    glory_df = matches[(matches['result'] == 'win') & (matches['t_weight'] >= 0.70)].sort_values(by=['t_weight', 'goal_diff'], ascending=[False, False]).head(5)
    top_glory = []
    for _, row in glory_df.iterrows():
        top_glory.append({
            "date": row['date'].strftime('%Y-%m-%d'),
            "opponent": row['opponent'],
            "tournament": row['tournament'],
            "result": "win",
            "weight": float(row['t_weight']),
            "reason": f"goal_diff: +{row['goal_diff']}"
        })
        
    trauma_df = matches[(matches['result'] == 'loss') & (matches['t_weight'] >= 0.70)].sort_values(by=['t_weight', 'goal_diff'], ascending=[False, True]).head(5)
    top_trauma = []
    for _, row in trauma_df.iterrows():
        top_trauma.append({
            "date": row['date'].strftime('%Y-%m-%d'),
            "opponent": row['opponent'],
            "tournament": row['tournament'],
            "result": "loss",
            "weight": float(row['t_weight']),
            "reason": f"goal_diff: {row['goal_diff']}"
        })
        
    recent_df = matches.sort_values(by='date', ascending=False).head(10)
    recent_context = []
    for _, row in recent_df.iterrows():
        recent_context.append({
            "date": row['date'].strftime('%Y-%m-%d'),
            "opponent": row['opponent'],
            "result": row['result'],
            "tournament": row['tournament']
        })
        
    # Era Slicing for Historian
    eras = [
        ("1872-1930", "1872-01-01", "1930-12-31"),
        ("1931-1970", "1931-01-01", "1970-12-31"),
        ("1971-1990", "1971-01-01", "1990-12-31"),
        ("1991-2010", "1991-01-01", "2010-12-31"),
        ("2011-2026", "2011-01-01", "2026-12-31")
    ]
    era_slices = {}
    for era_name, start, end in eras:
        mask = (matches['date'] >= start) & (matches['date'] <= end)
        era_matches = matches[mask]
        if len(era_matches) > 0:
            era_wins = (era_matches['result'] == 'win').sum()
            era_slices[era_name] = {
                "matches": len(era_matches),
                "win_rate": round(era_wins / len(era_matches) * 100, 1)
            }
            
    top_scorers = []
    if goalscorers_df is not None and not goalscorers_df.empty and 'team' in goalscorers_df.columns:
        gs = goalscorers_df.copy()
        gs['date'] = pd.to_datetime(gs['date'], errors='coerce')
        team_goals = gs[gs['team'] == team]
        scorer_counts = team_goals.groupby('scorer').size().sort_values(ascending=False).head(5)
        for scorer, goals in scorer_counts.items():
            top_scorers.append({"name": scorer, "goals": int(goals)})

    memory = {
        "team": team,
        "identity_scope": "international representative team",
        "historical_stats": historical_stats,
        "era_slices": era_slices,
        "status_scores": {
            "historical_base": round(team_stat['historical_base'], 1),
            "modern_power": round(team_stat['modern_power'], 1),
            "final_status_score": round(team_stat['final_status_score'], 1),
            "sample_confidence": team_stat['sample_confidence'],
            "global_exposure_gate": team_stat['global_exposure_gate'],
            "tier": team_stat['tier']
        },
        "main_tournaments": main_tournaments,
        "main_rivals": main_rivals,
        "top_scorers": top_scorers,
        "top_glory_matches": top_glory,
        "top_trauma_matches": top_trauma,
        "recent_context": recent_context,
        "data_warnings": [
            "The dataset does not include tactics, formations, possession, shots, xG, or player positions.",
            "Match stage is unavailable unless an external enrichment table is provided."
        ]
    }
    
    return memory
