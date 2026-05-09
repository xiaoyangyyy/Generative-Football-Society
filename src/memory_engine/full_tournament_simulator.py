import random
import pandas as pd
from itertools import combinations
from src.memory_engine.probability_engine import calculate_match_probabilities
from src.memory_engine.poisson_simulator import simulate_match_score, simulate_penalty_shootout
from src.simulation.tournament_2026 import WORLD_CUP_2026_GROUPS

GROUPS_2026 = {k.split()[-1]: v for k, v in WORLD_CUP_2026_GROUPS.items()}

def get_score(team, stats):
    if team in stats.index:
        return stats.loc[team]['final_status_score']
    return 30.0

def play_match_poisson(team_a, team_b, stats, is_knockout=False, verbose=True):
    """
    Simulate a single match using Poisson distribution.
    Returns: (winner_code, goals_a, goals_b, penalty_info, log_line)
      winner_code: 'a', 'b', or 'draw' (only in group stage)
    """
    sa = get_score(team_a, stats)
    sb = get_score(team_b, stats)
    
    goals_a, goals_b, xg_a, xg_b = simulate_match_score(sa, sb, is_knockout=is_knockout)
    
    penalty_str = ""
    
    if goals_a > goals_b:
        winner = 'a'
    elif goals_b > goals_a:
        winner = 'b'
    else:
        if is_knockout:
            pen_a, pen_b = simulate_penalty_shootout()
            penalty_str = f" **(Pen {pen_a}-{pen_b})**"
            winner = 'a' if pen_a > pen_b else 'b'
        else:
            winner = 'draw'
    
    if verbose:
        probs = calculate_match_probabilities(
            pd.Series({'final_status_score': sa}),
            pd.Series({'final_status_score': sb})
        )
        log = (
            f"- **{team_a}** {goals_a} - {goals_b} **{team_b}**{penalty_str} "
            f"| xG: {xg_a} vs {xg_b} "
            f"| Status: {sa:.1f} vs {sb:.1f} "
            f"| Win%: {probs['p_win_a']:.1f} / {probs['p_draw']:.1f} / {probs['p_win_b']:.1f}"
        )
    else:
        log = ""
    
    return winner, goals_a, goals_b, penalty_str, log


def simulate_full_tournament(stats, verbose=True):
    """
    Run one complete 48-team World Cup simulation with Poisson-generated scorelines.
    If verbose=True, generates a full Markdown report.
    If verbose=False (Monte Carlo mode), returns only the champion name.
    """
    report_lines = []
    if verbose:
        report_lines.append("# 2026 World Cup: Full Poisson Match Simulation\n")
        report_lines.append("## 1. Group Stage (小组赛)\n")
    
    group_standings = {}
    
    for group_name, teams in GROUPS_2026.items():
        if verbose:
            report_lines.append(f"### Group {group_name}")
        
        record = {t: {"pts": 0, "gf": 0, "ga": 0} for t in teams}
        
        matches = list(combinations(teams, 2))
        for team_a, team_b in matches:
            winner, ga, gb, pen, log = play_match_poisson(team_a, team_b, stats, is_knockout=False, verbose=verbose)
            
            if verbose:
                report_lines.append(log)
            
            record[team_a]["gf"] += ga
            record[team_a]["ga"] += gb
            record[team_b]["gf"] += gb
            record[team_b]["ga"] += ga
            
            if winner == 'a':
                record[team_a]["pts"] += 3
            elif winner == 'b':
                record[team_b]["pts"] += 3
            else:
                record[team_a]["pts"] += 1
                record[team_b]["pts"] += 1
        
        def sort_key(t):
            r = record[t]
            gd = r["gf"] - r["ga"]
            return (r["pts"], gd, r["gf"], get_score(t, stats))
        
        ranked = sorted(teams, key=sort_key, reverse=True)
        group_standings[group_name] = []
        for t in ranked:
            r = record[t]
            gd = r["gf"] - r["ga"]
            group_standings[group_name].append({
                "team": t, "pts": r["pts"], "gf": r["gf"], "ga": r["ga"], "gd": gd
            })
        
        if verbose:
            report_lines.append("")
            report_lines.append(f"| Pos | Team | Pts | GF | GA | GD |")
            report_lines.append(f"|-----|------|-----|----|----|-----|")
            for pos, entry in enumerate(group_standings[group_name], 1):
                gd_str = f"+{entry['gd']}" if entry['gd'] >= 0 else str(entry['gd'])
                report_lines.append(f"| {pos} | {entry['team']} | {entry['pts']} | {entry['gf']} | {entry['ga']} | {gd_str} |")
            report_lines.append("")
    
    # --- Advancing teams ---
    advancing_teams = []
    third_places = []
    
    for group_name, rankings in group_standings.items():
        advancing_teams.append(rankings[0]['team'])
        advancing_teams.append(rankings[1]['team'])
        third_places.append(rankings[2])
    
    third_places = sorted(
        third_places,
        key=lambda x: (x['pts'], x['gd'], x['gf'], get_score(x['team'], stats)),
        reverse=True
    )
    best_thirds = [x['team'] for x in third_places[:8]]
    advancing_teams.extend(best_thirds)
    
    if verbose:
        report_lines.append("## 2. Qualified Teams (晋级名单)\n")
        report_lines.append(f"**Top 2 from each group (24 teams)**: {', '.join(advancing_teams[:24])}")
        report_lines.append(f"**Best 8 Third-Places**: {', '.join(best_thirds)}\n")
    
    # --- Knockout stage ---
    random.shuffle(advancing_teams)
    
    stage_names = [
        "Round of 32 (1/16 决赛)",
        "Round of 16 (1/8 决赛)",
        "Quarter-Finals (1/4 决赛)",
        "Semi-Finals (半决赛)",
        "Final (决赛)"
    ]
    
    if verbose:
        report_lines.append("## 3. Knockout Stage (淘汰赛)\n")
    
    current_pool = advancing_teams
    
    for stage_name in stage_names:
        if len(current_pool) < 2:
            break
        
        if verbose:
            report_lines.append(f"### {stage_name}")
        
        next_pool = []
        for i in range(0, len(current_pool), 2):
            team_a = current_pool[i]
            team_b = current_pool[i+1]
            
            winner, ga, gb, pen, log = play_match_poisson(team_a, team_b, stats, is_knockout=True, verbose=verbose)
            
            if verbose:
                report_lines.append(log)
            
            next_pool.append(team_a if winner == 'a' else team_b)
        
        if verbose:
            report_lines.append("")
        current_pool = next_pool
    
    champion = current_pool[0]
    
    if verbose:
        report_lines.append(f"## 🏆 CHAMPIONS (冠军): **{champion}**")
        return "\n".join(report_lines)
    else:
        return champion
