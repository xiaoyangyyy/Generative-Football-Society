import random
import pandas as pd
from src.memory_engine.probability_engine import calculate_match_probabilities
from src.simulation.random_control import named_py_rng
from src.simulation.tournament_2026 import WORLD_CUP_2026_GROUPS

GROUPS_2026 = {k.split()[-1]: v for k, v in WORLD_CUP_2026_GROUPS.items()}

def get_knockout_opponents(stats, exclude_teams, stage, *, rng: random.Random):
    if stage == "Round of 32":
        pool = stats.nlargest(40, 'final_status_score').index.tolist()
    elif stage == "Round of 16":
        pool = stats.nlargest(24, 'final_status_score').index.tolist()
    elif stage == "Quarter-Final":
        pool = stats.nlargest(12, 'final_status_score').index.tolist()
    else:
        pool = stats.nlargest(8, 'final_status_score').index.tolist()
        
    valid_pool = [t for t in pool if t not in exclude_teams]
    if not valid_pool:
        valid_pool = [t for t in stats.index if t not in exclude_teams]
    
    return rng.choice(valid_pool)

def simulate_journey(team, stats, *, root_seed=42, rng=None):
    rng = rng or named_py_rng(root_seed, "legacy_journey", team)
    if team not in stats.index:
        raise ValueError(f"Team {team} not found in database.")
        
    my_group_name = None
    my_group_teams = []
    for g_name, teams in GROUPS_2026.items():
        if team in teams:
            my_group_name = g_name
            my_group_teams = [t for t in teams if t != team]
            break
            
    if not my_group_name:
        raise ValueError(f"Team {team} is not in the 2026 World Cup 48-team draw.")
        
    journey_log = []
    points = 0
    
    for opp in my_group_teams:
        if opp not in stats.index:
            stat_opp = pd.Series({'final_status_score': 30.0})
        else:
            stat_opp = stats.loc[opp]
            
        probs = calculate_match_probabilities(stats.loc[team], stat_opp)
        roll = rng.uniform(0, 100)
        
        if roll <= probs['p_win_a']:
            res = "win"
            points += 3
        elif roll <= probs['p_win_a'] + probs['p_draw']:
            res = "draw"
            points += 1
        else:
            res = "loss"
            
        journey_log.append({
            "stage": f"Group {my_group_name}",
            "opponent": opp,
            "result": res,
            "details": f"Points: {points}"
        })
        
    advances = False
    if points >= 4:
        advances = True
    elif points == 3:
        advances = rng.random() > 0.5
        
    if not advances:
        return journey_log, "Eliminated in Group Stage"
        
    stages = ["Round of 32", "Round of 16", "Quarter-Final", "Semi-Final", "Final"]
    played_teams = set(my_group_teams + [team])
    
    for stage in stages:
        opp = get_knockout_opponents(stats, played_teams, stage, rng=rng)
        played_teams.add(opp)
        
        stat_opp = stats.loc[opp]
        probs = calculate_match_probabilities(stats.loc[team], stat_opp)
        
        p_win_adj = probs['p_win_a'] + (probs['p_draw'] / 2)
        
        roll = rng.uniform(0, 100)
        if roll <= p_win_adj:
            res = "win"
            journey_log.append({
                "stage": stage,
                "opponent": opp,
                "result": res,
                "details": "Advanced"
            })
        else:
            res = "loss"
            journey_log.append({
                "stage": stage,
                "opponent": opp,
                "result": res,
                "details": "Eliminated"
            })
            return journey_log, f"Eliminated in {stage}"
            
    return journey_log, "World Cup Champions"
