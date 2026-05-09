import math
import pandas as pd

def calculate_match_probabilities(team_a_stats: pd.Series, team_b_stats: pd.Series) -> dict:
    """
    Calculate sociological match probabilities using a modified Elo/Logit function.
    Since Status Scores are on a 0-100 scale, a divisor of 20 makes a 10-point diff significant.
    """
    score_a = team_a_stats['final_status_score']
    score_b = team_b_stats['final_status_score']
    
    diff = score_a - score_b
    
    # P(A wins) base probability ignoring draws
    p_a_win_base = 1 / (1 + math.pow(10, -diff / 80.0))
    p_b_win_base = 1 - p_a_win_base
    
    # Draw probability is highest (~33%) when teams are evenly matched (diff = 0)
    # Uses a Gaussian curve with sigma=40
    draw_prob = 0.33 * math.exp(-(diff**2) / (2 * (40**2)))
    
    # Distribute remaining probability to wins
    p_a_win = p_a_win_base * (1 - draw_prob)
    p_b_win = p_b_win_base * (1 - draw_prob)
    
    total = p_a_win + p_b_win + draw_prob
    
    return {
        "score_a": round(score_a, 1),
        "score_b": round(score_b, 1),
        "p_win_a": round((p_a_win / total) * 100, 1),
        "p_draw": round((draw_prob / total) * 100, 1),
        "p_win_b": round((p_b_win / total) * 100, 1)
    }
