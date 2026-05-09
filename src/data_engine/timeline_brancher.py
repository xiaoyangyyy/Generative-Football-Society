import pandas as pd

def branch_timeline(df: pd.DataFrame, date: str, team_a: str, team_b: str, new_result_for_a: str) -> pd.DataFrame:
    df_branched = df.copy()
    target_date = pd.to_datetime(date)
    
    mask = (pd.to_datetime(df_branched['date']) == target_date) & \
           (((df_branched['home_team'] == team_a) & (df_branched['away_team'] == team_b)) | \
            ((df_branched['home_team'] == team_b) & (df_branched['away_team'] == team_a)))
            
    matches = df_branched[mask]
    if len(matches) == 0:
        raise ValueError(f"No match found on {date} between {team_a} and {team_b}")
        
    idx = matches.index[0]
    is_home = df_branched.loc[idx, 'home_team'] == team_a
    
    home_score = float(df_branched.loc[idx, 'home_score'])
    away_score = float(df_branched.loc[idx, 'away_score'])

    if new_result_for_a == 'win':
        if is_home:
            home_score = max(home_score, away_score + 1.0)
        else:
            away_score = max(away_score, home_score + 1.0)
    elif new_result_for_a == 'loss':
        if is_home:
            away_score = max(away_score, home_score + 1.0)
        else:
            home_score = max(home_score, away_score + 1.0)
    elif new_result_for_a == 'draw':
        eq_score = max(home_score, away_score)
        home_score = eq_score
        away_score = eq_score

    df_branched.loc[idx, 'home_score'] = home_score
    df_branched.loc[idx, 'away_score'] = away_score

    # Recompute downstream columns used by score engines.
    diff = home_score - away_score
    df_branched.loc[idx, 'goal_diff_home'] = diff
    df_branched.loc[idx, 'is_draw'] = diff == 0
    if diff > 0:
        df_branched.loc[idx, 'home_result'] = 'win'
        df_branched.loc[idx, 'away_result'] = 'loss'
        df_branched.loc[idx, 'winner_90min'] = df_branched.loc[idx, 'home_team']
    elif diff < 0:
        df_branched.loc[idx, 'home_result'] = 'loss'
        df_branched.loc[idx, 'away_result'] = 'win'
        df_branched.loc[idx, 'winner_90min'] = df_branched.loc[idx, 'away_team']
    else:
        df_branched.loc[idx, 'home_result'] = 'draw'
        df_branched.loc[idx, 'away_result'] = 'draw'
        df_branched.loc[idx, 'winner_90min'] = 'draw'
        
    return df_branched
