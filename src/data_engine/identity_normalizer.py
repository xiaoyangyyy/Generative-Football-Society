import pandas as pd
from src.data_engine.team_aliases import TEAM_ALIASES

def normalize_identities(df: pd.DataFrame, former_names_df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Map former team names to current (canonical) names
    2. Derive result fields: home_result, away_result, winner_90min, goal_diff_home, goal_diff_abs, total_goals, is_draw
    """
    df = df.copy()
    
    # 1. Map names
    name_map = dict(zip(former_names_df['former'], former_names_df['current']))
    
    # Keep original names just in case
    df['home_team_original'] = df['home_team']
    df['away_team_original'] = df['away_team']
    
    df['home_team'] = df['home_team'].replace(name_map).replace(TEAM_ALIASES)
    df['away_team'] = df['away_team'].replace(name_map).replace(TEAM_ALIASES)
    
    # 2. Derive results
    df['home_score'] = df['home_score'].astype(float)
    df['away_score'] = df['away_score'].astype(float)
    
    df['goal_diff_home'] = df['home_score'] - df['away_score']
    df['goal_diff_abs'] = df['goal_diff_home'].abs()
    df['total_goals'] = df['home_score'] + df['away_score']
    df['is_draw'] = df['home_score'] == df['away_score']
    
    def get_home_result(diff):
        if diff > 0: return 'win'
        if diff < 0: return 'loss'
        return 'draw'
        
    def get_away_result(diff):
        if diff < 0: return 'win'
        if diff > 0: return 'loss'
        return 'draw'
        
    def get_winner_90min(row):
        if row['goal_diff_home'] > 0: return row['home_team']
        if row['goal_diff_home'] < 0: return row['away_team']
        return 'draw'
        
    df['home_result'] = df['goal_diff_home'].apply(get_home_result)
    df['away_result'] = df['goal_diff_home'].apply(get_away_result)
    df['winner_90min'] = df.apply(get_winner_90min, axis=1)
    
    return df
