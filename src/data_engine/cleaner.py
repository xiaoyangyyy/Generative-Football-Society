import pandas as pd

def clean_results(df):
    """
    Cleans the raw international results data.
    """
    df = df.copy()
    # Convert date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Keep only completed matches; do not convert future fixtures into artificial 0-0 draws.
    df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
    df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
    df = df.dropna(subset=['date', 'home_team', 'away_team', 'home_score', 'away_score']).copy()
    df = df[df['home_team'].astype(str).str.strip() != df['away_team'].astype(str).str.strip()].copy()
    df = df[(df['home_score'] >= 0) & (df['away_score'] >= 0)].copy()
    df = df.drop_duplicates(
        subset=['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament'],
        keep='last',
    )
    
    df['goal_diff_home'] = df['home_score'] - df['away_score']
    
    # Simple result
    def get_res(row):
        if row['home_score'] > row['away_score']: return 'win'
        if row['home_score'] < row['away_score']: return 'loss'
        return 'draw'
    
    df['home_result'] = df.apply(get_res, axis=1)
    df['away_result'] = df['home_result'].map({'win': 'loss', 'loss': 'win', 'draw': 'draw'})
    
    # 90min winner
    df['winner_90min'] = df.apply(lambda r: r['home_team'] if r['home_result'] == 'win' else (r['away_team'] if r['home_result'] == 'loss' else None), axis=1)
    df['is_draw'] = df['home_result'] == 'draw'
    
    return df
