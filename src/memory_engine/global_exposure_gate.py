import pandas as pd

def compute_exposure_gate(stats: pd.DataFrame, team_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Global Exposure Gate:
    1. World Cup matches
    2. Matches vs Top 20
    Outputs: pass, partial_pass, fail
    """
    stats = stats.copy()
    
    wc_matches = team_matches[team_matches['tournament'] == 'FIFA World Cup']
    stats['wc_matches'] = wc_matches.groupby('team').size().reindex(stats.index, fill_value=0)
    
    top20_matches = team_matches[team_matches['is_top_opponent']]
    stats['top20_matches'] = top20_matches.groupby('team').size().reindex(stats.index, fill_value=0)
    
    def evaluate_gate(row):
        wc = row['wc_matches']
        top20 = row['top20_matches']
        
        if wc >= 10 and top20 >= 30:
            return 'pass'
        elif wc >= 3 or top20 >= 10:
            return 'partial_pass'
        else:
            return 'fail'
            
    stats['global_exposure_gate'] = stats.apply(evaluate_gate, axis=1)
    
    def get_tier(row):
        score = row['final_status_score']
        gate = row['global_exposure_gate']
        
        if score >= 75 and gate == 'pass':
            return 'Core Power'
        elif score >= 60 and gate in ['pass', 'partial_pass']:
            return 'Semi-Core'
        elif score >= 40:
            return 'Regional Power'
        elif score >= 25 and row['modern_power'] > row['historical_base'] + 10:
            return 'Emerging Actor'
        else:
            return 'Peripheral Actor'
            
    stats['tier'] = stats.apply(get_tier, axis=1)
    return stats
