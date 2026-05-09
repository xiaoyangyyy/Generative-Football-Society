import numpy as np

def calculate_time_decay(match_years_ago, tournament_weights=None, base_half_life: float = 15.0):
    """
    time_decay = exp(-lambda * years_ago)
    where lambda = ln(2) / half_life.
    
    If tournament_weights are provided, half_life is increased for major events:
    - Weight >= 0.8: 50 year half-life (Generational Legacy)
    - Weight >= 0.5: 25 year half-life (Mid-term Legacy)
    """
    # Initialize half_life array or scalar
    if tournament_weights is not None:
        # Dynamic half-life mapping
        # Major (0.8+) -> 50.0
        # Med (0.5-0.8) -> 25.0
        # Minor (<0.5) -> 15.0
        
        # Using numpy where for vectorized mapping if it's a series
        hl = np.where(tournament_weights >= 0.8, 50.0, 
                np.where(tournament_weights >= 0.5, 25.0, base_half_life))
    else:
        hl = base_half_life

    lam = np.log(2) / hl
    years_ago_clipped = np.clip(match_years_ago, 0, None)
    return np.exp(-lam * years_ago_clipped)
