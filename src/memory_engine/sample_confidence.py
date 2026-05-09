import pandas as pd
import numpy as np

def compute_sample_confidence(stats: pd.DataFrame, threshold: int = 80) -> pd.DataFrame:
    """
    sample_confidence_score = min(1, sqrt(weighted_match_count / threshold))
    """
    stats = stats.copy()
    
    stats['sample_confidence_score'] = np.minimum(1.0, np.sqrt(stats['total_matches'] / threshold))
    
    def get_confidence_label(score):
        if score >= 0.80: return 'high'
        if score >= 0.50: return 'medium'
        return 'low'
        
    stats['sample_confidence'] = stats['sample_confidence_score'].apply(get_confidence_label)
    return stats
