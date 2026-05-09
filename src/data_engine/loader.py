import pandas as pd
import os

def load_data(raw_data_dir: str):
    """Loads the four raw CSV files."""
    results_path = os.path.join(raw_data_dir, 'results.csv')
    goalscorers_path = os.path.join(raw_data_dir, 'goalscorers.csv')
    shootouts_path = os.path.join(raw_data_dir, 'shootouts.csv')
    former_names_path = os.path.join(raw_data_dir, 'former_names.csv')
    
    results_df = pd.read_csv(results_path)
    goalscorers_df = pd.read_csv(goalscorers_path)
    shootouts_df = pd.read_csv(shootouts_path)
    former_names_df = pd.read_csv(former_names_path)
    
    return {
        'results': results_df,
        'goalscorers': goalscorers_df,
        'shootouts': shootouts_df,
        'former_names': former_names_df
    }
