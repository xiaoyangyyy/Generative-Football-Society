import pandas as pd
import os


REQUIRED_COLUMNS = {
    "results": {"date", "home_team", "away_team", "home_score", "away_score"},
    "goalscorers": {"date", "home_team", "away_team"},
    "shootouts": {"date", "home_team", "away_team", "winner"},
    "former_names": {"former", "current"},
}


def _read_csv(path: str, kind: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required {kind} dataset not found: {path}")
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS[kind].difference(frame.columns))
    if missing:
        raise ValueError(f"Invalid {kind} dataset; missing columns: {', '.join(missing)}")
    return frame

def load_data(raw_data_dir: str):
    """Loads the four raw CSV files."""
    results_path = os.path.join(raw_data_dir, 'results.csv')
    goalscorers_path = os.path.join(raw_data_dir, 'goalscorers.csv')
    shootouts_path = os.path.join(raw_data_dir, 'shootouts.csv')
    former_names_path = os.path.join(raw_data_dir, 'former_names.csv')
    
    results_df = _read_csv(results_path, "results")
    goalscorers_df = _read_csv(goalscorers_path, "goalscorers")
    shootouts_df = _read_csv(shootouts_path, "shootouts")
    former_names_df = _read_csv(former_names_path, "former_names")
    
    return {
        'results': results_df,
        'goalscorers': goalscorers_df,
        'shootouts': shootouts_df,
        'former_names': former_names_df
    }
