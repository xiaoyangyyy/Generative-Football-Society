import os
import pandas as pd
from src.data_engine.loader import load_data
from src.data_engine.cleaner import clean_results
from src.data_engine.identity_normalizer import normalize_identities
from src.memory_engine.status_score import compute_team_status
from src.memory_engine.global_exposure_gate import compute_exposure_gate
from src.memory_engine.sample_confidence import compute_sample_confidence

def run_mvp1():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    
    print("Loading data...")
    data = load_data(raw_dir)
    results_df = data['results']
    shootouts_df = data['shootouts']
    former_names_df = data['former_names']
    
    print("Cleaning data...")
    cleaned_df = clean_results(results_df)
    
    print("Normalizing identities...")
    normalized_df = normalize_identities(cleaned_df, former_names_df)
    
    print("Computing status score...")
    stats, team_matches = compute_team_status(normalized_df, shootouts_df, current_year=2026)
    
    print("Computing exposure gate...")
    stats = compute_exposure_gate(stats, team_matches)
    
    print("Computing sample confidence...")
    stats = compute_sample_confidence(stats)
    
    # Sort and export top 20
    top20 = stats.sort_values(by='final_status_score', ascending=False).head(20)
    
    out_cols = [
        'total_matches', 'historical_base', 'modern_power', 'final_status_score', 
        'global_exposure_gate', 'sample_confidence', 'tier'
    ]
    
    output_path = os.path.join(base_dir, 'status_scores_top20.csv')
    top20[out_cols].to_csv(output_path)
    
    print(f"Done! Top 20 ranking exported to {output_path}")
    print("\nTop 10 Snapshot:")
    print(top20[['final_status_score', 'tier', 'global_exposure_gate']].head(10))

if __name__ == "__main__":
    run_mvp1()
