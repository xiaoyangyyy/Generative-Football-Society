import os
from src.data_engine.loader import load_data
from src.data_engine.cleaner import clean_results
from src.data_engine.identity_normalizer import normalize_identities
from src.memory_engine.status_score import compute_team_status
from src.memory_engine.global_exposure_gate import compute_exposure_gate
from src.memory_engine.sample_confidence import compute_sample_confidence
from src.memory_engine.full_tournament_simulator import simulate_full_tournament

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    
    print("Loading database...")
    data = load_data(raw_dir)
    cleaned_df = clean_results(data['results'])
    normalized_df = normalize_identities(cleaned_df, data['former_names'])
    
    print("Computing status metrics for all teams...")
    stats, team_matches = compute_team_status(normalized_df, data['shootouts'], current_year=2026)
    stats = compute_exposure_gate(stats, team_matches)
    stats = compute_sample_confidence(stats)
    
    print("Running Full Tournament Simulation (Math & Probability Engine)...")
    report = simulate_full_tournament(stats)
    
    out_dir = os.path.join(base_dir, 'outputs', 'tournament_simulations')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "2026_Full_Tournament_Report.md")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
        
    print(f"\n[Success] Full 48-Team Tournament Simulation Log saved to {out_path}")

if __name__ == "__main__":
    main()
