import os
import argparse
from src.data_engine.loader import load_data
from src.data_engine.cleaner import clean_results
from src.data_engine.identity_normalizer import normalize_identities
from src.memory_engine.status_score import compute_team_status
from src.memory_engine.global_exposure_gate import compute_exposure_gate
from src.memory_engine.sample_confidence import compute_sample_confidence
from src.memory_engine.team_memory_builder import build_team_memory
from src.journey_router import run_journey

def main():
    parser = argparse.ArgumentParser(description="Football Society Agents 2.1 - 2026 Journey Sandbox")
    parser.add_argument("team", type=str, help="Team to simulate")
    args = parser.parse_args()
    
    team_name = args.team
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    
    print("Loading database...")
    data = load_data(raw_dir)
    cleaned_df = clean_results(data['results'])
    normalized_df = normalize_identities(cleaned_df, data['former_names'])
    
    print("Computing metrics...")
    stats, team_matches = compute_team_status(normalized_df, data['shootouts'], current_year=2026)
    stats = compute_exposure_gate(stats, team_matches)
    stats = compute_sample_confidence(stats)
    
    try:
        print("Building structured memories...")
        memory = build_team_memory(team_name, stats, team_matches, data.get('goalscorers'))
    except ValueError as e:
        print(f"Error: {e}")
        return
        
    report = run_journey(team_name, memory, stats)
    
    out_dir = os.path.join(base_dir, 'outputs', 'journey_simulations')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{team_name.replace(' ', '_')}_2026_Journey.md")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
        
    print(f"\n[Success] Journey Simulation saved to {out_path}")

if __name__ == "__main__":
    main()
