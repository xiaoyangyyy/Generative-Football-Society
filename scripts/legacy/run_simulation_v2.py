import sys
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_engine.loader import load_data
from src.data_engine.cleaner import clean_results
from src.data_engine.identity_normalizer import normalize_identities
from src.memory_engine.status_score import compute_team_status
from src.simulation.engine import WorldEngine

def main():
    print("=== Football Society: Generative Simulation v2 (SOCIAL) ===")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    
    data = load_data(raw_dir)
    cleaned_df = clean_results(data['results'])
    normalized_df = normalize_identities(cleaned_df, data['former_names'])
    stats, _ = compute_team_status(normalized_df, data['shootouts'], current_year=2026)
    
    engine = WorldEngine(stats)
    
    # Run for 10 days
    for _ in range(10):
        engine.run_day()
    
    print("\n" + "="*50)
    print("SIMULATION LOG: TOP SOCIAL INTERACTIONS")
    print("="*50)
    
    # Show some social media threads
    for post in engine.feed.posts[-5:]:
        print(f"\n[{post['author']}]: {post['content']}")
        for reply in post['replies']:
            print(f"   └── [REPLY] {reply['author']}: {reply['content']}")

if __name__ == "__main__":
    main()
