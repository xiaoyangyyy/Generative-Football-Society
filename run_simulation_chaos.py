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
from src.simulation.chaos_engine import ChaosEngine

def main():
    print("=== Football Society: CHAOS MODE ACTIVATED ===")
    
    # Setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    data = load_data(raw_dir)
    cleaned_df = clean_results(data['results'])
    normalized_df = normalize_identities(cleaned_df, data['former_names'])
    stats, _ = compute_team_status(normalized_df, data['shootouts'], current_year=2026)
    
    engine = WorldEngine(stats)
    chaos = ChaosEngine(engine)
    
    # 1. Normal Evolution (3 days)
    for _ in range(3):
        engine.run_day()
        
    # 2. TRIGGER RAGNAROK
    chaos.trigger_black_swan("ragnarok")
    
    # 3. TRIGGER ORIENTAL ASCENDANCY
    chaos.trigger_black_swan("oriental_ascendancy")
    
    # 4. Evolution under Chaos (7 days)
    for _ in range(7):
        engine.run_day()
        
    # 5. Final Report
    print("\n" + "="*50)
    print("CHAOS SUMMARY: THE NEW WORLD ORDER")
    print("="*50)
    
    # Gossip Heatmap
    heatmap = chaos.get_gossip_keywords()
    print(f"\nGLOBAL GOSSIP HEATMAP: {heatmap}")
    
    # Top 5 New Status Leaders
    leaders = sorted(engine.agents.items(), key=lambda x: x[1].status_score, reverse=True)[:5]
    print("\nTOP 5 STATUS LEADERS (NEW ORDER):")
    for name, agent in leaders:
        print(f"- {name}: {agent.status_score:.1f} ({agent.tier})")

    # Show a few social posts during the chaos
    print("\nLATEST SOCIAL REACTIONS:")
    for post in engine.feed.posts[-5:]:
        print(f"[{post['author']}]: {post['content']}")

if __name__ == "__main__":
    main()
