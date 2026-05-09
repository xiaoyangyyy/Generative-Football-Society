import sys
print("[DIAG] Script started.")
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.simulation.world_cup_runner import build_world_and_tournament
from src.simulation.media_logic import MediaOutlet, simulate_counterfactual_media
from src.simulation.random_control import set_global_seed
from src.visualization.sentiment_viz import plot_sentiment_pulse

def main():
    print("=== WORLD CUP 2026: DEEP MEDIA & SENTIMENT LAB ===")
    used_seed = set_global_seed()
    if used_seed is not None:
        print(f"[SEED] Using fixed seed from env: {used_seed}")
    
    # 1. Setup Data & Engine
    base_dir = os.path.dirname(os.path.abspath(__file__))
    engine, tournament, _ = build_world_and_tournament(base_dir, require_tactics=False)
    
    # 2. Setup Sentiment Tracking
    sentiment_history = []
    match_step = 0
    
    # Custom match wrapper to track sentiment
    original_play_match = tournament.play_match
    def tracked_play_match(t1, t2, stage_name, llm, is_knockout=True):
        nonlocal match_step
        result = original_play_match(t1, t2, stage_name, llm, is_knockout)
        match_step += 1
        # Track sentiment for key teams
        for team in ["Argentina", "Brazil", "France", "England", "Japan"]:
            if team in engine.agents:
                sentiment_history.append({
                    "step": match_step,
                    "team": team,
                    "score": float(engine.agents[team].morale)  # hidden state morale already in [-1, 1]
                })
        return result
                
    tournament.play_match = tracked_play_match
    
    # 3. RUN FULL TOURNAMENT
    print("\n[LIVE] Simulating tournament and monitoring sentiment...")
    tournament.run_full_tournament()
    
    # 4. GENERATE VIZ
    viz_path = os.path.join(base_dir, "outputs", "visualizations", "world_cup_sentiment_pulse.png")
    os.makedirs(os.path.dirname(viz_path), exist_ok=True)
    plot_sentiment_pulse(sentiment_history, viz_path)
    
    # 5. COUNTERFACTUAL INFERENCE
    print("\n" + "="*60)
    print("COUNTERFACTUAL EXPERIMENT: REVERSE INFERENCE")
    print("="*60)
    simulate_counterfactual_media("Argentina", "failed to pass the group stage (eliminated by Jordan)")
    simulate_counterfactual_media("Japan", "reached the Semi-Finals for the first time")

if __name__ == "__main__":
    main()
