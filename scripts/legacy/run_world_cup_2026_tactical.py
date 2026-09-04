import sys
import os
import io
import argparse

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.simulation.world_cup_runner import build_world_and_tournament
from src.simulation.llm_engine import SimulationLLM
from src.simulation.random_control import set_global_seed

def main():
    parser = argparse.ArgumentParser(description="World Cup 2026 Tactical Engine")
    parser.add_argument("--quick", action="store_true", help="Skip full tournament and open lab immediately")
    parser.add_argument("--no-interactive", action="store_true", help="Do not open interactive lab loop")
    parser.add_argument("--scenario", type=str, default="", help="Run one counterfactual scenario and exit")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed for reproducibility")
    args = parser.parse_args()
    used_seed = set_global_seed(args.seed)

    print("\n" + "="*50)
    print("🚀 WORLD CUP 2026: PURE ENGLISH TACTICAL ENGINE")
    print("="*50)
    if used_seed is not None:
        print(f"[SEED] Using fixed seed: {used_seed}")
    
    # 1. Setup Data + Engine
    base_dir = os.path.dirname(os.path.abspath(__file__))
    engine, tournament, tactical_map = build_world_and_tournament(base_dir, require_tactics=True)
    
    # 2. Load the PURE ENGLISH Matrix
    print("[1/3] Loading Full English Tactical Database...")
    
    # Verification
    print("\n--- ENGLISH TACTICAL GENE CHECK ---")
    for team in ["Brazil", "Germany", "Japan", "Mexico"]:
        if team in tactical_map:
            print(f"  [EN] {team:10} -> {tactical_map[team]['formation']} ({tactical_map[team]['style_desc'][:40]}...)")
    print("----------------------------------\n")

    # 3. RUN SIMULATION
    if args.quick:
        print("[2/3] QUICK MODE: tournament simulation skipped.")
    else:
        print("[2/3] 🏟️ KICKOFF: FULL GLOBAL SIMULATION")
        tournament.run_full_tournament()
    
    # 4. INTERACTIVE LAB
    print("\n" + "!"*60)
    print("WELCOME TO THE WORLD CUP 2026 COUNTERFACTUAL LAB")
    print("!"*60)
    
    winner = tournament.qualified_teams[0] if tournament.qualified_teams else "Unknown"
    context_summary = f"Winner: {winner}. Tournament ended."
    if args.no_interactive and not args.scenario:
        print("[SYSTEM] Non-interactive mode enabled. Exiting after simulation.")
        return

    llm = SimulationLLM()

    if args.scenario:
        print("\n[ANALYZING] Running single scenario in non-interactive mode...")
        inference = llm.generate_free_inference(args.scenario, context_summary)
        print("\n" + "-"*50)
        print(inference)
        print("-"*50)
        return
    
    while True:
        try:
            user_scenario = input("\n[IF...] 请输入您的反向推演假设 (或 'exit' 退出): ")
        except EOFError:
            print("\n[SYSTEM] Input stream closed. Exiting lab.")
            break
        if user_scenario.lower() in ['exit', 'quit', 'e']:
            break
            
        print("\n[ANALYZING] Qwen is inferring the social consequences...")
        inference = llm.generate_free_inference(user_scenario, context_summary)
        print("\n" + "-"*50)
        print(inference)
        print("-"*50)

if __name__ == "__main__":
    main()
