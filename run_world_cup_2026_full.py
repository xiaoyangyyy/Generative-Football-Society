import sys
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.simulation.world_cup_runner import build_world_and_tournament
from src.simulation.random_control import set_global_seed


class _StreamTee:
    """Mirror stdout/stderr to outputs/full_run_latest.log for full archival."""

    def __init__(self, primary, secondary):
        self.primary = primary
        self.secondary = secondary

    def write(self, s):
        self.primary.write(s)
        try:
            self.secondary.write(s)
            self.secondary.flush()
        except Exception:
            pass

    def flush(self):
        self.primary.flush()
        try:
            self.secondary.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self.primary, name)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "outputs"), exist_ok=True)
    _log_path = os.path.join(base_dir, "outputs", "full_run_latest.log")
    _log_fp = open(_log_path, "w", encoding="utf-8")
    sys.stdout = _StreamTee(sys.stdout, _log_fp)
    sys.stderr = _StreamTee(sys.stderr, _log_fp)

    print("=== 🏆 WORLD CUP 2026: THE ULTIMATE FULL SIMULATION ===")
    used_seed = set_global_seed()
    if used_seed is not None:
        print(f"[SEED] Using fixed seed from env: {used_seed}")
    
    # Setup
    engine, tournament, _ = build_world_and_tournament(base_dir, require_tactics=False)
    
    # Seed historical memories before kickoff
    print("\n[SYSTEM] Seeding historical memories into agents...")
    for name, agent in engine.agents.items():
        if agent.tier == 'Core Power':
            agent.add_memory(f"Historical Heritage: We are the pillars of football history. Our status score is {agent.status_score:.1f}.", importance=10)
        elif agent.status_score > 60:
            agent.add_memory(f"Ambition: We are rising. World Cup 2026 is our stage to challenge the old order.", importance=8)

    # RUN THE ENTIRE TOURNAMENT
    tournament.run_full_tournament()
    
    print("\n" + "="*60)
    print("POST-TOURNAMENT STATUS INDEX (TOP 10)")
    print("="*60)
    if getattr(tournament, "final_result", None):
        champion = tournament.final_result.get("champion")
        runner_up = tournament.final_result.get("runner_up")
        if champion:
            print(f"TOURNAMENT CHAMPION: {champion}")
        if runner_up:
            print(f"RUNNER-UP: {runner_up}")
        print("-"*60)
    
    # Final Status Rankings
    final_leaders = sorted(engine.agents.items(), key=lambda x: x[1].status_score, reverse=True)[:10]
    for i, (name, agent) in enumerate(final_leaders):
        print(f"{i+1}. {name:20} Status: {agent.status_score:.1f} | Morale(long-term): {agent.morale:.1f}")

    # Top Trending Social Posts from the Final/Semis
    print("\n[TOURNAMENT ARCHIVE: TOP SOCIAL THREADS]")
    for post in engine.feed.posts[-8:]:
        print(f"[{post['author']}]: {post['content']}")

    print(f"\n[LOG] Full transcript also mirrored to: {_log_path}")
    _log_fp.flush()

if __name__ == "__main__":
    main()
