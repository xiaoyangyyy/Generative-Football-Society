import sys
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

from src.simulation.world_cup_runner import build_world_and_tournament
from src.simulation.random_control import set_global_seed
from src.config import API_KEY, BASE_URL, MODEL_NAME


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
    import argparse

    parser = argparse.ArgumentParser(description="Run full World Cup 2026 simulation")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from data/persistence/tournament_checkpoint.json",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "outputs"), exist_ok=True)
    _log_path = os.path.join(base_dir, "outputs", "full_run_latest.log")
    _log_mode = "a" if args.resume and os.path.isfile(_log_path) else "w"
    _log_fp = open(_log_path, _log_mode, encoding="utf-8")
    sys.stdout = _StreamTee(sys.stdout, _log_fp)
    sys.stderr = _StreamTee(sys.stderr, _log_fp)

    if _log_mode == "a":
        print("\n" + "=" * 60)
        print("=== WORLD CUP 2026: RESUME FROM CHECKPOINT ===")
        print("=" * 60)
    print("=== WORLD CUP 2026: FULL API SIMULATION ===")
    print(f"[LLM] model={MODEL_NAME} base_url={BASE_URL}")
    print(f"[LLM] api_key={'set' if API_KEY and 'your_' not in API_KEY else 'MISSING'}")
    from src.match_engine.ball_path_logger import (
        ball_log_wm_snapshot_enabled,
        ball_path_log_enabled,
        ball_path_log_txt_enabled,
    )
    from src.simulation.match_pipeline import micro_layer_enabled, micro_physics_score_enabled
    from src.simulation.score_path import resolve_score_path_mode, score_path_label

    score_path = resolve_score_path_mode()
    print(
        "[FLAGS] MATCH_MICRO=%s MATCH_COGNITIVE=%s score_path=%s BALL_LOG=%s "
        "BALL_LOG_TXT=%s WM_SNAPSHOT=%s SAVE_CARRYOVER=%s resume=%s"
        % (
            os.environ.get("MATCH_MICRO", "0"),
            os.environ.get("MATCH_COGNITIVE", "0"),
            score_path_label(score_path),
            "1" if ball_path_log_enabled() else "0",
            "1" if ball_path_log_txt_enabled() else "0",
            "1" if ball_log_wm_snapshot_enabled() else "0",
            os.environ.get("SAVE_CARRYOVER", "1"),
            args.resume,
        )
    )
    if score_path.value == "micro_replay":
        print("[FLAGS] MATCH_MICRO_SCORE=0 → macro score + micro replay (not recommended).")
    if micro_layer_enabled():
        from src.match_engine.micro_config import MicroMatchConfig

        mc = MicroMatchConfig()
        print(
            "[SYSTEMS] spatial=%s passing=%s phase3_shots=%s wall_pass=%s | "
            "locker_room=on injuries=on carryover=%s | calibration=v4"
            % (
                mc.enable_spatial,
                mc.enable_passing,
                mc.enable_phase3,
                mc.enable_wall_pass,
                os.environ.get("SAVE_CARRYOVER", "1"),
            )
        )
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
    tournament.run_full_tournament(resume=args.resume)
    
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
