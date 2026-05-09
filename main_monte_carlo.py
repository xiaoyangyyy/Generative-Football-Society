import os
import argparse
from collections import Counter
from src.data_engine.loader import load_data
from src.data_engine.cleaner import clean_results
from src.data_engine.identity_normalizer import normalize_identities
from src.memory_engine.status_score import compute_team_status
from src.memory_engine.global_exposure_gate import compute_exposure_gate
from src.memory_engine.sample_confidence import compute_sample_confidence
from src.memory_engine.full_tournament_simulator import simulate_full_tournament, GROUPS_2026
from src.simulation.random_control import set_global_seed

def main():
    parser = argparse.ArgumentParser(description="Football Society Agents 2.1 - Monte Carlo Mass Simulator")
    parser.add_argument("-n", "--num_sims", type=int, default=10000, help="Number of tournament simulations (default: 10000)")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed for reproducibility")
    args = parser.parse_args()
    
    n = args.num_sims
    used_seed = set_global_seed(args.seed)
    
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
    
    print(f"\nRunning {n:,} Monte Carlo World Cup Simulations (Poisson Engine)...")
    print("This may take a moment...\n")
    if used_seed is not None:
        print(f"Using seed: {used_seed}\n")
    
    champion_counts = Counter()
    
    progress_step = max(1, n // 10)
    for i in range(n):
        if (i + 1) % progress_step == 0:
            pct = ((i + 1) / n) * 100
            print(f"  Progress: {pct:.0f}% ({i+1:,}/{n:,})")
        
        champion = simulate_full_tournament(stats, verbose=False)
        champion_counts[champion] += 1
    
    # Build results table sorted by championship probability
    all_teams = set()
    for teams in GROUPS_2026.values():
        all_teams.update(teams)
    
    results = []
    for team in all_teams:
        wins = champion_counts.get(team, 0)
        prob = (wins / n) * 100
        score = stats.loc[team]['final_status_score'] if team in stats.index else 30.0
        results.append({
            "team": team,
            "wins": wins,
            "prob": prob,
            "score": score
        })
    
    results.sort(key=lambda x: x['prob'], reverse=True)
    
    # Generate report
    report_lines = [
        f"# 2026 World Cup: Monte Carlo Championship Probability Report",
        f"**Simulations Run:** {n:,}",
        f"**Engine:** Poisson Score Simulation + Penalty Shootout\n",
        "## Championship Probability Table (夺冠概率表)\n",
        "| Rank | Team | Status Score | Championships | Probability |",
        "|------|------|-------------|---------------|-------------|"
    ]
    
    for rank, r in enumerate(results, 1):
        bar_len = int(r['prob'] / 2)  # Scale bar to max ~50 chars
        bar = "█" * bar_len if bar_len > 0 else "▏"
        report_lines.append(
            f"| {rank} | **{r['team']}** | {r['score']:.1f} | {r['wins']:,} | {r['prob']:.2f}% {bar} |"
        )
    
    # Summary stats
    top5 = results[:5]
    report_lines.append(f"\n## Key Insights\n")
    report_lines.append(f"- **Most Likely Champion:** {results[0]['team']} ({results[0]['prob']:.2f}%)")
    report_lines.append(f"- **Top 5 Combined Probability:** {sum(r['prob'] for r in top5):.1f}%")
    
    zero_wins = [r for r in results if r['wins'] == 0]
    if zero_wins:
        report_lines.append(f"- **Teams with 0% Championship Chance:** {len(zero_wins)} teams")
    
    report = "\n".join(report_lines)
    
    out_dir = os.path.join(base_dir, 'outputs', 'monte_carlo')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"2026_MonteCarlo_{n}_sims.md")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n{'='*60}")
    print(f"  MONTE CARLO SIMULATION COMPLETE ({n:,} runs)")
    print(f"{'='*60}")
    print(f"\n  Top 10 Championship Probabilities:")
    for i, r in enumerate(results[:10], 1):
        print(f"  {i:2d}. {r['team']:20s}  {r['prob']:6.2f}%  ({r['wins']:,} wins)")
    print(f"\n  Full report saved to: {out_path}")

if __name__ == "__main__":
    main()
