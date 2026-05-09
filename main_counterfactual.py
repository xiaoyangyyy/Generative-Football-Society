import os
import argparse
from src.data_engine.loader import load_data
from src.data_engine.cleaner import clean_results
from src.data_engine.identity_normalizer import normalize_identities
from src.data_engine.timeline_brancher import branch_timeline
from src.memory_engine.status_score import compute_team_status
from src.memory_engine.global_exposure_gate import compute_exposure_gate
from src.agents.counterfactual_agent import CounterfactualAgent
from src.agents.critic_agent import CriticAgent

def main():
    parser = argparse.ArgumentParser(description="Football Society Agents 2.1 - Counterfactual Sandbox")
    parser.add_argument("date", type=str, help="Date of the match (YYYY-MM-DD)")
    parser.add_argument("team_a", type=str, help="Target Team to observe")
    parser.add_argument("team_b", type=str, help="Opponent in that match")
    parser.add_argument("new_result", type=str, choices=['win', 'draw', 'loss'], help="New result for Team A")
    args = parser.parse_args()
    
    date = args.date
    date_dt = None
    try:
        import pandas as pd
        date_dt = pd.to_datetime(date)
    except Exception:
        date_dt = date
    team_a = args.team_a
    team_b = args.team_b
    new_result = args.new_result
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    
    print("Loading Original Timeline...")
    data = load_data(raw_dir)
    cleaned_df = clean_results(data['results'])
    normalized_df = normalize_identities(cleaned_df, data['former_names'])
    
    mask = (normalized_df['date'] == date_dt) & \
           (((normalized_df['home_team'] == team_a) & (normalized_df['away_team'] == team_b)) | \
            ((normalized_df['home_team'] == team_b) & (normalized_df['away_team'] == team_a)))
    matches = normalized_df[mask]
    if len(matches) == 0:
        print(f"Match not found on {date}!")
        return
    
    row = matches.iloc[0]
    is_home = row['home_team'] == team_a
    if row['home_score'] > row['away_score']:
        orig_res = "win" if is_home else "loss"
    elif row['home_score'] < row['away_score']:
        orig_res = "loss" if is_home else "win"
    else:
        orig_res = "draw"
        
    print(f"Original Match Found: {team_a} result was {orig_res.upper()}.")
    
    print("Computing Original Universe Status Scores...")
    base_stats, base_matches = compute_team_status(normalized_df, data['shootouts'], current_year=2026)
    base_stats = compute_exposure_gate(base_stats, base_matches)
    
    if team_a not in base_stats.index:
        print("Team A not found in computed stats.")
        return
    baseline_scores = base_stats.loc[team_a][['historical_base', 'modern_power', 'final_status_score', 'c5_pressure']].to_dict()
    
    print(f"Branching Timeline (Forcing {team_a} {new_result.upper()} on {date})...")
    branched_df = branch_timeline(normalized_df, date, team_a, team_b, new_result)
    
    print("Computing Parallel Universe Status Scores...")
    branch_stats, branch_matches = compute_team_status(branched_df, data['shootouts'], current_year=2026)
    branch_stats = compute_exposure_gate(branch_stats, branch_matches)
    branched_scores = branch_stats.loc[team_a][['historical_base', 'modern_power', 'final_status_score', 'c5_pressure']].to_dict()
    
    print("\n[Sandbox] Delta Observed:")
    for k in baseline_scores:
        print(f" - {k}: {baseline_scores[k]:.1f} -> {branched_scores[k]:.1f}")
        
    print("\n[Sandbox] Awakening Counterfactual Agent...")
    cf_agent = CounterfactualAgent()
    epic = cf_agent.analyze(team_a, date, team_b, orig_res, new_result, baseline_scores, branched_scores)
    
    print("[Sandbox] Awakening Critic Agent...")
    critic = CriticAgent()
    critic_review = critic.analyze(epic)
    
    report = f"""# Counterfactual Sandbox: The Butterfly Effect
**Observed Subject:** {team_a}
**Temporal Anchor:** {date} vs {team_b}
**Intervention:** Altered result from **{orig_res.upper()}** to **{new_result.upper()}**

## 1. Quantitative Delta (Data Judge)
| Metric | Original Universe | Parallel Universe |
|--------|-------------------|-------------------|
| Historical Base | {baseline_scores['historical_base']:.1f} | {branched_scores['historical_base']:.1f} |
| Modern Power | {baseline_scores['modern_power']:.1f} | {branched_scores['modern_power']:.1f} |
| Final Status Score | {baseline_scores['final_status_score']:.1f} | {branched_scores['final_status_score']:.1f} |
| Pressure (C5) | {baseline_scores['c5_pressure']:.1f} | {branched_scores['c5_pressure']:.1f} |

## 2. Observer's Log (平行宇宙观察报告)
{epic}

## 3. 事实审查 (Critic's Claim Ledger) 🚨
{critic_review}
"""

    out_dir = os.path.join(base_dir, 'outputs', 'counterfactuals')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{team_a.replace(' ', '_')}_{date}_Branch.md")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
        
    print(f"\n[Success] Counterfactual Report saved to {out_path}")

if __name__ == "__main__":
    main()
