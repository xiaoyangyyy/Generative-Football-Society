import os
import argparse
from src.data_engine.loader import load_data
from src.data_engine.cleaner import clean_results
from src.data_engine.identity_normalizer import normalize_identities
from src.memory_engine.status_score import compute_team_status
from src.memory_engine.global_exposure_gate import compute_exposure_gate
from src.memory_engine.sample_confidence import compute_sample_confidence
from src.memory_engine.team_memory_builder import build_team_memory
from src.router import generate_team_report
from src.visualization.team_charts import generate_all_team_charts

def main():
    parser = argparse.ArgumentParser(description="Football Society Agents 2.1 MVP-2")
    parser.add_argument("team", type=str, help="Name of the national team to analyze")
    args = parser.parse_args()
    
    team_name = args.team
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    
    print(f"Loading data to analyze {team_name}...")
    data = load_data(raw_dir)
    cleaned_df = clean_results(data['results'])
    normalized_df = normalize_identities(cleaned_df, data['former_names'])
    
    print("Computing metrics...")
    stats, team_matches = compute_team_status(normalized_df, data['shootouts'], current_year=2026)
    stats = compute_exposure_gate(stats, team_matches)
    stats = compute_sample_confidence(stats)
    
    try:
        print("Building structured memory...")
        memory = build_team_memory(team_name, stats, team_matches, data.get('goalscorers'))
    except ValueError as e:
        print(e)
        return
        
    report = generate_team_report(team_name, memory)
    
    print("Generating Master Dashboard...")
    dashboard_path, _ = generate_all_team_charts(team_name, stats, team_matches)
    
    # Append visualization to report
    rel_dashboard = f"../visualizations/{team_name.replace(' ', '_')}/{os.path.basename(dashboard_path)}"
    
    visual_section = f"""
## 5. Master Dashboard (全景数据大屏)
![Master Dashboard]({rel_dashboard})
"""
    report += visual_section
    
    out_dir = os.path.join(base_dir, 'outputs', 'team_reports')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{team_name.replace(' ', '_')}_report.md")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
        
    print(f"\nDone! Report saved to {out_path}")

if __name__ == "__main__":
    main()
