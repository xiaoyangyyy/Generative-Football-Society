import matplotlib.pyplot as plt
import pandas as pd

def plot_sentiment_pulse(sentiment_data, output_path):
    """
    Plots the sentiment pulse of teams over the tournament.
    sentiment_data: List of dicts {'step': int, 'team': str, 'score': float}
    """
    df = pd.DataFrame(sentiment_data)
    if df.empty: return
    
    plt.figure(figsize=(12, 6), dpi=150)
    plt.style.use('dark_background')
    
    teams = df['team'].unique()
    colors = plt.cm.get_cmap('viridis', len(teams))
    
    for i, team in enumerate(teams):
        team_df = df[df['team'] == team]
        plt.plot(team_df['step'], team_df['score'], label=team, marker='o', linewidth=2, color=colors(i))
    
    plt.axhline(0, color='white', linestyle='--', alpha=0.3)
    plt.title("World Cup 2026: Global Sentiment Pulse Monitor", fontsize=16, color='#E74C3C', fontweight='bold')
    plt.xlabel("Tournament Progress (Matches Played)", fontsize=12)
    plt.ylabel("Public Sentiment Index (-1 to +1)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(alpha=0.1)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[VIZ] Sentiment Pulse saved to {output_path}")
    plt.close()
