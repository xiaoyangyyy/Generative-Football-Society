import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

def set_master_style():
    """Sets a premium, academic-style dashboard theme."""
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = '#FCFCFC'
    plt.rcParams['axes.edgecolor'] = '#D1D1D1'
    plt.rcParams['grid.color'] = '#EBEBEB'
    plt.rcParams['axes.labelcolor'] = '#2C3E50'
    plt.rcParams['xtick.color'] = '#2C3E50'
    plt.rcParams['ytick.color'] = '#2C3E50'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 9

def generate_all_team_charts(team_name, stats, team_matches):
    """Generates a high-density Master Dashboard for a team."""
    set_master_style()
    
    team_data = stats.loc[team_name]
    matches = team_matches[team_matches['team'] == team_name].copy()
    matches['year'] = matches['date'].dt.year
    
    out_dir = os.path.join('outputs', 'visualizations', team_name.replace(' ', '_'))
    os.makedirs(out_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 4, 3], width_ratios=[1, 1.5])
    
    # ---------------------------------------------------------
    # 1. TOP HEADER: Status Summary
    # ---------------------------------------------------------
    ax_head = fig.add_subplot(gs[0, :])
    ax_head.axis('off')
    status_score = team_data['final_status_score']
    tier = team_data['tier']
    exposure = team_data['global_exposure_gate']
    
    ax_head.text(0.5, 0.8, f"{team_name.upper()} - SOCIAL STATUS DASHBOARD", 
                 fontsize=24, fontweight='bold', ha='center', color='#2C3E50')
    ax_head.text(0.5, 0.4, f"Final Status Score: {status_score:.1f} | Tier: {tier} | Exposure: {exposure}", 
                 fontsize=14, ha='center', color='#7F8C8D')
    ax_head.axhline(0.2, color='#BDC3C7', linewidth=2)

    # ---------------------------------------------------------
    # 2. MIDDLE LEFT: Metric Radar
    # ---------------------------------------------------------
    ax_radar = fig.add_subplot(gs[1, 0], polar=True)
    labels = ['Win Rate', 'Goal Diff', 'Major Exp', 'Strong Opp', 'Pressure']
    values = [
        team_data['c1_win_rate'], team_data['c2_gd'], team_data['c3_major_exp'],
        team_data['c4_strong_opp'], team_data['c5_pressure']
    ]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax_radar.fill(angles, values, color='#3498DB', alpha=0.3)
    ax_radar.plot(angles, values, color='#2980B9', linewidth=2)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax_radar.set_ylim(0, 100)
    ax_radar.set_title("Metric Decomposition", pad=30, fontsize=14, fontweight='bold')

    # ---------------------------------------------------------
    # 3. MIDDLE RIGHT: Historical Pulse (Yearly)
    # ---------------------------------------------------------
    ax_pulse = fig.add_subplot(gs[1, 1])
    yearly_wr = matches.groupby('year').apply(lambda x: (x['result'] == 'win').mean() * 100)
    # Rolling average for smoothness
    rolling_wr = yearly_wr.rolling(window=10, min_periods=1).mean()
    
    ax_pulse.plot(yearly_wr.index, yearly_wr.values, color='#BDC3C7', alpha=0.4, linewidth=1, label='Yearly raw')
    ax_pulse.plot(rolling_wr.index, rolling_wr.values, color='#E74C3C', linewidth=3, label='10-Year Trend')
    
    # Annotate key peaks (Major Wins)
    glory_years = matches[matches['t_weight'] >= 1.0].groupby('year').size()
    for yr in glory_years.index:
        if yr in rolling_wr:
            ax_pulse.scatter(yr, rolling_wr[yr], color='gold', s=100, zorder=5, edgecolors='black')
            ax_pulse.annotate(f"WC {yr}", (yr, rolling_wr[yr]), xytext=(0, 10), 
                              textcoords='offset points', ha='center', fontsize=8, fontweight='bold')

    ax_pulse.set_ylim(0, 105)
    ax_pulse.set_title("150-Year Historical Pulse & Major Milestones", fontsize=14, fontweight='bold')
    ax_pulse.set_ylabel("Win Rate (%)")
    ax_pulse.grid(True, alpha=0.3)
    ax_pulse.legend()

    # ---------------------------------------------------------
    # 4. BOTTOM LEFT: Rivalry Matrix
    # ---------------------------------------------------------
    ax_rival = fig.add_subplot(gs[2, 0])
    rival_stats = matches.groupby('opponent')['result'].value_counts().unstack().fillna(0)
    # Filter for top 5 by total matches
    top_rivals = rival_stats.sum(axis=1).nlargest(5).index
    rival_plot_data = rival_stats.loc[top_rivals]
    
    # Ensure columns exist
    for col in ['win', 'draw', 'loss']:
        if col not in rival_plot_data.columns:
            rival_plot_data[col] = 0
            
    rival_plot_data = rival_plot_data[['win', 'draw', 'loss']]
    rival_plot_data.plot(kind='barh', stacked=True, ax=ax_rival, 
                         color=['#27AE60', '#F1C40F', '#E74C3C'], alpha=0.8)
    
    ax_rival.set_title("Top 5 Rivalry Matrix", fontsize=14, fontweight='bold')
    ax_rival.set_xlabel("Match Counts")
    ax_rival.set_ylabel("")
    ax_rival.legend(title="Result", loc='lower right')

    # ---------------------------------------------------------
    # 5. BOTTOM RIGHT: Data Table / Glory/Trauma
    # ---------------------------------------------------------
    ax_table = fig.add_subplot(gs[2, 1])
    ax_table.axis('off')
    
    # Quick Summary Table
    table_data = [
        ["Total Matches", len(matches)],
        ["Overall Win Rate", f"{(matches['result']=='win').mean()*100:.1f}%"],
        ["Historical Base", f"{team_data['historical_base']:.1f}"],
        ["Modern Power", f"{team_data['modern_power']:.1f}"],
        ["Global Win Rate", f"{team_data['global_win_rate']:.1f}%"]
    ]
    
    table = ax_table.table(cellText=table_data, colLabels=["Indicator", "Value"], 
                           loc='center', cellLoc='left', colWidths=[0.3, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax_table.set_title("Key Performance Indicators (KPI)", fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    dashboard_path = os.path.join(out_dir, f"{team_name.replace(' ', '_')}_master_dashboard.png")
    plt.savefig(dashboard_path, dpi=300)
    plt.close()
    
    # Compatibility with main_mvp2.py (it expects two paths)
    return dashboard_path, dashboard_path
