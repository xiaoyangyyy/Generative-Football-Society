import pandas as pd
import numpy as np
from .tournament_weights import get_tournament_weight
from .time_decay import calculate_time_decay

def compute_team_status(df: pd.DataFrame, shootouts_df: pd.DataFrame, current_year: int = 2026) -> pd.DataFrame:
    """
    Computes Status Score for each team.
    """
    df = df.copy()
    
    # Unpivot to team-match level
    home_df = df[['date', 'home_team', 'away_team', 'home_result', 'goal_diff_home', 'tournament', 'is_draw', 'winner_90min']].copy()
    home_df.columns = ['date', 'team', 'opponent', 'result', 'goal_diff', 'tournament', 'is_draw', 'winner_90min']
    home_df['is_home'] = True
    
    away_df = df[['date', 'away_team', 'home_team', 'away_result', 'goal_diff_home', 'tournament', 'is_draw', 'winner_90min']].copy()
    away_df['goal_diff_home'] = -away_df['goal_diff_home']
    away_df.columns = ['date', 'team', 'opponent', 'result', 'goal_diff', 'tournament', 'is_draw', 'winner_90min']
    away_df['is_home'] = False
    
    team_matches = pd.concat([home_df, away_df], ignore_index=True)
    
    team_matches['years_ago'] = current_year - team_matches['date'].dt.year
    team_matches['t_weight'] = team_matches['tournament'].apply(get_tournament_weight)
    team_matches['time_decay'] = calculate_time_decay(team_matches['years_ago'], tournament_weights=team_matches['t_weight'])
    
    result_points = {'win': 3, 'draw': 1, 'loss': 0}
    team_matches['points'] = team_matches['result'].map(result_points)
    
    if shootouts_df is not None and not shootouts_df.empty:
        shootouts_df['date'] = pd.to_datetime(shootouts_df['date'])
        # merge for home
        team_matches = team_matches.merge(
            shootouts_df[['date', 'home_team', 'winner']], 
            how='left', 
            left_on=['date', 'team'], 
            right_on=['date', 'home_team']
        )
        team_matches.rename(columns={'winner': 'winner_h'}, inplace=True)
        # merge for away
        team_matches = team_matches.merge(
            shootouts_df[['date', 'away_team', 'winner']], 
            how='left', 
            left_on=['date', 'team'], 
            right_on=['date', 'away_team']
        )
        team_matches.rename(columns={'winner': 'winner_a'}, inplace=True)
        
        team_matches['shootout_winner'] = team_matches['winner_h'].combine_first(team_matches['winner_a'])
        team_matches.drop(columns=['home_team_x', 'home_team_y', 'away_team', 'winner_h', 'winner_a'], inplace=True, errors='ignore')
    else:
        team_matches['shootout_winner'] = np.nan
        
    team_matches['shootout_win'] = team_matches['shootout_winner'] == team_matches['team']
    
    team_matches['w_points'] = team_matches['points'] * team_matches['t_weight'] * team_matches['time_decay']
    team_matches['w_max_points'] = 3 * team_matches['t_weight'] * team_matches['time_decay']
    
    team_matches['gd_score'] = np.tanh(team_matches['goal_diff'] / 3.0)
    team_matches['w_gd_score'] = team_matches['gd_score'] * team_matches['t_weight'] * team_matches['time_decay']
    team_matches['w_gd_base'] = team_matches['t_weight'] * team_matches['time_decay']
    
    major_tournaments = ['FIFA World Cup', 'UEFA Euro', 'Copa América']
    team_matches['is_major'] = team_matches['tournament'].isin(major_tournaments)
    team_matches['w_major_exp'] = team_matches['is_major'].astype(int) * team_matches['t_weight'] * team_matches['time_decay']
    
    is_pressure_event = (team_matches['t_weight'] >= 0.7) & (
        team_matches['shootout_win'] | 
        ((team_matches['result'] == 'win') & (team_matches['goal_diff'] >= 2))
    )
    team_matches['w_pressure'] = is_pressure_event.astype(int) * team_matches['t_weight'] * team_matches['time_decay']
    
    grouped = team_matches.groupby('team')
    stats = pd.DataFrame()
    stats['total_matches'] = grouped.size()
    
    stats['c1_win_rate'] = (grouped['w_points'].sum() / grouped['w_max_points'].sum().clip(lower=1e-9)) * 100.0
    stats['c2_gd'] = ((grouped['w_gd_score'].sum() / grouped['w_gd_base'].sum().clip(lower=1e-9)) + 1.0) / 2.0 * 100.0
    
    w_major_exp_sum = grouped['w_major_exp'].sum()
    stats['c3_major_exp'] = w_major_exp_sum / max(1.0, w_major_exp_sum.max()) * 100.0
    
    w_pressure_sum = grouped['w_pressure'].sum()
    stats['c5_pressure'] = w_pressure_sum / max(1.0, w_pressure_sum.max()) * 100.0
    
    # To prevent regional teams from inflating each other, we determine the True Top 20
    # based strictly on performances in the absolute highest tier global tournaments.
    global_matches = team_matches[team_matches['tournament'].isin(['FIFA World Cup', 'UEFA Euro', 'Copa América'])]
    g_grouped = global_matches.groupby('team')
    stats['global_win_rate'] = (g_grouped['w_points'].sum() / g_grouped['w_max_points'].sum().clip(lower=1e-9)).fillna(0) * 100.0
    stats['global_exp'] = g_grouped['w_major_exp'].sum()
    stats['global_exp'] = stats['global_exp'] / max(1.0, stats['global_exp'].max()) * 100.0
    
    stats['true_global_score'] = 0.5 * stats['global_win_rate'] + 0.5 * stats['global_exp'].fillna(0)
    top_20_teams = stats['true_global_score'].nlargest(20).index.tolist()
    team_matches['is_top_opponent'] = team_matches['opponent'].isin(top_20_teams)
    
    top_opp_matches = team_matches[team_matches['is_top_opponent']]
    top_opp_grouped = top_opp_matches.groupby('team')
    stats['c4_strong_opp'] = (top_opp_grouped['w_points'].sum() / top_opp_grouped['w_max_points'].sum().clip(lower=1e-9)).fillna(0) * 100.0
    
    stats['raw_score'] = (
        0.30 * stats['c1_win_rate'] +
        0.20 * stats['c2_gd'] +
        0.25 * stats['c3_major_exp'] +
        0.15 * stats['c4_strong_opp'] +
        0.10 * stats['c5_pressure']
    )
    
    modern_matches = team_matches[team_matches['years_ago'] <= 15]
    mod_grouped = modern_matches.groupby('team')
    
    stats['modern_win_rate'] = (mod_grouped['w_points'].sum() / mod_grouped['w_max_points'].sum().clip(lower=1e-9)) * 100
    stats['modern_gd'] = ((mod_grouped['w_gd_score'].sum() / mod_grouped['w_gd_base'].sum().clip(lower=1e-9)) + 1) / 2 * 100
    
    mod_exp_max = max(1.0, mod_grouped['w_major_exp'].sum().max())
    stats['modern_exp'] = (mod_grouped['w_major_exp'].sum() / mod_exp_max) * 100
    
    mod_press_max = max(1.0, mod_grouped['w_pressure'].sum().max())
    stats['modern_pressure'] = (mod_grouped['w_pressure'].sum() / mod_press_max) * 100
    
    mod_top_grouped = modern_matches[modern_matches['is_top_opponent']].groupby('team')
    stats['modern_strong_opp'] = (mod_top_grouped['w_points'].sum() / mod_top_grouped['w_max_points'].sum().clip(lower=1e-9)) * 100
    
    stats['modern_power'] = (
        0.30 * stats['modern_win_rate'].fillna(0) +
        0.20 * stats['modern_gd'].fillna(0) +
        0.25 * stats['modern_exp'].fillna(0) +
        0.15 * stats['modern_strong_opp'].fillna(0) +
        0.10 * stats['modern_pressure'].fillna(0)
    )
    
    stats['historical_base'] = stats['raw_score']
    stats['final_status_score'] = 0.30 * stats['historical_base'] + 0.70 * stats['modern_power'].fillna(0)
    
    max_final = stats['final_status_score'].max()
    stats['final_status_score'] = stats['final_status_score'] / max_final * 100.0
    stats['historical_base'] = stats['historical_base'] / stats['historical_base'].max() * 100.0
    stats['modern_power'] = stats['modern_power'] / stats['modern_power'].max() * 100.0
    
    return stats, team_matches
