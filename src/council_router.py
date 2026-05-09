from src.agents.team_agent import TeamAgent
from src.agents.historian_agent import HistorianAgent
from src.agents.sociologist_agent import SociologistAgent
from src.agents.critic_agent import CriticAgent
from src.memory_engine.probability_engine import calculate_match_probabilities

def run_council(team_a: str, team_b: str, memory_a: dict, memory_b: dict, stats_a, stats_b):
    print(f"\n[Council] Initiating sandbox for {team_a} vs {team_b}...")
    
    probs = calculate_match_probabilities(stats_a, stats_b)
    
    print(f"[Council] Awakening {team_a} Agent...")
    agent_a = TeamAgent()
    speech_a = agent_a.analyze(team_a, memory_a)
    
    print(f"[Council] Awakening {team_b} Agent...")
    agent_b = TeamAgent()
    speech_b = agent_b.analyze(team_b, memory_b)
    
    print(f"[Council] Awakening Historian Agent for {team_a}...")
    historian = HistorianAgent()
    hist_a = historian.analyze(memory_a)
    print(f"[Council] Awakening Historian Agent for {team_b}...")
    hist_b = historian.analyze(memory_b)
    
    print("[Council] Awakening Sociologist Agent...")
    socio = SociologistAgent()
    socio_a = socio.analyze(memory_a)
    socio_b = socio.analyze(memory_b)
    
    combined = f"""
    --- {team_a} Agent ---
    {speech_a}
    
    --- {team_b} Agent ---
    {speech_b}
    
    --- Historian ({team_a}) ---
    {hist_a}
    
    --- Sociologist ({team_a}) ---
    {socio_a}
    """
    
    print("[Council] Awakening Critic Agent...")
    critic = CriticAgent()
    critic_review = critic.analyze(combined)
    
    report = f"""# 2026 World Cup Simulation Council: {team_a} vs {team_b}

## 1. Data Judge: Sociological Probability Engine
基于底层 `Final Status Score` 演算出的冷酷客观数学概率：
*   **{team_a} (Score: {probs['score_a']})** 获胜概率: **{probs['p_win_a']}%**
*   **平局概率**: **{probs['p_draw']}%**
*   **{team_b} (Score: {probs['score_b']})** 获胜概率: **{probs['p_win_b']}%**

## 2. Clash of Identities (第一人称阵前宣战)
### ⚔️ {team_a} 的战前演说
{speech_a}

### ⚔️ {team_b} 的战前演说
{speech_b}

## 3. The Historian's Scroll (历史学家的时代定场诗)
### 📜 {team_a} 的历史纪传体
{hist_a}

### 📜 {team_b} 的历史纪传体
{hist_b}

## 4. Sociological Verdict (社会学阶层审判)
### ⚖️ 对 {team_a} 的阶层判词
{socio_a}

### ⚖️ 对 {team_b} 的阶层判词
{socio_b}

## 5. 事实最高法庭审查 (Critic's Claim Ledger) 🚨
{critic_review}
"""
    return report
