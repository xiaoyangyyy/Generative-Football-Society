import json
from src.agents.team_agent import TeamAgent
from src.agents.prophet_agent import ProphetAgent
from src.agents.sociologist_agent import SociologistAgent
from src.agents.critic_agent import CriticAgent
from src.memory_engine.tournament_simulator import simulate_journey

def run_journey(team_name: str, memory: dict, stats):
    print(f"\n[Journey] Initiating 2026 World Cup Simulation for {team_name}...")
    
    print("[Journey] Rolling the dice for the tournament path...")
    journey_log, final_fate = simulate_journey(team_name, stats)
    
    print(f"[Journey] Awakening {team_name} Agent for pre-tournament declaration...")
    team_agent = TeamAgent()
    speech = team_agent.analyze(team_name, memory)
    
    print(f"[Journey] Awakening Prophet Agent to write the epic...")
    prophet = ProphetAgent()
    epic = prophet.analyze(team_name, final_fate, journey_log)
    
    print("[Journey] Awakening Sociologist Agent for post-tournament verdict...")
    memory['2026_simulation_outcome'] = final_fate
    socio_agent = SociologistAgent()
    socio_analysis = socio_agent.analyze(memory)
    
    combined = f"""
    --- {team_name} Agent Pre-Tournament ---
    {speech}
    
    --- Prophet Epic ---
    {epic}
    
    --- Sociologist Post-Tournament ---
    {socio_analysis}
    """
    
    print("[Journey] Awakening Critic Agent...")
    critic_agent = CriticAgent()
    critic_review = critic_agent.analyze(combined)
    
    journey_table = "\n".join([f"- **{step['stage']}** vs {step['opponent']} -> **{step['result'].upper()}** ({step['details']})" for step in journey_log])
    
    report = f"""# 2026 World Cup Journey Simulation: {team_name}

## 1. 命运之轮 (Simulated Journey Log)
底层蒙特卡洛掷骰子引擎算出的客观赛程路线（{final_fate}）：
{journey_table}

## 2. 出征宣言 (Clash of Identities)
### ⚔️ {team_name} 的战前演说
{speech}

## 3. 远征史诗 (The Prophet's Tale)
### 📜 预言家的纪传体
{epic}

## 4. 阶层宣判 (Sociological Verdict)
### ⚖️ 赛后的社会学评估
{socio_analysis}

## 5. 事实最高法庭审查 (Critic's Claim Ledger) 🚨
{critic_review}
"""
    return report
