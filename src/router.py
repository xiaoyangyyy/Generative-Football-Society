from src.agents.team_agent import TeamAgent
from src.agents.historian_agent import HistorianAgent
from src.agents.sociologist_agent import SociologistAgent

def generate_team_report(team_name, memory):
    """
    Unified router for generating comprehensive team identity reports.
    """
    print(f"[Router] Generating comprehensive report for {team_name}...")
    
    agent = TeamAgent()
    speech = agent.analyze(team_name, memory)
    
    historian = HistorianAgent()
    history = historian.analyze(memory)
    
    sociologist = SociologistAgent()
    social = sociologist.analyze(memory)
    
    report = f"""# Society Intelligence Report: {team_name}
    
## 1. Identity Declaration (第一人称自述)
{speech}

## 2. Historical Context (历史学家的纪传)
{history}

## 3. Sociological Standing (社会学判词)
{social}
"""
    return report
