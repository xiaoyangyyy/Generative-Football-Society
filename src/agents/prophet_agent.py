import json
import os
from .base_agent import BaseAgent

class ProphetAgent(BaseAgent):
    def __init__(self):
        prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'prophet_prompt.txt')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            template = f.read()
        super().__init__(template)
        
    def analyze(self, team_name: str, final_fate: str, journey: list) -> str:
        journey_json = json.dumps(journey, indent=2, ensure_ascii=False)
        return self.generate(team_name=team_name, final_fate=final_fate, journey_json=journey_json)
