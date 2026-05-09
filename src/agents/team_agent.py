import os
import json
from .base_agent import BaseAgent

class TeamAgent(BaseAgent):
    def __init__(self):
        # Fallback if prompt file is missing
        prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'team_prompt.txt')
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                template = f.read()
        else:
            template = "Analyze the football team {team_name} with the following identity: {memory_json}"
        super().__init__(template)
        
    def analyze(self, team_name: str, memory: dict) -> str:
        memory_json = json.dumps(memory, indent=2, ensure_ascii=False)
        return self.generate(team_name=team_name, memory_json=memory_json)
