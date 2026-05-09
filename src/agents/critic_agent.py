import os
from .base_agent import BaseAgent

class CriticAgent(BaseAgent):
    def __init__(self):
        prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'critic_prompt.txt')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            template = f.read()
        super().__init__(template)
        
    def analyze(self, agent_outputs: str) -> str:
        return self.generate(agent_outputs=agent_outputs)
