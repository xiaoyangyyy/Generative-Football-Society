import json
import os
from .base_agent import BaseAgent

class HistorianAgent(BaseAgent):
    def __init__(self):
        prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'historian_prompt.txt')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            template = f.read()
        super().__init__(template)
        
    def analyze(self, memory: dict) -> str:
        memory_json = json.dumps(memory.get('era_slices', {}), indent=2, ensure_ascii=False)
        return self.generate(memory_json=memory_json)
