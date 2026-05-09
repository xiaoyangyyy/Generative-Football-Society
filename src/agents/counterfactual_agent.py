import os
import json
from .base_agent import BaseAgent

class CounterfactualAgent(BaseAgent):
    def __init__(self):
        prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'counterfactual_prompt.txt')
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                template = f.read()
        else:
            template = "Simulate a 'What-If' scenario for {team_name}. History: {history_json}. New Variable: {variable}"
        super().__init__(template)
        
    def analyze(self, *args, **kwargs) -> str:
        # Backward-compatible API:
        # 1) analyze(team_name, history_dict, variable)
        # 2) analyze(team_name, date, opponent, original_result, new_result, baseline_scores, branched_scores)
        if len(args) == 3 and not kwargs:
            team_name, history, variable = args
            history_json = json.dumps(history, indent=2, ensure_ascii=False)
            return self.generate(team_name=team_name, history_json=history_json, variable=variable)

        if len(args) == 7 and not kwargs:
            team_name, date, opponent, original_result, new_result, baseline_scores, branched_scores = args
            return self.generate(
                team_name=team_name,
                date=date,
                opponent=opponent,
                original_result=original_result,
                new_result=new_result,
                baseline_scores=json.dumps(baseline_scores, indent=2, ensure_ascii=False),
                branched_scores=json.dumps(branched_scores, indent=2, ensure_ascii=False),
            )

        raise ValueError("Unsupported analyze() signature for CounterfactualAgent.")
