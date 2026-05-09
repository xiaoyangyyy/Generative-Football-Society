from src.simulation.llm_engine import SimulationLLM

class BaseAgent:
    def __init__(self, template):
        self.template = template
        self.llm = SimulationLLM()
        
    def generate(self, **kwargs):
        # Format the template with provided arguments
        user_prompt = self.template.format(**kwargs)
        # Call the LLM (System prompt can be generic for these agents)
        system_prompt = "You are a specialized football society agent. Provide your analysis based on the context."
        return self.llm._call_llm(system_prompt, user_prompt)
