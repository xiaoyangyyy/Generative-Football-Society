from src.simulation.llm_engine import SimulationLLM


class MediaOutlet:
    def __init__(self, name="GLOBAL_FOOTBALL_NEWS"):
        self.name = name

    def publish(self, headline, summary):
        return {
            "outlet": self.name,
            "headline": headline,
            "summary": summary,
        }


def simulate_counterfactual_media(team_name, scenario):
    llm = SimulationLLM()
    system = "You are a football media newsroom editor."
    user = f"""Team: {team_name}
Counterfactual scenario: {scenario}
Write:
1) headline (one line)
2) 3-bullet reaction from mainstream media
3) 3-bullet reaction from social media
"""
    result = llm._call_llm(system, user)
    print(f"[MEDIA LAB] {team_name}")
    print(result)
    return result
