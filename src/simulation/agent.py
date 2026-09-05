import numpy as np

from src.simulation.agent_condition import AgentConditionMixin
from src.simulation.agent_governance import AgentGovernanceMixin
from src.simulation.agent_memory import AgentMemoryMixin
from src.simulation.agent_dynamics import AgentMatchDynamicsMixin
from src.simulation.agent_initialization import AgentInitializationMixin
from src.simulation.agent_psychology import AgentPsychologyMixin
from src.simulation.agent_reflection import AgentReflectionMixin
from src.simulation.agent_social import SocialAgentMixin
from src.simulation.agent_tactics import AgentTacticsMixin
from src.simulation.random_control import named_py_rng


class SocietyAgent(
    AgentInitializationMixin,
    AgentPsychologyMixin,
    AgentGovernanceMixin,
    AgentTacticsMixin,
    AgentReflectionMixin,
    AgentConditionMixin,
    AgentMatchDynamicsMixin,
    AgentMemoryMixin,
    SocialAgentMixin,
):
    @staticmethod
    def _finite(value, default=0.0):
        try:
            v = float(value)
        except Exception:
            return float(default)
        if not np.isfinite(v):
            return float(default)
        return v

    def __init__(self, name, stats, tactical_info=None, initialization_rng=None, random_root_seed=42):
        self.random_root_seed = int(random_root_seed)
        self._initialization_rng = initialization_rng or named_py_rng(
            self.random_root_seed, "initialization", name
        )
        self._initialize_identity(name, stats)
        self._initialize_roles_and_strategy(stats, tactical_info)
        self._initialize_runtime_state()
        self._initialize_affective_state()
        del self._initialization_rng

    def _infer_region(self):
        europe = {
            "England",
            "Germany",
            "France",
            "Spain",
            "Portugal",
            "Netherlands",
            "Belgium",
            "Croatia",
            "Austria",
            "Switzerland",
            "Sweden",
            "Scotland",
            "Turkey",
            "Bosnia and Herzegovina",
            "Norway",
            "Czech Republic",
        }
        south_america = {
            "Argentina",
            "Brazil",
            "Uruguay",
            "Colombia",
            "Paraguay",
            "Chile",
            "Ecuador",
            "Peru",
            "Venezuela",
            "Bolivia",
        }
        north_america = {
            "United States",
            "Mexico",
            "Canada",
            "Panama",
            "Qatar",
            "Curaçao",
        }
        africa = {
            "Morocco",
            "Egypt",
            "Ivory Coast",
            "Senegal",
            "Ghana",
            "Algeria",
            "South Africa",
            "Tunisia",
            "DR Congo",
            "Cape Verde",
        }
        asia = {
            "Japan",
            "South Korea",
            "Saudi Arabia",
            "Iraq",
            "Iran",
            "Jordan",
            "Uzbekistan",
            "Australia",
        }
        oceania = {"New Zealand", "Haiti"}

        if self.team_name in europe:
            return "Europe"
        if self.team_name in south_america:
            return "South America"
        if self.team_name in north_america:
            return "North/Central America"
        if self.team_name in africa:
            return "Africa"
        if self.team_name in asia:
            return "Asia"
        if self.team_name in oceania:
            return "Oceania"
        return "Global"

    def _infer_style_archetype(self):
        from src.match_engine.tactical_catalog import infer_archetype_from_text

        return infer_archetype_from_text(self.style_desc, self.formation)
