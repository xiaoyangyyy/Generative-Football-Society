"""Compatibility composition for the agent memory subsystems."""

from src.simulation.agent_memory_beliefs import AgentBeliefMemoryMixin
from src.simulation.agent_memory_write import AgentMemoryWriteMixin
from src.simulation.agent_memory_retrieval import AgentMemoryRetrievalMixin


class AgentMemoryMixin(
    AgentMemoryWriteMixin,
    AgentMemoryRetrievalMixin,
    AgentBeliefMemoryMixin,
):
    """Unified compatibility surface for all agent memory capabilities."""
