import warnings

import pytest

from src.match_engine.frame_world.controlled import (
    ActionConditionedFrameWorld,
    ControlledFrameConfig,
)
from src.match_engine.frame_world.model import (
    FrameWorldConfig,
    GraphTemporalFrameWorldModel,
)
from src.match_engine.frame_world.semantic import (
    SemanticActionFrameWorld,
    SemanticFrameConfig,
)


@pytest.mark.parametrize(
    'factory',
    (
        lambda: GraphTemporalFrameWorldModel(
            FrameWorldConfig(hidden_dim=16, graph_heads=4, graph_layers=1)
        ),
        lambda: ActionConditionedFrameWorld(
            ControlledFrameConfig(hidden_dim=16, graph_heads=4, graph_layers=1)
        ),
        lambda: SemanticActionFrameWorld(
            SemanticFrameConfig(hidden_dim=16, graph_heads=4, graph_layers=1)
        ),
    ),
)
def test_norm_first_encoders_disable_unavailable_nested_tensor_path(factory):
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='.*enable_nested_tensor.*')
        model = factory()

    assert model.graph.enable_nested_tensor is False
