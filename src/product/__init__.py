"""Cohesive product surface for Generative Football Society."""

from src.product.workspace import ProductWorkspace, StudioConfig
from src.product.pilot import PilotProtocol, ProspectivePilot
from src.product.control_plane import ProductControlPlane
from src.product.web import ProductWebApp, create_product_web_server
from src.product.recovery import ProductRecovery
from src.product.tasks import BackgroundMatchWorker, ProductTaskQueue, TaskConflict
from src.product.telemetry import ProductTelemetry
from src.product.season import SeasonPlan
from src.product.season_commitments import SeasonCommitmentPlan
from src.product.player_promises import PlayerPromisePlan, PlayerRolePromise

__all__ = [
    "BackgroundMatchWorker", "PilotProtocol", "ProductControlPlane",
    "ProductTaskQueue", "ProductTelemetry", "ProductWorkspace", "TaskConflict",
    "PlayerPromisePlan", "PlayerRolePromise", "ProductRecovery", "ProductWebApp",
    "ProspectivePilot", "StudioConfig",
    "SeasonCommitmentPlan", "SeasonPlan", "create_product_web_server",
]
