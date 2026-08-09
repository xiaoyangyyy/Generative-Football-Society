"""Cohesive product surface for Generative Football Society."""

from src.product.workspace import ProductWorkspace, StudioConfig
from src.product.pilot import PilotProtocol, ProspectivePilot
from src.product.control_plane import ProductControlPlane
from src.product.web import ProductWebApp, create_product_web_server
from src.product.recovery import ProductRecovery
from src.product.tasks import BackgroundMatchWorker, ProductTaskQueue, TaskConflict

__all__ = [
    "BackgroundMatchWorker", "PilotProtocol", "ProductControlPlane",
    "ProductTaskQueue", "ProductWorkspace", "TaskConflict",
    "ProductRecovery", "ProductWebApp", "ProspectivePilot", "StudioConfig",
    "create_product_web_server",
]
