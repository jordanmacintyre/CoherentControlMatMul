"""
Demo package for coherent photonic matrix multiply.

Provides visualization and interactive demonstration of the control loop.
"""

from .control_loop import CoherentController, ControlConfig
from .visualizer import ControlVisualizer

__all__ = [
    "CoherentController",
    "ControlConfig",
    "ControlVisualizer",
]
