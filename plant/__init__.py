"""
Photonic plant model for coherent 2x2 matrix multiply simulation.

This package provides physics-lite models of:
- MZI (Mach-Zehnder Interferometer) mesh for 2x2 unitary transforms
- Thermal phase shifters with RC dynamics, drift, and crosstalk
- Coherent receiver with I/Q detection, noise, and ADC quantization
"""

from .mzi_mesh import MZIMesh
from .thermal_dynamics import ThermalPhaseShifter
from .coherent_receiver import CoherentReceiver
from .plant_wrapper import PhotonicPlant

__all__ = [
    "MZIMesh",
    "ThermalPhaseShifter",
    "CoherentReceiver",
    "PhotonicPlant",
]
