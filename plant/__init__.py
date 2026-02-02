"""
Photonic plant model for coherent 2x2 matrix multiply simulation.

This package provides physics-lite models of:
- MZI (Mach-Zehnder Interferometer) mesh for 2x2 unitary transforms
- Thermal phase shifters with RC dynamics, drift, and crosstalk
- Coherent receiver with I/Q detection, noise, and ADC quantization
- Variable Optical Attenuators (VOAs) for amplitude control
- SVD plant for arbitrary (non-unitary) matrix realization
"""

from .mzi_mesh import MZIMesh
from .thermal_dynamics import ThermalPhaseShifter
from .coherent_receiver import CoherentReceiver
from .plant_wrapper import PhotonicPlant
from .voa import VariableOpticalAttenuator, DualVOA
from .svd_plant import SVDPhotonicPlant, validate_svd_realizability

__all__ = [
    "MZIMesh",
    "ThermalPhaseShifter",
    "CoherentReceiver",
    "PhotonicPlant",
    "VariableOpticalAttenuator",
    "DualVOA",
    "SVDPhotonicPlant",
    "validate_svd_realizability",
]
