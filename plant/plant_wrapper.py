"""
Unified photonic plant wrapper for cocotb simulation.

Combines the MZI mesh, thermal dynamics, and coherent receiver
into a single plant model that interfaces with the RTL controller.
"""

import numpy as np
from numpy.typing import NDArray

from .mzi_mesh import MZIMesh
from .thermal_dynamics import ThermalPhaseArray
from .coherent_receiver import DualCoherentReceiver


class PhotonicPlant:
    """
    Complete photonic plant model for 2x2 coherent matrix multiply.

    Combines:
    - MZI mesh for optical transform
    - Thermal phase shifters with dynamics, drift, and crosstalk
    - Dual coherent receivers for I/Q detection

    This class is the main interface for the cocotb testbench.
    """

    NUM_PHASES = 4
    NUM_OUTPUTS = 2

    def __init__(
        self,
        # Thermal dynamics parameters
        tau_thermal: float = 100e-6,
        drift_rate: float = 0.001,
        crosstalk_coeff: float = 0.05,
        # Receiver parameters
        receiver_gain: float = 1000.0,
        noise_std: float = 5.0,
        adc_bits: int = 12,
        # DAC parameters
        dac_bits: int = 16,
        phase_range: float = 2 * np.pi,
        # Timing
        dt: float = 10e-9,  # 10ns per simulation step (100 MHz clock)
        # Reproducibility
        seed: int | None = None,
    ):
        """
        Initialize photonic plant.

        Args:
            tau_thermal: Thermal time constant in seconds
            drift_rate: Phase drift rate in rad/sqrt(s)
            crosstalk_coeff: Crosstalk coefficient between adjacent heaters
            receiver_gain: Coherent receiver transimpedance gain
            noise_std: Receiver noise in ADC codes
            adc_bits: ADC resolution
            dac_bits: DAC resolution for phase control
            phase_range: Full-scale phase range in radians
            dt: Time step per simulation cycle
            seed: Random seed for reproducibility
        """
        self.dt = dt
        self.dac_bits = dac_bits
        self.phase_range = phase_range
        self.seed = seed

        # Create sub-components
        self.mzi = MZIMesh(seed=seed)

        self.thermal = ThermalPhaseArray(
            num_phases=self.NUM_PHASES,
            tau_thermal=tau_thermal,
            drift_rate=drift_rate,
            crosstalk_coeff=crosstalk_coeff,
            dac_bits=dac_bits,
            phase_range=phase_range,
            seed=None if seed is None else seed + 100,
        )

        self.receiver = DualCoherentReceiver(
            gain=receiver_gain,
            noise_std=noise_std,
            adc_bits=adc_bits,
            seed=None if seed is None else seed + 200,
        )

        # State
        self._cycle_count = 0
        self._last_dac_codes = [0] * self.NUM_PHASES

    def reset(self) -> None:
        """Reset plant to initial state."""
        self.thermal.reset()
        self._cycle_count = 0
        self._last_dac_codes = [0] * self.NUM_PHASES

    def step(
        self,
        dac_codes: list[int],
        x0: complex,
        x1: complex,
    ) -> tuple[complex, complex]:
        """
        Advance plant by one time step and compute outputs.

        Args:
            dac_codes: List of NUM_PHASES DAC codes for phase control
            x0: Complex input field at port 0
            x1: Complex input field at port 1

        Returns:
            Tuple of (y0, y1) complex output fields
        """
        self._cycle_count += 1
        self._last_dac_codes = list(dac_codes)

        # Update thermal dynamics to get actual phases
        actual_phases = self.thermal.step(self.dt, dac_codes)

        # Compute transfer matrix with actual (noisy) phases
        x = np.array([x0, x1], dtype=np.complex128)
        y = self.mzi.apply(x, actual_phases)

        return complex(y[0]), complex(y[1])

    def sample_outputs(
        self,
        dac_codes: list[int],
        x0: complex,
        x1: complex,
    ) -> tuple[int, int, int, int]:
        """
        Complete plant step: compute optical outputs and sample with receiver.

        Args:
            dac_codes: List of NUM_PHASES DAC codes for phase control
            x0: Complex input field at port 0
            x1: Complex input field at port 1

        Returns:
            Tuple of (I0, Q0, I1, Q1) ADC codes
        """
        y0, y1 = self.step(dac_codes, x0, x1)
        return self.receiver.sample(y0, y1)

    def get_transfer_matrix(
        self, dac_codes: list[int] | None = None
    ) -> NDArray[np.complex128]:
        """
        Get the current (or specified) transfer matrix.

        Args:
            dac_codes: Optional DAC codes. If None, uses current thermal state.

        Returns:
            2x2 complex transfer matrix
        """
        if dac_codes is not None:
            # Compute phases from DAC codes (instantaneous, no thermal dynamics)
            phases = np.array([
                (code / ((1 << self.dac_bits) - 1)) * self.phase_range
                for code in dac_codes
            ])
        else:
            # Use current thermal state
            phases = self.thermal.get_phases()

        return self.mzi.compute_transfer_matrix(phases)

    def get_ideal_transfer_matrix(
        self, dac_codes: list[int]
    ) -> NDArray[np.complex128]:
        """
        Get the ideal transfer matrix (no thermal dynamics or drift).

        Args:
            dac_codes: DAC codes for phase control

        Returns:
            2x2 complex transfer matrix
        """
        phases = np.array([
            (code / ((1 << self.dac_bits) - 1)) * self.phase_range
            for code in dac_codes
        ])
        return self.mzi.compute_transfer_matrix(phases)

    @property
    def cycle_count(self) -> int:
        """Number of simulation cycles executed."""
        return self._cycle_count


# Fixed-point conversion utilities for cocotb interface
def float_to_q1_15(value: float) -> int:
    """
    Convert floating-point value to Q1.15 fixed-point.

    Args:
        value: Float in range [-1, 1)

    Returns:
        16-bit signed integer in Q1.15 format
    """
    if value < -1 or value >= 1:
        raise ValueError(f"Value {value} out of Q1.15 range [-1, 1)")
    scaled = int(np.round(value * (1 << 15)))
    # Clamp to 16-bit signed range
    return max(-32768, min(32767, scaled))


def q1_15_to_float(code: int) -> float:
    """
    Convert Q1.15 fixed-point to floating-point.

    Args:
        code: 16-bit signed integer in Q1.15 format

    Returns:
        Float in range [-1, 1)
    """
    # Sign-extend if needed
    if code >= 32768:
        code -= 65536
    return code / (1 << 15)


def validate_weights(w0: float, w1: float, w2: float, w3: float) -> bool:
    """
    Validate that all weights are in [-1, 1].

    Args:
        w0, w1, w2, w3: Target matrix weights

    Returns:
        True if all weights are valid

    Raises:
        ValueError: If any weight is out of range
    """
    weights = [w0, w1, w2, w3]
    for i, w in enumerate(weights):
        if w < -1 or w > 1:
            raise ValueError(f"Weight w{i}={w} out of range [-1, 1]")
    return True
