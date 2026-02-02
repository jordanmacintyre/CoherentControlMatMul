"""
Thermal dynamics model for phase shifters.

Models the thermal response of heater-based phase shifters including:
- First-order RC thermal time constant
- Random drift (Brownian motion)
- Crosstalk between adjacent heaters
"""

import numpy as np
from numpy.typing import NDArray


def dac_to_phase(dac_code: int, dac_bits: int = 16, phase_range: float = 2 * np.pi) -> float:
    """
    Convert DAC code to phase in radians.

    Args:
        dac_code: DAC code (unsigned integer)
        dac_bits: DAC resolution in bits
        phase_range: Full-scale phase range in radians

    Returns:
        Phase in radians
    """
    dac_max = (1 << dac_bits) - 1
    return (dac_code / dac_max) * phase_range


class ThermalPhaseShifter:
    """
    Models heater-to-phase thermal dynamics with full physics.

    Features:
    - First-order RC thermal response
    - Random drift (Brownian motion)
    - Crosstalk from neighboring heaters
    """

    def __init__(
        self,
        tau_thermal: float = 100e-6,
        drift_rate: float = 0.001,
        crosstalk_coeff: float = 0.05,
        dac_bits: int = 16,
        phase_range: float = 2 * np.pi,
        seed: int | None = None,
    ):
        """
        Initialize thermal phase shifter.

        Args:
            tau_thermal: Thermal time constant in seconds (typical ~100us)
            drift_rate: Drift rate in rad/sqrt(s) for Brownian motion
            crosstalk_coeff: Coupling coefficient from neighboring heater (0-1)
            dac_bits: DAC resolution in bits
            phase_range: Full-scale phase range in radians
            seed: Random seed for reproducibility
        """
        self.tau = tau_thermal
        self.drift_rate = drift_rate
        self.crosstalk = crosstalk_coeff
        self.dac_bits = dac_bits
        self.phase_range = phase_range
        self.rng = np.random.default_rng(seed)

        # Internal state
        self.phase_actual = 0.0  # Current physical phase
        self.phase_target = 0.0  # Target phase from DAC
        self.drift_offset = 0.0  # Accumulated drift

    def reset(self) -> None:
        """Reset internal state to initial conditions."""
        self.phase_actual = 0.0
        self.phase_target = 0.0
        self.drift_offset = 0.0

    def step(
        self, dt: float, dac_code: int, neighbor_phase: float = 0.0
    ) -> float:
        """
        Update phase based on DAC command, thermal dynamics, and crosstalk.

        Args:
            dt: Time step in seconds
            dac_code: DAC code commanding target phase
            neighbor_phase: Phase of neighboring heater (for crosstalk)

        Returns:
            Current actual phase in radians
        """
        # Convert DAC to target phase
        self.phase_target = dac_to_phase(
            dac_code, self.dac_bits, self.phase_range
        )

        # First-order thermal response (RC dynamics)
        # phase_actual approaches phase_target exponentially
        alpha = 1 - np.exp(-dt / self.tau)
        self.phase_actual += alpha * (self.phase_target - self.phase_actual)

        # Random drift (Brownian motion)
        # Standard deviation scales with sqrt(dt)
        self.drift_offset += self.drift_rate * np.sqrt(dt) * self.rng.standard_normal()

        # Crosstalk contribution from neighboring heater
        crosstalk_contribution = self.crosstalk * neighbor_phase

        return self.phase_actual + self.drift_offset + crosstalk_contribution

    def get_phase(self, neighbor_phase: float = 0.0) -> float:
        """
        Get current phase without advancing time.

        Args:
            neighbor_phase: Phase of neighboring heater (for crosstalk)

        Returns:
            Current actual phase in radians
        """
        return self.phase_actual + self.drift_offset + self.crosstalk * neighbor_phase


class ThermalPhaseArray:
    """
    Array of thermal phase shifters with inter-element crosstalk.
    """

    def __init__(
        self,
        num_phases: int = 4,
        tau_thermal: float = 100e-6,
        drift_rate: float = 0.001,
        crosstalk_coeff: float = 0.05,
        dac_bits: int = 16,
        phase_range: float = 2 * np.pi,
        seed: int | None = None,
    ):
        """
        Initialize array of thermal phase shifters.

        Args:
            num_phases: Number of phase shifters
            tau_thermal: Thermal time constant in seconds
            drift_rate: Drift rate in rad/sqrt(s)
            crosstalk_coeff: Coupling coefficient between adjacent heaters
            dac_bits: DAC resolution in bits
            phase_range: Full-scale phase range in radians
            seed: Random seed for reproducibility
        """
        self.num_phases = num_phases
        self.rng = np.random.default_rng(seed)

        # Create individual phase shifters with derived seeds
        self.shifters = [
            ThermalPhaseShifter(
                tau_thermal=tau_thermal,
                drift_rate=drift_rate,
                crosstalk_coeff=crosstalk_coeff,
                dac_bits=dac_bits,
                phase_range=phase_range,
                seed=None if seed is None else seed + i,
            )
            for i in range(num_phases)
        ]

    def reset(self) -> None:
        """Reset all phase shifters to initial conditions."""
        for shifter in self.shifters:
            shifter.reset()

    def step(self, dt: float, dac_codes: list[int]) -> NDArray[np.float64]:
        """
        Update all phases with crosstalk coupling.

        Args:
            dt: Time step in seconds
            dac_codes: List of DAC codes for each phase shifter

        Returns:
            Array of actual phase values in radians
        """
        if len(dac_codes) != self.num_phases:
            raise ValueError(
                f"Expected {self.num_phases} DAC codes, got {len(dac_codes)}"
            )

        # First pass: update each shifter without crosstalk
        # (We need previous phases for crosstalk calculation)
        previous_phases = np.array([s.get_phase() for s in self.shifters])

        # Second pass: compute actual phases with crosstalk
        phases = np.zeros(self.num_phases, dtype=np.float64)
        for i, (shifter, dac_code) in enumerate(zip(self.shifters, dac_codes)):
            # Sum crosstalk from neighbors (ring topology for simplicity)
            neighbor_phase = 0.0
            if i > 0:
                neighbor_phase += previous_phases[i - 1]
            if i < self.num_phases - 1:
                neighbor_phase += previous_phases[i + 1]
            neighbor_phase /= 2  # Average contribution from neighbors

            phases[i] = shifter.step(dt, dac_code, neighbor_phase)

        return phases

    def get_phases(self) -> NDArray[np.float64]:
        """
        Get current phases without advancing time.

        Returns:
            Array of current actual phase values in radians
        """
        return np.array([s.get_phase() for s in self.shifters], dtype=np.float64)
