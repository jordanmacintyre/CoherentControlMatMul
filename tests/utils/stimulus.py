"""
Stimulus generation utilities for cocotb tests.

Provides test vector generation and fixed-point conversion functions.
"""

import numpy as np
from numpy.typing import NDArray


def float_to_q1_15(value: float) -> int:
    """
    Convert floating-point value to Q1.15 fixed-point.

    Args:
        value: Float in range [-1, 1)

    Returns:
        16-bit signed integer in Q1.15 format

    Raises:
        ValueError: If value is outside valid range
    """
    if value < -1 or value > 1:
        raise ValueError(f"Value {value} out of Q1.15 range [-1, 1]")
    # Clamp to slightly less than 1 for Q1.15 representation
    value = min(value, 1.0 - 2**-15)
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
    # Sign-extend if needed (for unsigned representation)
    if code >= 32768:
        code -= 65536
    return code / (1 << 15)


def float_to_phase_dac(phase_rad: float, dac_bits: int = 16) -> int:
    """
    Convert phase in radians to DAC code.

    Args:
        phase_rad: Phase in radians [0, 2*pi)
        dac_bits: DAC resolution in bits

    Returns:
        Unsigned DAC code
    """
    # Normalize to [0, 1)
    phase_normalized = (phase_rad % (2 * np.pi)) / (2 * np.pi)
    dac_max = (1 << dac_bits) - 1
    return int(np.round(phase_normalized * dac_max))


def phase_dac_to_float(dac_code: int, dac_bits: int = 16) -> float:
    """
    Convert DAC code to phase in radians.

    Args:
        dac_code: Unsigned DAC code
        dac_bits: DAC resolution in bits

    Returns:
        Phase in radians [0, 2*pi)
    """
    dac_max = (1 << dac_bits) - 1
    return (dac_code / dac_max) * 2 * np.pi


class StimulusGenerator:
    """
    Generates test stimuli for coherent matrix multiply tests.
    """

    def __init__(self, seed: int | None = None):
        """
        Initialize stimulus generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)

    def random_weights(self, num: int = 4) -> NDArray[np.float64]:
        """
        Generate random weight values in [-1, 1].

        Args:
            num: Number of weights to generate

        Returns:
            Array of random weights
        """
        return self.rng.uniform(-1, 1, num)

    def random_inputs(self, num: int = 2) -> NDArray[np.float64]:
        """
        Generate random input values in [-1, 1].

        Args:
            num: Number of inputs to generate

        Returns:
            Array of random inputs
        """
        return self.rng.uniform(-1, 1, num)

    def basis_vectors(self) -> list[tuple[float, float]]:
        """
        Generate basis vectors for matrix measurement.

        Returns:
            List of (x0, x1) tuples: [(1,0), (0,1)]
        """
        return [(1.0, 0.0), (0.0, 1.0)]

    def edge_case_inputs(self) -> list[tuple[float, float]]:
        """
        Generate edge case input vectors.

        Returns:
            List of (x0, x1) tuples
        """
        return [
            (1.0, 0.0),    # Basis 0
            (0.0, 1.0),    # Basis 1
            (0.707, 0.707),  # Equal magnitude
            (-0.707, 0.707),  # Opposite signs
            (0.999, 0.0),    # Near max
            (-0.999, 0.0),   # Near min
        ]

    def sweep_inputs(
        self, num_points: int = 10
    ) -> list[tuple[float, float]]:
        """
        Generate a sweep of input values.

        Args:
            num_points: Number of points in sweep

        Returns:
            List of (x0, x1) tuples
        """
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        return [(np.cos(a) * 0.9, np.sin(a) * 0.9) for a in angles]

    def identity_matrix_weights(self) -> tuple[float, float, float, float]:
        """
        Generate weights for identity matrix.

        Returns:
            (w0, w1, w2, w3) = (1, 0, 0, 1)
        """
        return (1.0, 0.0, 0.0, 1.0)

    def swap_matrix_weights(self) -> tuple[float, float, float, float]:
        """
        Generate weights for swap/exchange matrix.

        Returns:
            (w0, w1, w2, w3) = (0, 1, 1, 0)
        """
        return (0.0, 1.0, 1.0, 0.0)

    def hadamard_like_weights(self) -> tuple[float, float, float, float]:
        """
        Generate weights for Hadamard-like matrix (normalized).

        Returns:
            (w0, w1, w2, w3) = (0.707, 0.707, 0.707, -0.707)
        """
        h = 1.0 / np.sqrt(2)
        return (h, h, h, -h)


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
