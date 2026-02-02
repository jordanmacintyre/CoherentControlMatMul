"""
Golden reference model for coherent matrix multiply.

Provides high-precision reference implementation for test verification.
"""

import numpy as np
from numpy.typing import NDArray


class GoldenModel:
    """
    Golden reference model for 2x2 matrix multiply.

    Uses double-precision floating-point for reference calculations.
    """

    def __init__(self):
        """Initialize golden model."""
        self._target_matrix = np.eye(2, dtype=np.float64)

    def set_weights(
        self, w0: float, w1: float, w2: float, w3: float
    ) -> None:
        """
        Set target matrix weights.

        Matrix layout:
            M = [[w0, w1],
                 [w2, w3]]

        Args:
            w0: Element (0,0)
            w1: Element (0,1)
            w2: Element (1,0)
            w3: Element (1,1)
        """
        self._target_matrix = np.array(
            [[w0, w1], [w2, w3]], dtype=np.float64
        )

    @property
    def target_matrix(self) -> NDArray[np.float64]:
        """Get the target matrix."""
        return self._target_matrix.copy()

    def compute(self, x0: float, x1: float) -> tuple[float, float]:
        """
        Compute reference output for given inputs.

        Args:
            x0: Input 0
            x1: Input 1

        Returns:
            Tuple of (y0, y1) reference outputs
        """
        x = np.array([x0, x1], dtype=np.float64)
        y = self._target_matrix @ x
        return float(y[0]), float(y[1])

    def compute_batch(
        self, inputs: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """
        Compute reference outputs for multiple inputs.

        Args:
            inputs: List of (x0, x1) tuples

        Returns:
            List of (y0, y1) reference outputs
        """
        return [self.compute(x0, x1) for x0, x1 in inputs]

    def compute_error(
        self,
        actual_y0: float,
        actual_y1: float,
        x0: float,
        x1: float,
    ) -> float:
        """
        Compute error between actual and expected outputs.

        Args:
            actual_y0: Actual output 0
            actual_y1: Actual output 1
            x0: Input 0
            x1: Input 1

        Returns:
            RMS error
        """
        expected_y0, expected_y1 = self.compute(x0, x1)
        error = np.sqrt(
            (actual_y0 - expected_y0) ** 2 + (actual_y1 - expected_y1) ** 2
        )
        return float(error)

    def compute_matrix_error(
        self, measured_matrix: NDArray[np.float64]
    ) -> float:
        """
        Compute Frobenius norm error between measured and target matrices.

        Args:
            measured_matrix: 2x2 measured matrix

        Returns:
            Frobenius norm of (measured - target)
        """
        diff = measured_matrix - self._target_matrix
        return float(np.linalg.norm(diff, 'fro'))

    def compute_relative_error(
        self,
        actual_y0: float,
        actual_y1: float,
        x0: float,
        x1: float,
    ) -> float:
        """
        Compute relative error between actual and expected outputs.

        Args:
            actual_y0: Actual output 0
            actual_y1: Actual output 1
            x0: Input 0
            x1: Input 1

        Returns:
            Relative error (error / expected_magnitude)
        """
        expected_y0, expected_y1 = self.compute(x0, x1)
        expected_mag = np.sqrt(expected_y0**2 + expected_y1**2)

        if expected_mag < 1e-10:
            # Avoid division by zero for near-zero expected output
            return 0.0 if (abs(actual_y0) < 1e-10 and abs(actual_y1) < 1e-10) else float('inf')

        error = np.sqrt(
            (actual_y0 - expected_y0) ** 2 + (actual_y1 - expected_y1) ** 2
        )
        return float(error / expected_mag)


def reconstruct_matrix_from_iq(
    col0_i0: float, col0_q0: float, col0_i1: float, col0_q1: float,
    col1_i0: float, col1_q0: float, col1_i1: float, col1_q1: float,
    gain: float = 1000.0,
) -> NDArray[np.complex128]:
    """
    Reconstruct measured matrix from I/Q measurements.

    Measurements are taken with basis inputs:
    - Column 0: x = [1, 0] -> y = [M00, M10]
    - Column 1: x = [0, 1] -> y = [M01, M11]

    Args:
        col0_i0, col0_q0: I/Q for output 0 with input [1,0]
        col0_i1, col0_q1: I/Q for output 1 with input [1,0]
        col1_i0, col1_q0: I/Q for output 0 with input [0,1]
        col1_i1, col1_q1: I/Q for output 1 with input [0,1]
        gain: Receiver gain for scaling

    Returns:
        2x2 complex measured matrix
    """
    M00 = complex(col0_i0, col0_q0) / gain
    M10 = complex(col0_i1, col0_q1) / gain
    M01 = complex(col1_i0, col1_q0) / gain
    M11 = complex(col1_i1, col1_q1) / gain

    return np.array([[M00, M01], [M10, M11]], dtype=np.complex128)


def extract_real_matrix(
    complex_matrix: NDArray[np.complex128],
) -> NDArray[np.float64]:
    """
    Extract real part of complex matrix.

    For real-only targets, the imaginary components represent phase error.

    Args:
        complex_matrix: 2x2 complex matrix

    Returns:
        2x2 real matrix (real parts only)
    """
    return np.real(complex_matrix).astype(np.float64)


def compute_q_residual(
    complex_matrix: NDArray[np.complex128],
) -> float:
    """
    Compute Q-residual (imaginary energy) of matrix.

    For well-calibrated real-only targets, this should be small.

    Args:
        complex_matrix: 2x2 complex matrix

    Returns:
        Sum of squared imaginary components
    """
    imag_part = np.imag(complex_matrix)
    return float(np.sum(imag_part**2))
