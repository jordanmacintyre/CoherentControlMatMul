"""
MZI (Mach-Zehnder Interferometer) mesh model for 2x2 photonic transforms.

Implements a configurable 2x2 unitary/near-unitary transform using phase shifters.
The mesh can implement any 2x2 unitary matrix via the Clements/Reck decomposition.
"""

import numpy as np
from numpy.typing import NDArray


class MZIMesh:
    """
    2x2 MZI mesh implementing unitary transforms.

    The mesh uses 4 phase shifters:
    - phi[0]: theta - internal MZI phase (controls power splitting)
    - phi[1]: phi_in0 - input phase shifter on port 0
    - phi[2]: phi_in1 - input phase shifter on port 1
    - phi[3]: phi_out - output phase shifter (global phase)

    Transfer matrix form:
        M = exp(j*phi_out) * [e^{j*phi_in0} * cos(theta/2),  j*sin(theta/2)        ]
                             [j*sin(theta/2),                 e^{j*phi_in1} * cos(theta/2)]
    """

    NUM_PHASES = 4

    def __init__(self, seed: int | None = None):
        """
        Initialize MZI mesh.

        Args:
            seed: Random seed for reproducibility (used in noise injection)
        """
        self.rng = np.random.default_rng(seed)
        # Internal phase state (radians)
        self._phases = np.zeros(self.NUM_PHASES, dtype=np.float64)

    @property
    def phases(self) -> NDArray[np.float64]:
        """Current phase values in radians."""
        return self._phases.copy()

    @phases.setter
    def phases(self, values: NDArray[np.float64]) -> None:
        """Set phase values (radians)."""
        if len(values) != self.NUM_PHASES:
            raise ValueError(f"Expected {self.NUM_PHASES} phases, got {len(values)}")
        self._phases = np.asarray(values, dtype=np.float64)

    def set_phases_from_dac(
        self, dac_codes: list[int], dac_bits: int = 16, phase_range: float = 2 * np.pi
    ) -> None:
        """
        Set phases from DAC codes.

        Args:
            dac_codes: List of DAC codes (unsigned integers)
            dac_bits: DAC resolution in bits
            phase_range: Full-scale phase range in radians (default 2*pi)
        """
        dac_max = (1 << dac_bits) - 1
        self._phases = np.array(
            [code / dac_max * phase_range for code in dac_codes], dtype=np.float64
        )

    def compute_transfer_matrix(
        self, phases: NDArray[np.float64] | None = None
    ) -> NDArray[np.complex128]:
        """
        Compute the 2x2 complex transfer matrix for given phases.

        Args:
            phases: Optional phase array. If None, uses internal state.

        Returns:
            2x2 complex transfer matrix
        """
        if phases is None:
            phases = self._phases

        theta = phases[0]  # MZI splitting angle
        phi_in0 = phases[1]  # Input 0 phase
        phi_in1 = phases[2]  # Input 1 phase
        phi_out = phases[3]  # Output global phase

        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)

        # Build transfer matrix
        # Standard MZI with external phase shifters
        M = np.array(
            [
                [np.exp(1j * phi_in0) * cos_half, 1j * sin_half],
                [1j * sin_half, np.exp(1j * phi_in1) * cos_half],
            ],
            dtype=np.complex128,
        )

        # Apply output phase
        M *= np.exp(1j * phi_out)

        return M

    def apply(
        self, x: NDArray[np.complex128], phases: NDArray[np.float64] | None = None
    ) -> NDArray[np.complex128]:
        """
        Apply the MZI mesh transform to input field.

        Args:
            x: Input complex field vector [x0, x1]
            phases: Optional phase array. If None, uses internal state.

        Returns:
            Output complex field vector [y0, y1]
        """
        M = self.compute_transfer_matrix(phases)
        return M @ x

    @staticmethod
    def decompose_unitary(U: NDArray[np.complex128]) -> NDArray[np.float64]:
        """
        Decompose a 2x2 unitary matrix into MZI phases.

        This finds phases [theta, phi_in0, phi_in1, phi_out] such that
        compute_transfer_matrix(phases) approximates U.

        Args:
            U: 2x2 unitary matrix to decompose

        Returns:
            Array of 4 phase values in radians
        """
        # Extract global phase
        det = np.linalg.det(U)
        global_phase = np.angle(det) / 2
        U_normalized = U * np.exp(-1j * global_phase)

        # For a unitary with our structure:
        # [e^{j*phi0}*cos(t/2), j*sin(t/2)]
        # [j*sin(t/2),          e^{j*phi1}*cos(t/2)]
        #
        # |U[0,1]| = |U[1,0]| = sin(t/2)
        # So theta = 2 * arcsin(|U[0,1]|)

        sin_half = np.abs(U_normalized[0, 1])
        sin_half = np.clip(sin_half, 0, 1)  # Numerical safety
        theta = 2 * np.arcsin(sin_half)

        cos_half = np.cos(theta / 2)
        if cos_half > 1e-10:
            phi_in0 = np.angle(U_normalized[0, 0] / cos_half)
            phi_in1 = np.angle(U_normalized[1, 1] / cos_half)
        else:
            # Edge case: full crossover (theta = pi)
            phi_in0 = 0
            phi_in1 = 0

        return np.array([theta, phi_in0, phi_in1, global_phase], dtype=np.float64)

    @staticmethod
    def closest_unitary(M: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Find the closest unitary matrix to M (in Frobenius norm).

        Uses polar decomposition: M = U * P where U is unitary.

        Args:
            M: Input matrix (possibly non-unitary)

        Returns:
            Closest unitary matrix
        """
        U, S, Vh = np.linalg.svd(M)
        return U @ Vh
