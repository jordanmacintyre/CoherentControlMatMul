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

        Uses the standard Reck/Clements MZI decomposition form that can
        realize ANY 2x2 unitary matrix:

            M = D2 · R(θ) · D1

        where:
            D1 = diag(e^{jφ0}, 1) - input phase on port 0
            D2 = diag(e^{jφ1}, e^{jφ_out}) - output phases
            R(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]] - rotation

        This gives:
            M = [[e^{j(φ0+φ1)}cos(θ/2), -e^{jφ1}sin(θ/2)],
                 [e^{j(φ0+φ_out)}sin(θ/2), e^{jφ_out}cos(θ/2)]]

        Args:
            phases: Optional phase array. If None, uses internal state.

        Returns:
            2x2 complex transfer matrix
        """
        if phases is None:
            phases = self._phases

        theta = phases[0]    # Rotation/splitting angle
        phi_in0 = phases[1]  # Input phase on port 0
        phi_in1 = phases[2]  # Output phase on port 0
        phi_out = phases[3]  # Output phase on port 1 (global-ish)

        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)

        # Build transfer matrix using Reck/Clements structure
        # This can realize ANY 2x2 unitary
        M = np.array(
            [
                [np.exp(1j * (phi_in0 + phi_in1)) * cos_half,
                 -np.exp(1j * phi_in1) * sin_half],
                [np.exp(1j * (phi_in0 + phi_out)) * sin_half,
                 np.exp(1j * phi_out) * cos_half],
            ],
            dtype=np.complex128,
        )

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

        For the Reck/Clements structure:
            M = [[e^{j(φ0+φ1)}cos(θ/2), -e^{jφ1}sin(θ/2)],
                 [e^{j(φ0+φ_out)}sin(θ/2), e^{jφ_out}cos(θ/2)]]

        Any 2x2 unitary can be exactly decomposed into this form.

        Args:
            U: 2x2 unitary matrix to decompose

        Returns:
            Array of 4 phase values [theta, phi_in0, phi_in1, phi_out] in radians
        """
        # From the matrix structure:
        # |U[0,1]| = sin(θ/2) and |U[1,0]| = sin(θ/2)
        # |U[0,0]| = cos(θ/2) and |U[1,1]| = cos(θ/2)

        # Get theta from magnitudes (average for numerical stability)
        sin_half = (np.abs(U[0, 1]) + np.abs(U[1, 0])) / 2
        cos_half = (np.abs(U[0, 0]) + np.abs(U[1, 1])) / 2

        # Normalize in case of numerical errors
        norm = np.sqrt(sin_half**2 + cos_half**2)
        if norm > 1e-10:
            sin_half /= norm
            cos_half /= norm

        sin_half = np.clip(sin_half, 0, 1)
        theta = 2 * np.arcsin(sin_half)

        # Now solve for phases from the matrix elements
        if cos_half > 1e-10 and sin_half > 1e-10:
            # M[1,1] = e^{jφ_out}cos(θ/2)  =>  φ_out = angle(U[1,1])
            phi_out = np.angle(U[1, 1])

            # M[0,1] = -e^{jφ1}sin(θ/2)  =>  φ1 = angle(-U[0,1]) = angle(U[0,1]) + π
            phi_in1 = np.angle(-U[0, 1])

            # M[0,0] = e^{j(φ0+φ1)}cos(θ/2)  =>  φ0 = angle(U[0,0]) - φ1
            phi_in0 = np.angle(U[0, 0]) - phi_in1

        elif cos_half > 1e-10:
            # sin_half ≈ 0: nearly identity
            # M ≈ [[e^{j(φ0+φ1)}, 0], [0, e^{jφ_out}]]
            phi_out = np.angle(U[1, 1])
            phi_in0 = 0
            phi_in1 = np.angle(U[0, 0])

        else:
            # cos_half ≈ 0: nearly full swap (θ ≈ π)
            # M ≈ [[0, -e^{jφ1}], [e^{j(φ0+φ_out)}, 0]]
            phi_in1 = np.angle(-U[0, 1])
            phi_out = 0
            phi_in0 = np.angle(U[1, 0]) - phi_out

        return np.array([theta, phi_in0, phi_in1, phi_out], dtype=np.float64)

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
