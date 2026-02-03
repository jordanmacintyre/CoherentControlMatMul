"""
SVD-based photonic plant for arbitrary 2x2 matrix realization.

Implements M = U · Σ · V† architecture:
- V† MZI mesh (4 phases): Input-side unitary transform
- Σ diagonal VOAs (2 attenuators): Singular value scaling
- U MZI mesh (4 phases): Output-side unitary transform

This allows realization of any 2x2 matrix with weights in [-1, 1],
not just unitary matrices like the single-mesh architecture.
"""

import numpy as np
from numpy.typing import NDArray

from .mzi_mesh import MZIMesh
from .thermal_dynamics import ThermalPhaseArray
from .coherent_receiver import DualCoherentReceiver
from .voa import DualVOA


class SVDPhotonicPlant:
    """
    Complete SVD photonic plant model for arbitrary 2x2 matrix multiply.

    Architecture:
        Input x → [MZI V† (4 phases)] → [VOA Σ (2 gains)] → [MZI U (4 phases)] → Output y

    Signal flow implements: y = U · Σ · V† · x

    This class extends PhotonicPlant to support arbitrary (non-unitary) matrices
    via the SVD decomposition.
    """

    # Parameter counts
    NUM_PHASES_V = 4      # V† mesh phases
    NUM_PHASES_U = 4      # U mesh phases
    NUM_VOAS = 2          # Diagonal attenuators
    TOTAL_PARAMS = NUM_PHASES_V + NUM_VOAS + NUM_PHASES_U  # 10 total

    NUM_OUTPUTS = 2

    def __init__(
        self,
        # Thermal dynamics parameters
        tau_thermal: float = 100e-6,
        drift_rate: float = 0.001,
        crosstalk_coeff: float = 0.05,
        # Receiver parameters
        receiver_gain: float = 2047.0,  # Fills ADC range: 1.0 optical → 2047 LSB
        noise_std: float = 5.0,
        adc_bits: int = 12,
        # DAC parameters
        dac_bits: int = 16,
        phase_range: float = 2 * np.pi,
        # VOA parameters
        voa_bits: int = 16,
        max_attenuation_db: float = 40.0,
        voa_insertion_loss_db: float = 0.0,  # Zero insertion loss for ideal simulation
        # Timing
        dt: float = 10e-9,  # 10ns per simulation step (100 MHz clock)
        # Reproducibility
        seed: int | None = None,
    ):
        """
        Initialize SVD photonic plant.

        Args:
            tau_thermal: Thermal time constant in seconds
            drift_rate: Phase drift rate in rad/sqrt(s)
            crosstalk_coeff: Crosstalk coefficient between adjacent heaters
            receiver_gain: Coherent receiver transimpedance gain
            noise_std: Receiver noise in ADC codes
            adc_bits: ADC resolution
            dac_bits: DAC resolution for phase control
            phase_range: Full-scale phase range in radians
            voa_bits: DAC resolution for VOA control
            max_attenuation_db: Maximum VOA attenuation
            voa_insertion_loss_db: VOA insertion loss
            dt: Time step per simulation cycle
            seed: Random seed for reproducibility
        """
        self.dt = dt
        self.dac_bits = dac_bits
        self.voa_bits = voa_bits
        self.phase_range = phase_range
        self.seed = seed

        # Create MZI meshes
        self.mzi_v = MZIMesh(seed=seed)  # V† (input side)
        self.mzi_u = MZIMesh(seed=None if seed is None else seed + 50)  # U (output side)

        # Create thermal phase arrays for both meshes
        self.thermal_v = ThermalPhaseArray(
            num_phases=self.NUM_PHASES_V,
            tau_thermal=tau_thermal,
            drift_rate=drift_rate,
            crosstalk_coeff=crosstalk_coeff,
            dac_bits=dac_bits,
            phase_range=phase_range,
            seed=None if seed is None else seed + 100,
        )

        self.thermal_u = ThermalPhaseArray(
            num_phases=self.NUM_PHASES_U,
            tau_thermal=tau_thermal,
            drift_rate=drift_rate,
            crosstalk_coeff=crosstalk_coeff,
            dac_bits=dac_bits,
            phase_range=phase_range,
            seed=None if seed is None else seed + 200,
        )

        # Create dual VOA for singular values
        self.voas = DualVOA(
            dac_bits=voa_bits,
            max_attenuation_db=max_attenuation_db,
            insertion_loss_db=voa_insertion_loss_db,
            seed=None if seed is None else seed + 300,
        )

        # Create coherent receiver
        self.receiver = DualCoherentReceiver(
            gain=receiver_gain,
            noise_std=noise_std,
            adc_bits=adc_bits,
            seed=None if seed is None else seed + 400,
        )

        # State
        self._cycle_count = 0
        self._last_dac_codes_v = [0] * self.NUM_PHASES_V
        self._last_dac_codes_voa = [0] * self.NUM_VOAS
        self._last_dac_codes_u = [0] * self.NUM_PHASES_U

    def reset(self) -> None:
        """Reset plant to initial state."""
        self.thermal_v.reset()
        self.thermal_u.reset()
        self._cycle_count = 0
        self._last_dac_codes_v = [0] * self.NUM_PHASES_V
        self._last_dac_codes_voa = [0] * self.NUM_VOAS
        self._last_dac_codes_u = [0] * self.NUM_PHASES_U

    def _parse_dac_codes(
        self, dac_codes: list[int]
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Parse combined DAC codes into component groups.

        Layout: [v0, v1, v2, v3, σ0, σ1, u0, u1, u2, u3]

        Args:
            dac_codes: List of TOTAL_PARAMS (10) DAC codes

        Returns:
            Tuple of (v_phases, voa_codes, u_phases)
        """
        if len(dac_codes) != self.TOTAL_PARAMS:
            raise ValueError(
                f"Expected {self.TOTAL_PARAMS} DAC codes, got {len(dac_codes)}"
            )

        v_codes = dac_codes[0:4]              # V† phases
        voa_codes = dac_codes[4:6]            # Σ attenuations
        u_codes = dac_codes[6:10]             # U phases

        return v_codes, voa_codes, u_codes

    def step(
        self,
        dac_codes: list[int],
        x0: complex,
        x1: complex,
    ) -> tuple[complex, complex]:
        """
        Advance plant by one time step and compute outputs.

        Signal flow: x → V† → Σ → U → y

        Args:
            dac_codes: List of TOTAL_PARAMS DAC codes
                       [v0,v1,v2,v3, σ0,σ1, u0,u1,u2,u3]
            x0: Complex input field at port 0
            x1: Complex input field at port 1

        Returns:
            Tuple of (y0, y1) complex output fields
        """
        self._cycle_count += 1

        # Parse DAC codes
        v_codes, voa_codes, u_codes = self._parse_dac_codes(dac_codes)
        self._last_dac_codes_v = v_codes
        self._last_dac_codes_voa = voa_codes
        self._last_dac_codes_u = u_codes

        # Update thermal dynamics to get actual phases
        phases_v = self.thermal_v.step(self.dt, v_codes)
        phases_u = self.thermal_u.step(self.dt, u_codes)

        # Signal flow: x → V† → Σ → U → y
        x = np.array([x0, x1], dtype=np.complex128)

        # V† transform (input side)
        y = self.mzi_v.apply(x, phases_v)

        # Σ diagonal attenuation
        y = self.voas.apply(y, voa_codes)

        # U transform (output side)
        y = self.mzi_u.apply(y, phases_u)

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
            dac_codes: List of TOTAL_PARAMS DAC codes
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
        Get the current (or specified) transfer matrix M = U·Σ·V†.

        Args:
            dac_codes: Optional DAC codes. If None, uses current thermal state.

        Returns:
            2x2 complex transfer matrix
        """
        if dac_codes is not None:
            v_codes, voa_codes, u_codes = self._parse_dac_codes(dac_codes)

            # Compute phases from DAC codes (instantaneous)
            dac_max = (1 << self.dac_bits) - 1
            phases_v = np.array([
                (code / dac_max) * self.phase_range for code in v_codes
            ])
            phases_u = np.array([
                (code / dac_max) * self.phase_range for code in u_codes
            ])

            # Get VOA transmissions
            sigma = self.voas.dac_to_sigma(voa_codes)
        else:
            # Use current thermal state
            phases_v = self.thermal_v.get_phases()
            phases_u = self.thermal_u.get_phases()
            sigma = self.voas.dac_to_sigma(self._last_dac_codes_voa)

        # Compute component matrices
        V_dagger = self.mzi_v.compute_transfer_matrix(phases_v)
        Sigma = np.diag(sigma)
        U = self.mzi_u.compute_transfer_matrix(phases_u)

        # Full matrix: M = U · Σ · V†
        return U @ Sigma @ V_dagger

    def get_ideal_transfer_matrix(
        self, dac_codes: list[int]
    ) -> NDArray[np.complex128]:
        """
        Get the ideal transfer matrix (no thermal dynamics or drift).

        Args:
            dac_codes: DAC codes for control

        Returns:
            2x2 complex transfer matrix
        """
        return self.get_transfer_matrix(dac_codes)

    @staticmethod
    def compute_svd_targets(
        M: NDArray[np.float64]
    ) -> tuple[NDArray[np.complex128], NDArray[np.float64], NDArray[np.complex128]]:
        """
        Compute SVD decomposition for a target matrix.

        Args:
            M: Target 2x2 real matrix

        Returns:
            Tuple of (U, sigma, Vh) where M ≈ U @ diag(sigma) @ Vh
        """
        U, sigma, Vh = np.linalg.svd(M)
        return U, sigma, Vh

    @property
    def cycle_count(self) -> int:
        """Number of simulation cycles executed."""
        return self._cycle_count


def validate_svd_realizability(
    w0: float, w1: float, w2: float, w3: float,
    max_sigma: float = 1.0
) -> tuple[bool, str]:
    """
    Check if a target matrix can be realized by the SVD architecture.

    The SVD architecture can realize any matrix with singular values ≤ max_sigma.

    Args:
        w0, w1, w2, w3: Target matrix weights
        max_sigma: Maximum achievable singular value (typically 1.0 minus insertion loss)

    Returns:
        Tuple of (is_realizable, reason_if_not)
    """
    M = np.array([[w0, w1], [w2, w3]])
    _, sigma, _ = np.linalg.svd(M)

    if np.max(sigma) > max_sigma:
        return False, f"Singular value {np.max(sigma):.4f} exceeds max {max_sigma:.4f}"

    return True, "Matrix is realizable"
