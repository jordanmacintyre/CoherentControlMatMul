"""
Variable Optical Attenuator (VOA) model for amplitude modulation.

VOAs are used in the SVD decomposition to implement the diagonal
singular value matrix Σ. Each VOA attenuates the optical field
amplitude to realize a target singular value σ ∈ [0, 1].
"""

import numpy as np
from numpy.typing import NDArray


class VariableOpticalAttenuator:
    """
    Models electro-optic amplitude modulation via variable optical attenuation.

    In an SVD photonic architecture (M = U·Σ·V†), VOAs implement the
    diagonal Σ matrix by attenuating the optical field amplitude.

    Attenuation is controlled via a DAC code that sets the transmission:
        transmission = 10^(-attenuation_dB / 20)

    The transmission ranges from 1.0 (no attenuation) to ~0 (max attenuation).
    """

    def __init__(
        self,
        dac_bits: int = 16,
        max_attenuation_db: float = 40.0,
        insertion_loss_db: float = 0.5,
        seed: int | None = None,
    ):
        """
        Initialize VOA model.

        Args:
            dac_bits: DAC resolution for attenuation control
            max_attenuation_db: Maximum attenuation in dB
            insertion_loss_db: Fixed insertion loss (always present)
            seed: Random seed for reproducibility
        """
        self.dac_bits = dac_bits
        self.max_attenuation_db = max_attenuation_db
        self.insertion_loss_db = insertion_loss_db
        self.rng = np.random.default_rng(seed)

        self._dac_max = (1 << dac_bits) - 1
        self._insertion_transmission = 10.0 ** (-insertion_loss_db / 20.0)

    def dac_to_transmission(self, dac_code: int) -> float:
        """
        Convert DAC code to optical field transmission.

        Args:
            dac_code: Unsigned integer in [0, 2^dac_bits - 1]
                     0 = no attenuation (max transmission)
                     max = full attenuation (min transmission)

        Returns:
            Amplitude transmission factor in (0, 1]
        """
        # Normalize DAC code to [0, 1]
        normalized = dac_code / self._dac_max

        # Convert to attenuation in dB (0 = no atten, 1 = max atten)
        attenuation_db = normalized * self.max_attenuation_db

        # Convert to amplitude transmission
        variable_transmission = 10.0 ** (-attenuation_db / 20.0)

        # Include insertion loss
        return self._insertion_transmission * variable_transmission

    def sigma_to_dac(self, sigma: float) -> int:
        """
        Convert singular value σ ∈ [0, 1] to DAC code.

        The mapping is:
            σ → attenuation_dB → dac_code

        Where σ = transmission (accounting for insertion loss).

        Args:
            sigma: Target singular value in [0, 1]

        Returns:
            DAC code to achieve the target σ
        """
        # Clamp sigma to valid range
        sigma = np.clip(sigma, 1e-6, 1.0)

        # Account for insertion loss
        target_variable_transmission = sigma / self._insertion_transmission
        target_variable_transmission = np.clip(target_variable_transmission, 1e-6, 1.0)

        # Convert to attenuation in dB
        attenuation_db = -20.0 * np.log10(target_variable_transmission)

        # Clamp to VOA range
        attenuation_db = np.clip(attenuation_db, 0.0, self.max_attenuation_db)

        # Convert to DAC code
        normalized = attenuation_db / self.max_attenuation_db
        dac_code = int(np.round(normalized * self._dac_max))

        return dac_code

    def apply(self, field: complex, dac_code: int) -> complex:
        """
        Apply attenuation to an optical field.

        Args:
            field: Complex optical field amplitude
            dac_code: Attenuation control DAC code

        Returns:
            Attenuated complex field
        """
        transmission = self.dac_to_transmission(dac_code)
        return field * transmission

    def get_achievable_sigma(self) -> tuple[float, float]:
        """
        Get the achievable range of singular values.

        Returns:
            (min_sigma, max_sigma) tuple
        """
        # Max sigma: no variable attenuation, just insertion loss
        max_sigma = self._insertion_transmission

        # Min sigma: full attenuation
        min_sigma = self.dac_to_transmission(self._dac_max)

        return (min_sigma, max_sigma)


class DualVOA:
    """
    Pair of VOAs for implementing the 2x2 diagonal Σ matrix.

    In the SVD architecture, this implements:
        Σ = diag(σ₀, σ₁)
    """

    def __init__(
        self,
        dac_bits: int = 16,
        max_attenuation_db: float = 40.0,
        insertion_loss_db: float = 0.5,
        seed: int | None = None,
    ):
        """
        Initialize dual VOA.

        Args:
            dac_bits: DAC resolution for attenuation control
            max_attenuation_db: Maximum attenuation in dB
            insertion_loss_db: Fixed insertion loss
            seed: Random seed for reproducibility
        """
        self.voas = [
            VariableOpticalAttenuator(
                dac_bits=dac_bits,
                max_attenuation_db=max_attenuation_db,
                insertion_loss_db=insertion_loss_db,
                seed=None if seed is None else seed,
            ),
            VariableOpticalAttenuator(
                dac_bits=dac_bits,
                max_attenuation_db=max_attenuation_db,
                insertion_loss_db=insertion_loss_db,
                seed=None if seed is None else seed + 1,
            ),
        ]
        self.dac_bits = dac_bits

    def apply(
        self, y: NDArray[np.complex128], dac_codes: list[int]
    ) -> NDArray[np.complex128]:
        """
        Apply diagonal attenuation to a 2-element field vector.

        Args:
            y: Complex field vector [y0, y1]
            dac_codes: List of 2 DAC codes [dac0, dac1]

        Returns:
            Attenuated field vector
        """
        return np.array([
            self.voas[0].apply(y[0], dac_codes[0]),
            self.voas[1].apply(y[1], dac_codes[1]),
        ], dtype=np.complex128)

    def sigma_to_dac(self, sigma: NDArray[np.float64]) -> list[int]:
        """
        Convert singular values to DAC codes.

        Args:
            sigma: Array of 2 singular values [σ₀, σ₁]

        Returns:
            List of 2 DAC codes
        """
        return [
            self.voas[0].sigma_to_dac(sigma[0]),
            self.voas[1].sigma_to_dac(sigma[1]),
        ]

    def dac_to_sigma(self, dac_codes: list[int]) -> NDArray[np.float64]:
        """
        Convert DAC codes to effective singular values.

        Args:
            dac_codes: List of 2 DAC codes

        Returns:
            Array of 2 transmission values (effective σ)
        """
        return np.array([
            self.voas[0].dac_to_transmission(dac_codes[0]),
            self.voas[1].dac_to_transmission(dac_codes[1]),
        ])
