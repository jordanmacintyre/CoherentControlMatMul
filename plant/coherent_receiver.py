"""
Coherent receiver model with I/Q detection.

Models the coherent detection of optical fields including:
- Balanced detection (I/Q quadratures)
- Thermal/shot noise
- ADC quantization and clipping
"""

import numpy as np
from numpy.typing import NDArray


class CoherentReceiver:
    """
    I/Q coherent detection with noise and ADC quantization.

    Models a balanced coherent receiver that converts complex optical
    fields to I (in-phase) and Q (quadrature) electrical signals.
    """

    def __init__(
        self,
        gain: float = 1000.0,
        noise_std: float = 5.0,
        adc_bits: int = 12,
        seed: int | None = None,
    ):
        """
        Initialize coherent receiver.

        Args:
            gain: Transimpedance gain (V/field amplitude)
            noise_std: Noise standard deviation in ADC codes
            adc_bits: ADC resolution in bits
            seed: Random seed for reproducibility
        """
        self.gain = gain
        self.noise_std = noise_std
        self.adc_bits = adc_bits
        self.adc_max = (1 << (adc_bits - 1)) - 1  # Max positive value
        self.rng = np.random.default_rng(seed)

    def sample_single(self, y_complex: complex) -> tuple[int, int]:
        """
        Convert a single complex optical field to I/Q ADC codes.

        Args:
            y_complex: Complex optical field amplitude

        Returns:
            Tuple of (I, Q) ADC codes as integers
        """
        # Convert complex field to I/Q voltages
        I_voltage = self.gain * np.real(y_complex)
        Q_voltage = self.gain * np.imag(y_complex)

        # Add noise
        I_noisy = I_voltage + self.noise_std * self.rng.standard_normal()
        Q_noisy = Q_voltage + self.noise_std * self.rng.standard_normal()

        # Quantize and clip
        I_adc = int(np.clip(np.round(I_noisy), -self.adc_max - 1, self.adc_max))
        Q_adc = int(np.clip(np.round(Q_noisy), -self.adc_max - 1, self.adc_max))

        return I_adc, Q_adc

    def sample(
        self, y: NDArray[np.complex128]
    ) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """
        Convert complex optical fields to I/Q ADC codes.

        Args:
            y: Array of complex optical field amplitudes

        Returns:
            Tuple of (I, Q) arrays as integer ADC codes
        """
        y = np.atleast_1d(y)

        # Convert complex fields to I/Q voltages
        I_voltage = self.gain * np.real(y)
        Q_voltage = self.gain * np.imag(y)

        # Add noise
        I_noisy = I_voltage + self.noise_std * self.rng.standard_normal(y.shape)
        Q_noisy = Q_voltage + self.noise_std * self.rng.standard_normal(y.shape)

        # Quantize and clip
        I_adc = np.clip(
            np.round(I_noisy), -self.adc_max - 1, self.adc_max
        ).astype(np.int32)
        Q_adc = np.clip(
            np.round(Q_noisy), -self.adc_max - 1, self.adc_max
        ).astype(np.int32)

        return I_adc, Q_adc

    def reconstruct_complex(
        self, I: int | NDArray[np.int32], Q: int | NDArray[np.int32]
    ) -> complex | NDArray[np.complex128]:
        """
        Reconstruct complex field estimate from I/Q measurements.

        This is the inverse of the forward path (without noise recovery).

        Args:
            I: In-phase ADC code(s)
            Q: Quadrature ADC code(s)

        Returns:
            Estimated complex field amplitude(s)
        """
        return (np.asarray(I) + 1j * np.asarray(Q)) / self.gain

    def is_saturated(self, I: int, Q: int) -> bool:
        """
        Check if ADC is saturated (clipping occurred).

        Args:
            I: In-phase ADC code
            Q: Quadrature ADC code

        Returns:
            True if either I or Q is at the rail
        """
        return (
            abs(I) >= self.adc_max or abs(Q) >= self.adc_max
        )


class DualCoherentReceiver:
    """
    Dual-channel coherent receiver for 2x2 matrix multiply outputs.
    """

    def __init__(
        self,
        gain: float = 1000.0,
        noise_std: float = 5.0,
        adc_bits: int = 12,
        seed: int | None = None,
    ):
        """
        Initialize dual coherent receiver.

        Args:
            gain: Transimpedance gain (V/field amplitude)
            noise_std: Noise standard deviation in ADC codes
            adc_bits: ADC resolution in bits
            seed: Random seed for reproducibility
        """
        self.receivers = [
            CoherentReceiver(gain, noise_std, adc_bits,
                           seed=None if seed is None else seed + i)
            for i in range(2)
        ]
        self.gain = gain
        self.noise_std = noise_std
        self.adc_bits = adc_bits

    def sample(
        self, y0: complex, y1: complex
    ) -> tuple[int, int, int, int]:
        """
        Sample both channels.

        Args:
            y0: Complex field at output 0
            y1: Complex field at output 1

        Returns:
            Tuple of (I0, Q0, I1, Q1) ADC codes
        """
        I0, Q0 = self.receivers[0].sample_single(y0)
        I1, Q1 = self.receivers[1].sample_single(y1)
        return I0, Q0, I1, Q1

    def reconstruct_complex(
        self, I0: int, Q0: int, I1: int, Q1: int
    ) -> tuple[complex, complex]:
        """
        Reconstruct complex field estimates from I/Q measurements.

        Args:
            I0, Q0: Channel 0 I/Q ADC codes
            I1, Q1: Channel 1 I/Q ADC codes

        Returns:
            Tuple of (y0_est, y1_est) complex field estimates
        """
        y0 = self.receivers[0].reconstruct_complex(I0, Q0)
        y1 = self.receivers[1].reconstruct_complex(I1, Q1)
        return complex(y0), complex(y1)

    def is_saturated(
        self, I0: int, Q0: int, I1: int, Q1: int
    ) -> bool:
        """
        Check if any channel is saturated.

        Returns:
            True if any I/Q value is at the rail
        """
        return (
            self.receivers[0].is_saturated(I0, Q0) or
            self.receivers[1].is_saturated(I1, Q1)
        )
