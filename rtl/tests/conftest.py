"""
pytest/cocotb configuration and fixtures for co-simulation tests.

This module provides:
- Shared pytest fixtures for plant models
- cocotb configuration for waveform dumping
- Helper fixtures for common test patterns
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Environment Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest for cocotb testing."""
    # Enable waveform dumping if requested
    if os.environ.get('DUMP_WAVES', '0') == '1':
        os.environ['COCOTB_WAVES'] = '1'


# =============================================================================
# Plant Model Fixtures
# =============================================================================

@pytest.fixture
def photonic_plant():
    """Create a PhotonicPlant with default test parameters."""
    from plant.plant_wrapper import PhotonicPlant

    return PhotonicPlant(
        tau_thermal=100e-6,
        drift_rate=0.001,
        crosstalk_coeff=0.05,
        receiver_gain=2047.0,  # ADC full-scale = 1.0 optical
        noise_std=2.0,
        adc_bits=12,
        dac_bits=16,
        seed=42,
    )


@pytest.fixture
def svd_photonic_plant():
    """Create an SVDPhotonicPlant with default test parameters."""
    from plant.svd_plant import SVDPhotonicPlant

    return SVDPhotonicPlant(
        tau_thermal=100e-6,
        drift_rate=0.001,
        crosstalk_coeff=0.05,
        receiver_gain=2047.0,  # ADC full-scale = 1.0 optical
        noise_std=2.0,
        adc_bits=12,
        dac_bits=16,
        voa_bits=16,
        max_attenuation_db=40.0,
        seed=42,
    )


@pytest.fixture
def low_noise_plant():
    """Create a low-noise plant for precision tests."""
    from plant.plant_wrapper import PhotonicPlant

    return PhotonicPlant(
        noise_std=0.5,
        drift_rate=0.0,  # No drift
        seed=42,
    )


@pytest.fixture
def noisy_plant():
    """Create a high-noise plant for robustness tests."""
    from plant.plant_wrapper import PhotonicPlant

    return PhotonicPlant(
        noise_std=10.0,
        drift_rate=0.01,  # Higher drift
        seed=42,
    )


# =============================================================================
# Target Matrix Fixtures
# =============================================================================

@pytest.fixture
def identity_weights():
    """Identity matrix weights."""
    return (0.99997, 0.0, 0.0, 0.99997)  # Near 1.0 for Q1.15


@pytest.fixture
def hadamard_weights():
    """Hadamard matrix weights."""
    from math import sqrt
    h = 1.0 / sqrt(2)
    return (h, h, h, -h)


@pytest.fixture
def swap_weights():
    """Swap matrix weights."""
    return (0.0, 0.99997, 0.99997, 0.0)


@pytest.fixture
def diagonal_weights():
    """Diagonal (non-unitary) matrix weights for SVD mode."""
    return (0.5, 0.0, 0.0, 0.3)


@pytest.fixture
def arbitrary_weights():
    """Arbitrary matrix weights for SVD mode."""
    return (0.7, -0.3, 0.2, -0.5)


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def voa_converter():
    """Create VOA for sigma ↔ DAC conversions."""
    from plant.voa import DualVOA

    return DualVOA(
        dac_bits=16,
        max_attenuation_db=40.0,
        insertion_loss_db=0.0,
    )


@pytest.fixture
def q15_converter():
    """Provide Q1.15 conversion functions."""
    from plant.plant_wrapper import float_to_q1_15, q1_15_to_float

    class Q15Converter:
        @staticmethod
        def to_fixed(val: float) -> int:
            return float_to_q1_15(val)

        @staticmethod
        def to_float(code: int) -> float:
            return q1_15_to_float(code)

    return Q15Converter()


# =============================================================================
# Test Configuration Fixtures
# =============================================================================

@pytest.fixture
def calibration_params():
    """Default calibration parameters matching RTL."""
    return {
        'settle_cycles': 16,
        'avg_samples': 8,
        'max_iterations': 1000,
        'lock_threshold': 0x100,
        'lock_count': 4,
        'phase_step_initial': 0x1000,
        'phase_step_min': 0x0040,
    }


@pytest.fixture
def timing_params():
    """Timing parameters for simulation."""
    return {
        'clock_period_ns': 10,
        'reset_cycles': 10,
        'calibration_timeout_cycles': 50000,
        'evaluation_timeout_cycles': 1000,
    }
