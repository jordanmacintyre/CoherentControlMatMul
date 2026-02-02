"""
pytest/cocotb configuration and fixtures.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest


@pytest.fixture
def scoreboard():
    """Provide a fresh MetricsScoreboard for each test."""
    from tests.utils.scoreboard import MetricsScoreboard
    return MetricsScoreboard()


@pytest.fixture
def stimulus_generator():
    """Provide a seeded StimulusGenerator."""
    from tests.utils.stimulus import StimulusGenerator
    return StimulusGenerator(seed=42)


@pytest.fixture
def golden_model():
    """Provide a GoldenModel instance."""
    from tests.utils.golden_model import GoldenModel
    return GoldenModel()
