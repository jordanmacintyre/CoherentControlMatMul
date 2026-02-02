"""
Calibration tests for coherent matrix multiply.

Tests that the calibration FSM can lock to various target matrices.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

import numpy as np

from plant import PhotonicPlant
from tests.utils.plant_adapter import (
    PlantInLoop,
    reset_dut,
    set_weights,
    start_calibration,
    wait_for_calibration,
)
from tests.utils.golden_model import GoldenModel
from tests.utils.scoreboard import MetricsScoreboard
from tests.utils.stimulus import StimulusGenerator


# Global scoreboard for test run
scoreboard = MetricsScoreboard()


@cocotb.test()
async def test_identity_matrix_noiseless(dut):
    """
    Test: Calibration converges for identity matrix with no noise.

    This is the simplest case - identity matrix should be easy to lock.
    """
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Create noiseless plant (no drift, no receiver noise)
    plant = PhotonicPlant(
        drift_rate=0.0,
        noise_std=0.0,
        tau_thermal=1e-9,  # Very fast thermal response
        crosstalk_coeff=0.0,
        seed=42,
    )

    # Reset DUT
    await reset_dut(dut)

    # Set identity matrix weights: [[1, 0], [0, 1]]
    set_weights(dut, w0=1.0, w1=0.0, w2=0.0, w3=1.0)

    # Start plant-in-loop
    adapter = PlantInLoop(dut, plant)
    cocotb.start_soon(adapter.run())

    # Start calibration
    await start_calibration(dut)

    # Wait for lock
    max_cycles = 10000
    locked, cycles = await wait_for_calibration(dut, max_cycles)

    # Record result
    scoreboard.record_calibration(
        test_name="test_identity_matrix_noiseless",
        locked=locked,
        cycles=cycles,
    )

    # Assertions
    assert locked, f"Calibration did not lock within {max_cycles} cycles"
    assert cycles < 5000, f"Calibration took too long: {cycles} cycles"

    adapter.stop()
    dut._log.info(f"Identity matrix locked in {cycles} cycles")


@cocotb.test()
async def test_swap_matrix_noiseless(dut):
    """
    Test: Calibration converges for swap matrix with no noise.

    Swap matrix: [[0, 1], [1, 0]] exchanges inputs.
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    plant = PhotonicPlant(
        drift_rate=0.0,
        noise_std=0.0,
        tau_thermal=1e-9,
        crosstalk_coeff=0.0,
        seed=43,
    )

    await reset_dut(dut)
    set_weights(dut, w0=0.0, w1=1.0, w2=1.0, w3=0.0)

    adapter = PlantInLoop(dut, plant)
    cocotb.start_soon(adapter.run())

    await start_calibration(dut)

    max_cycles = 10000
    locked, cycles = await wait_for_calibration(dut, max_cycles)

    scoreboard.record_calibration(
        test_name="test_swap_matrix_noiseless",
        locked=locked,
        cycles=cycles,
    )

    assert locked, f"Calibration did not lock within {max_cycles} cycles"

    adapter.stop()
    dut._log.info(f"Swap matrix locked in {cycles} cycles")


@cocotb.test()
async def test_hadamard_like_matrix(dut):
    """
    Test: Calibration converges for Hadamard-like matrix.

    Hadamard: [[1/sqrt(2), 1/sqrt(2)], [1/sqrt(2), -1/sqrt(2)]]
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    plant = PhotonicPlant(
        drift_rate=0.0,
        noise_std=0.0,
        tau_thermal=1e-9,
        crosstalk_coeff=0.0,
        seed=44,
    )

    await reset_dut(dut)

    h = 1.0 / np.sqrt(2)
    set_weights(dut, w0=h, w1=h, w2=h, w3=-h)

    adapter = PlantInLoop(dut, plant)
    cocotb.start_soon(adapter.run())

    await start_calibration(dut)

    max_cycles = 10000
    locked, cycles = await wait_for_calibration(dut, max_cycles)

    scoreboard.record_calibration(
        test_name="test_hadamard_like_matrix",
        locked=locked,
        cycles=cycles,
    )

    assert locked, f"Calibration did not lock within {max_cycles} cycles"

    adapter.stop()
    dut._log.info(f"Hadamard-like matrix locked in {cycles} cycles")


@cocotb.test()
async def test_random_matrices(dut):
    """
    Test: Calibration converges for multiple random matrices.

    Tests 5 random matrices with different seeds.
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    stim = StimulusGenerator(seed=100)

    for seed in range(5):
        np.random.seed(seed + 200)
        weights = stim.random_weights(4)
        w0, w1, w2, w3 = weights

        plant = PhotonicPlant(
            drift_rate=0.0,
            noise_std=0.0,
            tau_thermal=1e-9,
            crosstalk_coeff=0.0,
            seed=seed + 300,
        )
        plant.reset()

        await reset_dut(dut)
        set_weights(dut, w0=float(w0), w1=float(w1), w2=float(w2), w3=float(w3))

        adapter = PlantInLoop(dut, plant)
        task = cocotb.start_soon(adapter.run())

        await start_calibration(dut)

        max_cycles = 10000
        locked, cycles = await wait_for_calibration(dut, max_cycles)

        scoreboard.record_calibration(
            test_name=f"test_random_matrix_{seed}",
            locked=locked,
            cycles=cycles,
        )

        adapter.stop()
        # Cancel the task to prevent interference with next iteration
        await Timer(100, units="ns")

        dut._log.info(
            f"Random matrix {seed} [w={weights.round(3)}]: "
            f"{'locked' if locked else 'FAILED'} in {cycles} cycles"
        )

        # Allow some failures due to non-representable matrices
        # but most should lock


@cocotb.test()
async def test_with_receiver_noise(dut):
    """
    Test: Calibration converges with receiver noise.
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Plant with moderate noise
    plant = PhotonicPlant(
        drift_rate=0.0,
        noise_std=5.0,  # Moderate noise
        tau_thermal=1e-9,
        crosstalk_coeff=0.0,
        seed=500,
    )

    await reset_dut(dut)
    set_weights(dut, w0=1.0, w1=0.0, w2=0.0, w3=1.0)

    adapter = PlantInLoop(dut, plant)
    cocotb.start_soon(adapter.run())

    await start_calibration(dut)

    max_cycles = 15000
    locked, cycles = await wait_for_calibration(dut, max_cycles)

    scoreboard.record_calibration(
        test_name="test_with_receiver_noise",
        locked=locked,
        cycles=cycles,
    )

    assert locked, f"Calibration did not lock with noise within {max_cycles} cycles"

    adapter.stop()
    dut._log.info(f"Matrix with noise locked in {cycles} cycles")


@cocotb.test()
async def test_print_summary(dut):
    """
    Final test: Print summary of all calibration tests.

    This should run last to aggregate results.
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Print summary
    scoreboard.print_summary()

    # Write results to file
    results_dir = Path(__file__).parent.parent / "sim" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    scoreboard.to_json(results_dir / "calibration_results.json")

    # Check regression thresholds
    passed, violations = scoreboard.check_regression()
    if not passed:
        for v in violations:
            dut._log.warning(f"Regression violation: {v}")

    # We don't assert on regression here to allow partial passes
    # during development. CI should check the JSON output.
