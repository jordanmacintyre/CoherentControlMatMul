"""
Compute correctness tests for coherent matrix multiply.

Tests that evaluation mode produces correct outputs after calibration.
"""

import sys
from pathlib import Path

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
    set_inputs,
    start_calibration,
    start_evaluation,
    wait_for_calibration,
    wait_for_evaluation,
)
from tests.utils.golden_model import GoldenModel
from tests.utils.scoreboard import MetricsScoreboard
from tests.utils.stimulus import StimulusGenerator, q1_15_to_float


scoreboard = MetricsScoreboard()


async def calibrate_and_verify(dut, plant, w0, w1, w2, w3, max_cal_cycles=10000):
    """
    Helper: Calibrate to target weights and verify lock.

    Returns:
        True if calibration locked, False otherwise
    """
    set_weights(dut, w0=w0, w1=w1, w2=w2, w3=w3)

    adapter = PlantInLoop(dut, plant)
    cocotb.start_soon(adapter.run())

    await start_calibration(dut)
    locked, cycles = await wait_for_calibration(dut, max_cal_cycles)

    return locked, adapter


@cocotb.test()
async def test_identity_compute(dut):
    """
    Test: Compute mode with identity matrix.

    After locking to identity, outputs should match inputs.
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    plant = PhotonicPlant(
        drift_rate=0.0,
        noise_std=0.0,
        tau_thermal=1e-9,
        crosstalk_coeff=0.0,
        seed=1000,
    )

    await reset_dut(dut)

    # Calibrate to identity
    locked, adapter = await calibrate_and_verify(
        dut, plant, w0=1.0, w1=0.0, w2=0.0, w3=1.0
    )
    assert locked, "Failed to lock identity matrix"

    # Golden model
    golden = GoldenModel()
    golden.set_weights(1.0, 0.0, 0.0, 1.0)

    # Test with various inputs
    test_inputs = [
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.5),
        (-0.5, 0.5),
        (0.9, -0.3),
    ]

    errors = []
    for x0, x1 in test_inputs:
        set_inputs(dut, x0, x1)
        await start_evaluation(dut)
        done, _ = await wait_for_evaluation(dut)
        assert done, "Evaluation did not complete"

        # Read outputs
        y0_code = int(dut.y0_out.value.signed_integer)
        y1_code = int(dut.y1_out.value.signed_integer)
        y0_actual = q1_15_to_float(y0_code)
        y1_actual = q1_15_to_float(y1_code)

        # Compare to golden
        y0_expected, y1_expected = golden.compute(x0, x1)
        error = np.sqrt((y0_actual - y0_expected)**2 + (y1_actual - y1_expected)**2)
        errors.append(error)

        dut._log.info(
            f"Input ({x0:.2f}, {x1:.2f}): "
            f"Output ({y0_actual:.3f}, {y1_actual:.3f}), "
            f"Expected ({y0_expected:.3f}, {y1_expected:.3f}), "
            f"Error {error:.4f}"
        )

    rms_error = np.sqrt(np.mean(np.array(errors)**2))
    scoreboard.record_evaluation(
        test_name="test_identity_compute",
        passed=rms_error < 0.1,
        output_error=rms_error,
        details={"num_tests": len(test_inputs)},
    )

    assert rms_error < 0.1, f"RMS error {rms_error:.4f} exceeds threshold"

    adapter.stop()


@cocotb.test()
async def test_swap_compute(dut):
    """
    Test: Compute mode with swap matrix.

    After locking to swap, y0 should equal x1 and vice versa.
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    plant = PhotonicPlant(
        drift_rate=0.0,
        noise_std=0.0,
        tau_thermal=1e-9,
        crosstalk_coeff=0.0,
        seed=1001,
    )

    await reset_dut(dut)

    locked, adapter = await calibrate_and_verify(
        dut, plant, w0=0.0, w1=1.0, w2=1.0, w3=0.0
    )
    assert locked, "Failed to lock swap matrix"

    golden = GoldenModel()
    golden.set_weights(0.0, 1.0, 1.0, 0.0)

    test_inputs = [
        (1.0, 0.0),
        (0.0, 1.0),
        (0.7, -0.7),
    ]

    errors = []
    for x0, x1 in test_inputs:
        set_inputs(dut, x0, x1)
        await start_evaluation(dut)
        done, _ = await wait_for_evaluation(dut)
        assert done, "Evaluation did not complete"

        y0_actual = q1_15_to_float(int(dut.y0_out.value.signed_integer))
        y1_actual = q1_15_to_float(int(dut.y1_out.value.signed_integer))
        y0_expected, y1_expected = golden.compute(x0, x1)

        error = np.sqrt((y0_actual - y0_expected)**2 + (y1_actual - y1_expected)**2)
        errors.append(error)

    rms_error = np.sqrt(np.mean(np.array(errors)**2))
    scoreboard.record_evaluation(
        test_name="test_swap_compute",
        passed=rms_error < 0.1,
        output_error=rms_error,
    )

    assert rms_error < 0.1, f"RMS error {rms_error:.4f} exceeds threshold"

    adapter.stop()


@cocotb.test()
async def test_random_input_sweep(dut):
    """
    Test: Sweep random inputs through locked matrix.

    Generates 20 random input vectors and verifies outputs.
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    plant = PhotonicPlant(
        drift_rate=0.0,
        noise_std=0.0,
        tau_thermal=1e-9,
        crosstalk_coeff=0.0,
        seed=1002,
    )

    await reset_dut(dut)

    # Use Hadamard-like matrix
    h = 1.0 / np.sqrt(2)
    locked, adapter = await calibrate_and_verify(
        dut, plant, w0=h, w1=h, w2=h, w3=-h
    )
    assert locked, "Failed to lock Hadamard matrix"

    golden = GoldenModel()
    golden.set_weights(h, h, h, -h)

    stim = StimulusGenerator(seed=2000)
    errors = []

    for i in range(20):
        x = stim.random_inputs(2)
        x0, x1 = float(x[0]), float(x[1])

        set_inputs(dut, x0, x1)
        await start_evaluation(dut)
        done, _ = await wait_for_evaluation(dut)
        assert done, f"Evaluation {i} did not complete"

        y0_actual = q1_15_to_float(int(dut.y0_out.value.signed_integer))
        y1_actual = q1_15_to_float(int(dut.y1_out.value.signed_integer))

        error = golden.compute_error(y0_actual, y1_actual, x0, x1)
        errors.append(error)

    rms_error = np.sqrt(np.mean(np.array(errors)**2))
    max_error = np.max(errors)

    scoreboard.record_evaluation(
        test_name="test_random_input_sweep",
        passed=rms_error < 0.1,
        output_error=rms_error,
        details={"max_error": max_error, "num_tests": 20},
    )

    dut._log.info(f"Random sweep: RMS error = {rms_error:.4f}, Max error = {max_error:.4f}")
    assert rms_error < 0.1, f"RMS error {rms_error:.4f} exceeds threshold"

    adapter.stop()


@cocotb.test()
async def test_edge_case_inputs(dut):
    """
    Test: Edge case inputs (corners of valid range).
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    plant = PhotonicPlant(
        drift_rate=0.0,
        noise_std=0.0,
        tau_thermal=1e-9,
        crosstalk_coeff=0.0,
        seed=1003,
    )

    await reset_dut(dut)

    locked, adapter = await calibrate_and_verify(
        dut, plant, w0=1.0, w1=0.0, w2=0.0, w3=1.0
    )
    assert locked, "Failed to lock identity matrix"

    golden = GoldenModel()
    golden.set_weights(1.0, 0.0, 0.0, 1.0)

    stim = StimulusGenerator()
    edge_inputs = stim.edge_case_inputs()

    errors = []
    for x0, x1 in edge_inputs:
        set_inputs(dut, x0, x1)
        await start_evaluation(dut)
        done, _ = await wait_for_evaluation(dut)
        assert done, "Evaluation did not complete"

        y0_actual = q1_15_to_float(int(dut.y0_out.value.signed_integer))
        y1_actual = q1_15_to_float(int(dut.y1_out.value.signed_integer))

        error = golden.compute_error(y0_actual, y1_actual, x0, x1)
        errors.append(error)

        dut._log.info(f"Edge ({x0:.3f}, {x1:.3f}): error = {error:.4f}")

    rms_error = np.sqrt(np.mean(np.array(errors)**2))
    scoreboard.record_evaluation(
        test_name="test_edge_case_inputs",
        passed=rms_error < 0.1,
        output_error=rms_error,
    )

    assert rms_error < 0.1, f"RMS error {rms_error:.4f} exceeds threshold"

    adapter.stop()


@cocotb.test()
async def test_print_compute_summary(dut):
    """
    Final test: Print compute test summary.
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    scoreboard.print_summary()

    results_dir = Path(__file__).parent.parent / "sim" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    scoreboard.to_json(results_dir / "compute_results.json")
