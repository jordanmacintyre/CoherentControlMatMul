"""
Edge case tests for coherent matrix multiply.

Tests boundary conditions, error handling, and unusual inputs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from plant import PhotonicPlant
from tests.utils.plant_adapter import (
    PlantInLoop,
    reset_dut,
    set_weights,
    start_calibration,
    wait_for_calibration,
)
from tests.utils.stimulus import float_to_q1_15, validate_weights
from tests.utils.scoreboard import MetricsScoreboard


scoreboard = MetricsScoreboard()


@cocotb.test()
async def test_zero_matrix(dut):
    """
    Test: Calibration to zero matrix.

    All zeros is a valid target but may be hard to achieve
    with a unitary MZI mesh.
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    plant = PhotonicPlant(
        drift_rate=0.0,
        noise_std=0.0,
        tau_thermal=1e-9,
        seed=4000,
    )

    await reset_dut(dut)
    set_weights(dut, w0=0.0, w1=0.0, w2=0.0, w3=0.0)

    adapter = PlantInLoop(dut, plant)
    cocotb.start_soon(adapter.run())

    await start_calibration(dut)
    locked, cycles = await wait_for_calibration(dut, 10000)

    scoreboard.record_calibration(
        test_name="test_zero_matrix",
        locked=locked,
        cycles=cycles,
    )

    # Zero matrix may not lock (not achievable with unitary)
    # We just verify it doesn't crash
    dut._log.info(f"Zero matrix: {'locked' if locked else 'did not lock'} in {cycles} cycles")

    adapter.stop()


@cocotb.test()
async def test_max_positive_weights(dut):
    """
    Test: Weights at maximum positive value.

    All weights near +1.0.
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    plant = PhotonicPlant(
        drift_rate=0.0,
        noise_std=0.0,
        tau_thermal=1e-9,
        seed=4001,
    )

    await reset_dut(dut)
    # All weights at 0.99 (near max)
    set_weights(dut, w0=0.99, w1=0.99, w2=0.99, w3=0.99)

    adapter = PlantInLoop(dut, plant)
    cocotb.start_soon(adapter.run())

    await start_calibration(dut)
    locked, cycles = await wait_for_calibration(dut, 10000)

    scoreboard.record_calibration(
        test_name="test_max_positive_weights",
        locked=locked,
        cycles=cycles,
    )

    dut._log.info(f"Max positive weights: {'locked' if locked else 'did not lock'}")

    adapter.stop()


@cocotb.test()
async def test_max_negative_weights(dut):
    """
    Test: Weights at maximum negative value.

    All weights near -1.0.
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    plant = PhotonicPlant(
        drift_rate=0.0,
        noise_std=0.0,
        tau_thermal=1e-9,
        seed=4002,
    )

    await reset_dut(dut)
    set_weights(dut, w0=-0.99, w1=-0.99, w2=-0.99, w3=-0.99)

    adapter = PlantInLoop(dut, plant)
    cocotb.start_soon(adapter.run())

    await start_calibration(dut)
    locked, cycles = await wait_for_calibration(dut, 10000)

    scoreboard.record_calibration(
        test_name="test_max_negative_weights",
        locked=locked,
        cycles=cycles,
    )

    dut._log.info(f"Max negative weights: {'locked' if locked else 'did not lock'}")

    adapter.stop()


@cocotb.test()
async def test_mixed_sign_weights(dut):
    """
    Test: Weights with mixed signs.
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    plant = PhotonicPlant(
        drift_rate=0.0,
        noise_std=0.0,
        tau_thermal=1e-9,
        seed=4003,
    )

    await reset_dut(dut)
    set_weights(dut, w0=0.9, w1=-0.9, w2=-0.9, w3=0.9)

    adapter = PlantInLoop(dut, plant)
    cocotb.start_soon(adapter.run())

    await start_calibration(dut)
    locked, cycles = await wait_for_calibration(dut, 10000)

    scoreboard.record_calibration(
        test_name="test_mixed_sign_weights",
        locked=locked,
        cycles=cycles,
    )

    dut._log.info(f"Mixed sign weights: {'locked' if locked else 'did not lock'}")

    adapter.stop()


@cocotb.test()
async def test_small_weights(dut):
    """
    Test: Very small weight values.

    Near-zero weights test numerical precision.
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    plant = PhotonicPlant(
        drift_rate=0.0,
        noise_std=0.0,
        tau_thermal=1e-9,
        seed=4004,
    )

    await reset_dut(dut)
    set_weights(dut, w0=0.01, w1=0.01, w2=-0.01, w3=0.01)

    adapter = PlantInLoop(dut, plant)
    cocotb.start_soon(adapter.run())

    await start_calibration(dut)
    locked, cycles = await wait_for_calibration(dut, 10000)

    scoreboard.record_calibration(
        test_name="test_small_weights",
        locked=locked,
        cycles=cycles,
    )

    dut._log.info(f"Small weights: {'locked' if locked else 'did not lock'}")

    adapter.stop()


@cocotb.test()
async def test_weight_validation_python():
    """
    Test: Python-side weight validation.

    This is a pure Python test (no DUT required).
    """
    # Valid weights should pass
    assert validate_weights(0.5, -0.5, 0.0, 1.0) is True
    assert validate_weights(-1.0, 1.0, 0.0, 0.0) is True

    # Out-of-range weights should raise ValueError
    try:
        validate_weights(1.5, 0.0, 0.0, 0.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "out of range" in str(e)

    try:
        validate_weights(0.0, 0.0, -1.5, 0.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "out of range" in str(e)


@cocotb.test()
async def test_fixed_point_conversion():
    """
    Test: Fixed-point conversion accuracy.

    Verifies Q1.15 conversions are correct.
    """
    from tests.utils.stimulus import q1_15_to_float

    # Test known values
    assert float_to_q1_15(0.0) == 0
    assert float_to_q1_15(0.5) == 16384  # 0.5 * 2^15
    assert float_to_q1_15(-0.5) == -16384
    assert float_to_q1_15(1.0 - 2**-15) == 32767  # Max positive

    # Test round-trip
    for val in [0.0, 0.5, -0.5, 0.25, -0.75, 0.123]:
        code = float_to_q1_15(val)
        recovered = q1_15_to_float(code)
        error = abs(recovered - val)
        assert error < 2**-14, f"Round-trip error too large for {val}: {error}"


@cocotb.test()
async def test_repeated_calibration(dut):
    """
    Test: Multiple calibration cycles.

    Tests that calibration can be restarted after completion.
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    plant = PhotonicPlant(
        drift_rate=0.0,
        noise_std=0.0,
        tau_thermal=1e-9,
        seed=4005,
    )

    for i in range(3):
        plant.reset()
        await reset_dut(dut)

        # Different weights each time
        weights = [
            (1.0, 0.0, 0.0, 1.0),  # Identity
            (0.0, 1.0, 1.0, 0.0),  # Swap
            (0.5, 0.5, 0.5, -0.5),  # Custom
        ][i]

        set_weights(dut, *weights)

        adapter = PlantInLoop(dut, plant)
        task = cocotb.start_soon(adapter.run())

        await start_calibration(dut)
        locked, cycles = await wait_for_calibration(dut, 8000)

        scoreboard.record_calibration(
            test_name=f"test_repeated_calibration_{i}",
            locked=locked,
            cycles=cycles,
        )

        adapter.stop()

        dut._log.info(f"Calibration {i}: {'locked' if locked else 'failed'} in {cycles} cycles")


@cocotb.test()
async def test_print_edge_case_summary(dut):
    """Print edge case test summary."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    scoreboard.print_summary()

    results_dir = Path(__file__).parent.parent / "sim" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    scoreboard.to_json(results_dir / "edge_case_results.json")
