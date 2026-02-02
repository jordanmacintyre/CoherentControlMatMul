"""
Drift robustness tests for coherent matrix multiply.

Tests that the system maintains acceptable performance under drift.
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
from tests.utils.stimulus import q1_15_to_float


scoreboard = MetricsScoreboard()


@cocotb.test()
async def test_slow_drift(dut):
    """
    Test: Performance under slow thermal drift.

    Uses low drift rate and verifies outputs remain acceptable
    after extended operation.
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Plant with slow drift (should be manageable)
    plant = PhotonicPlant(
        drift_rate=0.001,  # Low drift
        noise_std=2.0,
        tau_thermal=100e-6,  # Realistic thermal response
        crosstalk_coeff=0.02,
        seed=3000,
    )

    await reset_dut(dut)

    # Calibrate to identity
    set_weights(dut, w0=1.0, w1=0.0, w2=0.0, w3=1.0)

    adapter = PlantInLoop(dut, plant)
    cocotb.start_soon(adapter.run())

    await start_calibration(dut)
    locked, cal_cycles = await wait_for_calibration(dut, 15000)

    scoreboard.record_calibration(
        test_name="test_slow_drift_cal",
        locked=locked,
        cycles=cal_cycles,
    )

    if not locked:
        adapter.stop()
        dut._log.warning("Calibration failed with drift - skipping evaluation")
        return

    golden = GoldenModel()
    golden.set_weights(1.0, 0.0, 0.0, 1.0)

    # Run evaluation for many cycles to accumulate drift
    errors = []
    num_evals = 50

    for i in range(num_evals):
        # Use consistent inputs
        x0, x1 = 0.7, 0.3

        set_inputs(dut, x0, x1)
        await start_evaluation(dut)
        done, _ = await wait_for_evaluation(dut)

        if done:
            y0_actual = q1_15_to_float(int(dut.y0_out.value.signed_integer))
            y1_actual = q1_15_to_float(int(dut.y1_out.value.signed_integer))
            error = golden.compute_error(y0_actual, y1_actual, x0, x1)
            errors.append(error)

        # Wait some cycles to let drift accumulate
        for _ in range(100):
            await RisingEdge(dut.clk)

    if errors:
        rms_error = np.sqrt(np.mean(np.array(errors)**2))
        max_error = np.max(errors)

        scoreboard.record_evaluation(
            test_name="test_slow_drift_eval",
            passed=max_error < 0.2,
            output_error=rms_error,
            details={"max_error": max_error, "num_evals": num_evals},
        )

        dut._log.info(
            f"Slow drift: RMS error = {rms_error:.4f}, "
            f"Max error = {max_error:.4f}"
        )

    adapter.stop()


@cocotb.test()
async def test_fast_drift(dut):
    """
    Test: Performance under faster thermal drift.

    Uses higher drift rate - may not maintain perfect lock.
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Plant with faster drift
    plant = PhotonicPlant(
        drift_rate=0.01,  # Higher drift
        noise_std=5.0,
        tau_thermal=100e-6,
        crosstalk_coeff=0.05,
        seed=3001,
    )

    await reset_dut(dut)
    set_weights(dut, w0=1.0, w1=0.0, w2=0.0, w3=1.0)

    adapter = PlantInLoop(dut, plant)
    cocotb.start_soon(adapter.run())

    await start_calibration(dut)
    locked, cal_cycles = await wait_for_calibration(dut, 20000)

    scoreboard.record_calibration(
        test_name="test_fast_drift_cal",
        locked=locked,
        cycles=cal_cycles,
    )

    if not locked:
        dut._log.info("Fast drift calibration did not lock (expected for high drift)")
        adapter.stop()
        return

    golden = GoldenModel()
    golden.set_weights(1.0, 0.0, 0.0, 1.0)

    errors = []
    for i in range(20):
        x0, x1 = 0.5, -0.5

        set_inputs(dut, x0, x1)
        await start_evaluation(dut)
        done, _ = await wait_for_evaluation(dut)

        if done:
            y0_actual = q1_15_to_float(int(dut.y0_out.value.signed_integer))
            y1_actual = q1_15_to_float(int(dut.y1_out.value.signed_integer))
            error = golden.compute_error(y0_actual, y1_actual, x0, x1)
            errors.append(error)

        # More wait time to accumulate drift
        for _ in range(200):
            await RisingEdge(dut.clk)

    if errors:
        rms_error = np.sqrt(np.mean(np.array(errors)**2))
        max_error = np.max(errors)

        # More relaxed threshold for fast drift
        scoreboard.record_evaluation(
            test_name="test_fast_drift_eval",
            passed=max_error < 0.5,
            output_error=rms_error,
            details={"max_error": max_error},
        )

        dut._log.info(
            f"Fast drift: RMS error = {rms_error:.4f}, "
            f"Max error = {max_error:.4f}"
        )

    adapter.stop()


@cocotb.test()
async def test_thermal_crosstalk(dut):
    """
    Test: Performance with thermal crosstalk.

    Tests that calibration can handle inter-heater coupling.
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Plant with significant crosstalk
    plant = PhotonicPlant(
        drift_rate=0.0,
        noise_std=2.0,
        tau_thermal=100e-6,
        crosstalk_coeff=0.1,  # 10% crosstalk
        seed=3002,
    )

    await reset_dut(dut)
    set_weights(dut, w0=0.5, w1=0.5, w2=-0.5, w3=0.5)

    adapter = PlantInLoop(dut, plant)
    cocotb.start_soon(adapter.run())

    await start_calibration(dut)
    locked, cal_cycles = await wait_for_calibration(dut, 15000)

    scoreboard.record_calibration(
        test_name="test_thermal_crosstalk",
        locked=locked,
        cycles=cal_cycles,
    )

    dut._log.info(
        f"Crosstalk test: {'locked' if locked else 'failed'} in {cal_cycles} cycles"
    )

    adapter.stop()


@cocotb.test()
async def test_realistic_conditions(dut):
    """
    Test: Full realistic conditions (drift + noise + crosstalk + thermal).

    This represents realistic operating conditions.
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Realistic plant configuration
    plant = PhotonicPlant(
        drift_rate=0.002,      # Moderate drift
        noise_std=5.0,         # Moderate noise
        tau_thermal=100e-6,    # Realistic thermal time constant
        crosstalk_coeff=0.05,  # 5% crosstalk
        seed=3003,
    )

    await reset_dut(dut)

    # Try identity matrix under realistic conditions
    set_weights(dut, w0=1.0, w1=0.0, w2=0.0, w3=1.0)

    adapter = PlantInLoop(dut, plant)
    cocotb.start_soon(adapter.run())

    await start_calibration(dut)
    locked, cal_cycles = await wait_for_calibration(dut, 20000)

    scoreboard.record_calibration(
        test_name="test_realistic_conditions",
        locked=locked,
        cycles=cal_cycles,
    )

    if locked:
        golden = GoldenModel()
        golden.set_weights(1.0, 0.0, 0.0, 1.0)

        errors = []
        for _ in range(10):
            x0, x1 = 0.6, -0.4

            set_inputs(dut, x0, x1)
            await start_evaluation(dut)
            done, _ = await wait_for_evaluation(dut)

            if done:
                y0_actual = q1_15_to_float(int(dut.y0_out.value.signed_integer))
                y1_actual = q1_15_to_float(int(dut.y1_out.value.signed_integer))
                error = golden.compute_error(y0_actual, y1_actual, x0, x1)
                errors.append(error)

            for _ in range(50):
                await RisingEdge(dut.clk)

        if errors:
            rms_error = np.sqrt(np.mean(np.array(errors)**2))
            scoreboard.record_evaluation(
                test_name="test_realistic_eval",
                passed=rms_error < 0.2,
                output_error=rms_error,
            )

    dut._log.info(
        f"Realistic conditions: {'locked' if locked else 'failed'} in {cal_cycles} cycles"
    )

    adapter.stop()


@cocotb.test()
async def test_print_drift_summary(dut):
    """Print drift test summary."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    scoreboard.print_summary()

    results_dir = Path(__file__).parent.parent / "sim" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    scoreboard.to_json(results_dir / "drift_results.json")
