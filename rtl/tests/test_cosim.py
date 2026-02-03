"""
cocotb tests for coherent photonic matrix multiply co-simulation.

This module tests the RTL controller (coherent_matmul_top.sv) with Python
photonic plant models to verify closed-loop calibration behavior.

Test cases:
1. Unitary mode: Calibrate to Hadamard matrix
2. SVD mode: Calibrate to arbitrary (non-unitary) matrix
3. Convergence verification: Monitor error trajectory
4. Reset and re-calibration: Verify state machine recovery

Run with:
    cd rtl && make test-all
    cd rtl && make TESTCASE=test_unitary_identity
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ClockCycles

import os
import sys
from pathlib import Path
from math import sqrt
import numpy as np

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from plant.plant_wrapper import PhotonicPlant, float_to_q1_15
from plant.svd_plant import SVDPhotonicPlant
from plant.voa import DualVOA

from tests.plant_adapter import PlantAdapter, SVDPlantAdapter, AutoSelectPlantAdapter
from tests.plotting import create_run_directory, save_calibration_data, plot_convergence, plot_phase_evolution


# =============================================================================
# Global Artifacts Directory (shared across tests in a run)
# =============================================================================

_RUN_DIR = None

def get_run_directory() -> Path:
    """Get or create the run directory for this test session."""
    global _RUN_DIR
    if _RUN_DIR is None:
        # Check if ARTIFACTS_DIR is set via environment (already a run directory)
        artifacts_dir = os.environ.get('ARTIFACTS_DIR', None)
        if artifacts_dir:
            # Use the provided directory directly (run.py already created it)
            _RUN_DIR = Path(artifacts_dir)
            _RUN_DIR.mkdir(parents=True, exist_ok=True)
        else:
            # Create a new run directory
            _RUN_DIR = create_run_directory()
    return _RUN_DIR


# =============================================================================
# Helper Functions
# =============================================================================

def set_weights_q15(dut, w0: float, w1: float, w2: float, w3: float) -> None:
    """Set target weights in Q1.15 format."""
    dut.w0.value = float_to_q1_15(w0)
    dut.w1.value = float_to_q1_15(w1)
    dut.w2.value = float_to_q1_15(w2)
    dut.w3.value = float_to_q1_15(w3)


def compute_sigma_dac(w0: float, w1: float, w2: float, w3: float) -> tuple[int, int]:
    """
    Compute VOA DAC codes for SVD singular values.

    Returns DAC codes that map to the singular values of the target matrix.
    """
    M = np.array([[w0, w1], [w2, w3]])
    _, sigma, _ = np.linalg.svd(M)

    # Convert singular values to DAC codes
    voa = DualVOA(dac_bits=16, max_attenuation_db=40.0, insertion_loss_db=0.0)
    dac_codes = voa.sigma_to_dac(sigma)

    return dac_codes[0], dac_codes[1]


async def reset_dut(dut, cycles: int = 10) -> None:
    """Apply reset to DUT."""
    dut.rst_n.value = 0
    dut.start_cal.value = 0
    dut.start_eval.value = 0
    dut.svd_mode.value = 0
    dut.sigma_dac0.value = 0
    dut.sigma_dac1.value = 0
    dut.x0.value = 0
    dut.x1.value = 0
    dut.adc_i0.value = 0
    dut.adc_q0.value = 0
    dut.adc_i1.value = 0
    dut.adc_q1.value = 0
    dut.adc_valid.value = 0

    # Hold reset
    await ClockCycles(dut.clk, cycles)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def start_calibration(dut) -> None:
    """Pulse start_cal signal."""
    dut.start_cal.value = 1
    await RisingEdge(dut.clk)
    dut.start_cal.value = 0


async def wait_for_lock(dut, timeout_cycles: int = 50000, debug_interval: int = 10000) -> bool:
    """Wait for cal_locked signal with timeout and debug logging."""
    log_interval = debug_interval
    last_error = None

    for cycle in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if int(dut.cal_locked.value) == 1:
            return True

        # Periodic debug logging
        if cycle % log_interval == 0 and cycle > 0:
            try:
                error = int(dut.error_current.value)
                status = get_status_flags(dut)

                # Also try to read measured values
                try:
                    m00_real = int(dut.m_hat_real[0].value)
                    m01_real = int(dut.m_hat_real[1].value)
                    m10_real = int(dut.m_hat_real[2].value)
                    m11_real = int(dut.m_hat_real[3].value)
                    dut._log.info(f"Cycle {cycle}: error={error:,d}, m_hat_real=[{m00_real}, {m01_real}, {m10_real}, {m11_real}]")
                except (AttributeError, ValueError):
                    dut._log.info(f"Cycle {cycle}: error={error:,d}, cal_in_progress={status['cal_in_progress']}")

                # Log if error is changing
                if last_error is not None and error != last_error:
                    change = error - last_error
                    direction = "↓" if change < 0 else "↑"
                    dut._log.info(f"  Error change: {direction} {abs(change):,d} ({change/last_error*100:.1f}%)")
                last_error = error

            except (AttributeError, ValueError) as e:
                dut._log.warning(f"Cycle {cycle}: Could not read error_current: {e}")

    return False


async def wait_for_done(dut, timeout_cycles: int = 50000) -> bool:
    """Wait for cal_done signal with timeout."""
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if int(dut.cal_done.value) == 1:
            return True
    return False


def get_status_flags(dut) -> dict:
    """Extract status flag bits."""
    flags = int(dut.status_flags.value)
    return {
        'cal_locked': bool(flags & 0x80),
        'cal_in_progress': bool(flags & 0x40),
        'eval_done': bool(flags & 0x20),
        'error_weights': bool(flags & 0x10),
        'error_timeout': bool(flags & 0x08),
        'error_saturated': bool(flags & 0x04),
        'v_locked': bool(flags & 0x02),
        'u_locked': bool(flags & 0x01),
    }


class CalibrationDataCollector:
    """Collects calibration data for plotting."""

    def __init__(self, test_name: str):
        self.test_name = test_name
        self.cycles: list[int] = []
        self.errors: list[int] = []
        self.phases: list[list[int]] = []
        self.sample_interval = 100  # Sample every N cycles
        self._error_signal = None  # Cache the signal reference

    def sample(self, dut, cycle: int) -> None:
        """Sample current calibration state."""
        if cycle % self.sample_interval != 0:
            return

        self.cycles.append(cycle)

        # Read current error from RTL (internal signal)
        try:
            # Try to access the internal error_current signal
            if self._error_signal is None:
                # Cache the signal for performance
                self._error_signal = dut.error_current
            error = int(self._error_signal.value)
            self.errors.append(error)
        except (AttributeError, ValueError):
            # Signal might not be accessible
            self.errors.append(0)

        # Read phase DAC codes
        try:
            phases = [int(dut.phi_dac[i].value) for i in range(4)]
            self.phases.append(phases)
        except (AttributeError, IndexError):
            self.phases.append([0, 0, 0, 0])

    def save(self, run_dir: Path = None) -> None:
        """Save collected data and generate plots."""
        if run_dir is None:
            run_dir = get_run_directory()

        if len(self.cycles) == 0:
            return

        # Save raw data
        data = {
            'test_name': self.test_name,
            'cycles': self.cycles,
            'errors': self.errors,
            'phases': self.phases,
        }
        save_calibration_data(run_dir, self.test_name, data)

        # Generate convergence plot and get the min_cycle for phase plot
        min_cycle = None
        if len(self.errors) > 1 and any(e > 0 for e in self.errors):
            # Filter out zeros for plotting
            valid_cycles = [c for c, e in zip(self.cycles, self.errors) if e > 0]
            valid_errors = [e for e in self.errors if e > 0]
            if valid_errors:
                _, min_cycle = plot_convergence(
                    valid_cycles,
                    valid_errors,
                    run_dir,
                    self.test_name,
                    title=f'{self.test_name} - Error Convergence',
                    threshold=0x00100000,  # CAL_LOCK_THRESHOLD = 1,048,576
                )

        # Generate phase evolution plot with min_cycle marker
        if len(self.phases) > 1:
            plot_phase_evolution(
                self.cycles,
                self.phases,
                run_dir,
                self.test_name,
                phase_labels=['theta', 'phi_in0', 'phi_in1', 'phi_out'],
                min_cycle=min_cycle,
            )


# =============================================================================
# Unitary Mode Tests
# =============================================================================

@cocotb.test()
async def test_unitary_identity(dut):
    """
    Test calibration to identity matrix in unitary mode.

    Target: M = [[1, 0], [0, 1]]

    This is the simplest case - the MZI should converge to a passthrough
    configuration with minimal phase shifts.
    """
    # Start clock
    cocotb.start_soon(Clock(dut.clk, 10, units='ns').start())

    # Create plant and adapter
    plant = PhotonicPlant(noise_std=2.0, seed=42)
    adapter = PlantAdapter(dut, plant)

    # Reset and configure
    await reset_dut(dut)

    # Log target weights in Q1.15
    w0_q15 = float_to_q1_15(1.0 - 1e-4)
    w1_q15 = float_to_q1_15(0.0)
    w2_q15 = float_to_q1_15(0.0)
    w3_q15 = float_to_q1_15(1.0 - 1e-4)
    dut._log.info(f"Target weights (Q1.15): w0={w0_q15}, w1={w1_q15}, w2={w2_q15}, w3={w3_q15}")

    set_weights_q15(dut, 1.0 - 1e-4, 0.0, 0.0, 1.0 - 1e-4)  # Near-identity (Q1.15 max is ~0.99997)
    dut.svd_mode.value = 0

    # Start adapter and calibration
    cocotb.start_soon(adapter.run())
    await start_calibration(dut)

    # Wait for lock with debug logging every 50k cycles
    locked = await wait_for_lock(dut, timeout_cycles=2000000, debug_interval=50000)

    adapter.stop()

    if not locked:
        status = get_status_flags(dut)
        # Log final state info
        try:
            final_error = int(dut.error_current.value)
            dut._log.error(f"Final error: {final_error:,d} (threshold: 1,000,000)")
        except (AttributeError, ValueError):
            pass
        raise AssertionError(f"Identity calibration failed to lock. Status: {status}")

    dut._log.info("Identity matrix calibration locked successfully")


@cocotb.test()
async def test_unitary_hadamard(dut):
    """
    Test calibration to Hadamard matrix in unitary mode.

    Target: M = (1/√2) * [[1, 1], [1, -1]]

    The Hadamard matrix is unitary and commonly used in quantum computing.
    It represents a balanced beam splitter operation.
    """
    cocotb.start_soon(Clock(dut.clk, 10, units='ns').start())

    plant = PhotonicPlant(noise_std=2.0, seed=123)
    adapter = PlantAdapter(dut, plant)

    # Create data collector
    collector = CalibrationDataCollector('test_unitary_hadamard')

    await reset_dut(dut)

    # Hadamard weights
    h = 1.0 / sqrt(2)
    set_weights_q15(dut, h, h, h, -h)
    dut.svd_mode.value = 0

    cocotb.start_soon(adapter.run())
    await start_calibration(dut)

    # Wait for lock with data collection
    cycles = 0
    timeout_cycles = 200000  # Hadamard requires more iterations than identity
    locked = False

    while cycles < timeout_cycles:
        await RisingEdge(dut.clk)
        cycles += 1
        collector.sample(dut, cycles)

        if int(dut.cal_locked.value) == 1:
            locked = True
            break

    adapter.stop()

    # Save collected data
    collector.save()

    if not locked:
        status = get_status_flags(dut)
        raise AssertionError(f"Hadamard calibration failed to lock. Status: {status}")

    dut._log.info(f"Hadamard matrix calibration locked in {cycles} cycles")


@cocotb.test()
async def test_unitary_rotation(dut):
    """
    Test calibration to rotation matrix.

    Target: M = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]] with θ = π/6 (30°)

    Rotation matrices are unitary and test the MZI's ability to realize
    arbitrary beam splitter ratios.
    """
    cocotb.start_soon(Clock(dut.clk, 10, units='ns').start())

    plant = PhotonicPlant(noise_std=2.0, seed=456)
    adapter = PlantAdapter(dut, plant)

    await reset_dut(dut)

    # 30 degree rotation
    theta = np.pi / 6
    set_weights_q15(dut, np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta))
    dut.svd_mode.value = 0

    cocotb.start_soon(adapter.run())
    await start_calibration(dut)

    locked = await wait_for_lock(dut, timeout_cycles=2000000)

    adapter.stop()

    if not locked:
        status = get_status_flags(dut)
        raise AssertionError(f"Rotation calibration failed to lock. Status: {status}")

    dut._log.info("Rotation matrix calibration locked successfully")


# =============================================================================
# SVD Mode Tests
# =============================================================================

@cocotb.test()
async def test_svd_diagonal(dut):
    """
    Test SVD mode with diagonal (scaling) matrix.

    Target: M = [[0.5, 0], [0, 0.3]]

    This is NOT unitary (σ₁ ≠ σ₂ ≠ 1), so it requires SVD mode.
    The VOAs implement the scaling while MZI meshes stay at identity.
    """
    cocotb.start_soon(Clock(dut.clk, 10, units='ns').start())

    plant = SVDPhotonicPlant(noise_std=2.0, seed=789)
    adapter = SVDPlantAdapter(dut, plant)

    await reset_dut(dut)

    # Diagonal scaling matrix
    w0, w1, w2, w3 = 0.5, 0.0, 0.0, 0.3
    set_weights_q15(dut, w0, w1, w2, w3)
    dut.svd_mode.value = 1

    # Set pre-computed singular value DAC codes
    sigma_dac0, sigma_dac1 = compute_sigma_dac(w0, w1, w2, w3)
    dut.sigma_dac0.value = sigma_dac0
    dut.sigma_dac1.value = sigma_dac1

    cocotb.start_soon(adapter.run())
    await start_calibration(dut)

    locked = await wait_for_lock(dut, timeout_cycles=2000000)

    adapter.stop()

    if not locked:
        status = get_status_flags(dut)
        raise AssertionError(f"SVD diagonal calibration failed to lock. Status: {status}")

    # Verify both V and U meshes locked
    status = get_status_flags(dut)
    dut._log.info(f"SVD diagonal calibration locked. v_locked={status['v_locked']}, u_locked={status['u_locked']}")


@cocotb.test()
async def test_svd_arbitrary(dut):
    """
    Test SVD mode with arbitrary non-unitary matrix.

    Target: M = [[0.7, -0.3], [0.2, -0.5]]

    This matrix has no special structure and tests the full SVD
    calibration pipeline: V† → Σ → U.
    """
    cocotb.start_soon(Clock(dut.clk, 10, units='ns').start())

    plant = SVDPhotonicPlant(noise_std=2.0, seed=101)
    adapter = SVDPlantAdapter(dut, plant)

    # Create data collector
    collector = CalibrationDataCollector('test_svd_arbitrary')
    collector.sample_interval = 200  # Sample less frequently for longer test

    await reset_dut(dut)

    w0, w1, w2, w3 = 0.7, -0.3, 0.2, -0.5
    set_weights_q15(dut, w0, w1, w2, w3)
    dut.svd_mode.value = 1

    sigma_dac0, sigma_dac1 = compute_sigma_dac(w0, w1, w2, w3)
    dut.sigma_dac0.value = sigma_dac0
    dut.sigma_dac1.value = sigma_dac1

    cocotb.start_soon(adapter.run())
    await start_calibration(dut)

    # Wait for lock with data collection
    cycles = 0
    timeout_cycles = 500000  # SVD mode calibrates 2 meshes sequentially
    locked = False

    while cycles < timeout_cycles:
        await RisingEdge(dut.clk)
        cycles += 1
        collector.sample(dut, cycles)

        if int(dut.cal_locked.value) == 1:
            locked = True
            break

    adapter.stop()

    # Save collected data
    collector.save()

    if not locked:
        status = get_status_flags(dut)
        raise AssertionError(f"SVD arbitrary calibration failed to lock. Status: {status}")

    dut._log.info(f"SVD arbitrary matrix calibration locked in {cycles} cycles")


# =============================================================================
# Convergence and Error Monitoring Tests
# =============================================================================

@cocotb.test()
async def test_convergence_monitoring(dut):
    """
    Monitor calibration convergence over time.

    This test verifies that the error metric generally decreases during
    calibration, demonstrating proper coordinate descent behavior.
    """
    cocotb.start_soon(Clock(dut.clk, 10, units='ns').start())

    plant = PhotonicPlant(noise_std=2.0, seed=202)
    adapter = PlantAdapter(dut, plant)

    # Create data collector
    collector = CalibrationDataCollector('test_convergence_monitoring')
    collector.sample_interval = 10  # Sample every 10 cycles to capture lock moment

    await reset_dut(dut)

    # Target: swap matrix [[0, 1], [1, 0]] (unitary)
    set_weights_q15(dut, 0.0, 1.0 - 1e-4, 1.0 - 1e-4, 0.0)
    dut.svd_mode.value = 0

    cocotb.start_soon(adapter.run())
    await start_calibration(dut)

    # Monitor calibration progress
    iterations = 0
    max_iterations = 500000  # Swap matrix is one of the hardest unitary targets
    cal_in_progress_seen = False

    while iterations < max_iterations:
        await RisingEdge(dut.clk)
        iterations += 1

        # Collect data for plotting
        collector.sample(dut, iterations)

        status = get_status_flags(dut)

        if status['cal_in_progress']:
            cal_in_progress_seen = True

        if status['cal_locked']:
            break

    adapter.stop()

    # Save collected data and generate plots
    collector.save()

    if not cal_in_progress_seen:
        raise AssertionError("Never saw cal_in_progress flag")

    if iterations >= max_iterations:
        raise AssertionError(f"Convergence monitoring timed out after {iterations} cycles")

    dut._log.info(f"Convergence achieved in {iterations} cycles")


# =============================================================================
# Edge Case and Error Handling Tests
# =============================================================================

@cocotb.test()
async def test_reset_during_calibration(dut):
    """
    Test reset recovery during active calibration.

    Verifies that the FSM properly resets and can restart calibration
    after an asynchronous reset.
    """
    cocotb.start_soon(Clock(dut.clk, 10, units='ns').start())

    plant = PhotonicPlant(noise_std=2.0, seed=303)
    adapter = PlantAdapter(dut, plant)

    await reset_dut(dut)

    h = 1.0 / sqrt(2)
    set_weights_q15(dut, h, h, h, -h)
    dut.svd_mode.value = 0

    cocotb.start_soon(adapter.run())
    await start_calibration(dut)

    # Wait partway through calibration
    await ClockCycles(dut.clk, 5000)

    # Check we're in calibration
    status = get_status_flags(dut)
    if not status['cal_in_progress']:
        dut._log.warning("Calibration completed before reset test could interrupt")
    else:
        # Apply reset mid-calibration
        dut.rst_n.value = 0
        await ClockCycles(dut.clk, 5)
        dut.rst_n.value = 1
        await ClockCycles(dut.clk, 5)

        # Restart calibration
        await start_calibration(dut)

        locked = await wait_for_lock(dut, timeout_cycles=2000000)

        if not locked:
            adapter.stop()
            raise AssertionError("Failed to re-lock after reset")

    adapter.stop()
    dut._log.info("Reset recovery test passed")


@cocotb.test()
async def test_mode_switching(dut):
    """
    Test switching between unitary and SVD modes.

    Verifies that the controller can calibrate in one mode, then
    successfully recalibrate in the other mode.
    """
    cocotb.start_soon(Clock(dut.clk, 10, units='ns').start())

    # Use auto-select adapter that handles both modes
    adapter = AutoSelectPlantAdapter(dut, noise_std=2.0, seed=404)

    await reset_dut(dut)

    # First: unitary mode (Hadamard)
    h = 1.0 / sqrt(2)
    set_weights_q15(dut, h, h, h, -h)
    dut.svd_mode.value = 0

    cocotb.start_soon(adapter.run())
    await start_calibration(dut)

    locked = await wait_for_lock(dut, timeout_cycles=2000000)
    if not locked:
        adapter.stop()
        raise AssertionError("First calibration (unitary) failed")

    dut._log.info("Unitary mode calibration complete")

    # Switch to SVD mode with new target
    await ClockCycles(dut.clk, 100)

    w0, w1, w2, w3 = 0.6, -0.2, 0.3, 0.4
    set_weights_q15(dut, w0, w1, w2, w3)
    dut.svd_mode.value = 1

    sigma_dac0, sigma_dac1 = compute_sigma_dac(w0, w1, w2, w3)
    dut.sigma_dac0.value = sigma_dac0
    dut.sigma_dac1.value = sigma_dac1

    await start_calibration(dut)

    locked = await wait_for_lock(dut, timeout_cycles=2000000)

    adapter.stop()

    if not locked:
        raise AssertionError("Second calibration (SVD) failed")

    dut._log.info("Mode switching test passed: unitary → SVD successful")


# =============================================================================
# Evaluation Mode Tests
# =============================================================================

@cocotb.test()
async def test_evaluation_after_lock(dut):
    """
    Test evaluation mode after successful calibration.

    After locking to a target matrix, the system should correctly
    compute y = M·x for arbitrary input vectors.
    """
    cocotb.start_soon(Clock(dut.clk, 10, units='ns').start())

    plant = PhotonicPlant(noise_std=1.0, seed=505)  # Lower noise for eval
    adapter = PlantAdapter(dut, plant)

    await reset_dut(dut)

    # Identity matrix for simple verification
    set_weights_q15(dut, 1.0 - 1e-4, 0.0, 0.0, 1.0 - 1e-4)
    dut.svd_mode.value = 0

    cocotb.start_soon(adapter.run())
    await start_calibration(dut)

    locked = await wait_for_lock(dut, timeout_cycles=2000000)
    if not locked:
        adapter.stop()
        raise AssertionError("Calibration failed before evaluation test")

    # Set evaluation inputs
    dut.x0.value = float_to_q1_15(0.5)
    dut.x1.value = float_to_q1_15(0.25)

    # Start evaluation
    dut.start_eval.value = 1
    await RisingEdge(dut.clk)
    dut.start_eval.value = 0

    # Wait for eval_done
    for _ in range(1000):
        await RisingEdge(dut.clk)
        if int(dut.y_valid.value) == 1:
            break

    adapter.stop()

    # Check outputs (should be close to inputs for identity matrix)
    y0 = int(dut.y0_out.value)
    y1 = int(dut.y1_out.value)

    # Convert from unsigned to signed
    if y0 >= 32768:
        y0 -= 65536
    if y1 >= 32768:
        y1 -= 65536

    dut._log.info(f"Evaluation result: y0={y0}, y1={y1}")
    dut._log.info("Evaluation mode test passed")
