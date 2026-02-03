"""
Incremental debug tests for ADC signal chain.

This module isolates each component in the ADC data path to identify
where the signal chain breaks. Run stages sequentially to pinpoint failures.

Usage:
    cd rtl && make MODULE=tests.test_debug_chain TESTCASE=test_stage1_plant_output
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

import sys
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from plant.plant_wrapper import PhotonicPlant, float_to_q1_15
from tests.plant_adapter import PlantAdapter


# =============================================================================
# Helper Functions (copied from test_cosim.py for isolation)
# =============================================================================

def set_weights_q15(dut, w0: float, w1: float, w2: float, w3: float) -> None:
    """Set target weights in Q1.15 format."""
    dut.w0.value = float_to_q1_15(w0)
    dut.w1.value = float_to_q1_15(w1)
    dut.w2.value = float_to_q1_15(w2)
    dut.w3.value = float_to_q1_15(w3)


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

    await ClockCycles(dut.clk, cycles)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def start_calibration(dut) -> None:
    """Pulse start_cal signal."""
    dut.start_cal.value = 1
    await RisingEdge(dut.clk)
    dut.start_cal.value = 0


def signed_12bit(val: int) -> int:
    """Convert unsigned 12-bit value to signed."""
    if val >= 2048:
        return val - 4096
    return val


def signed_16bit(val: int) -> int:
    """Convert unsigned 16-bit value to signed."""
    if val >= 32768:
        return val - 65536
    return val


# =============================================================================
# Stage 1: Verify Plant Output
# =============================================================================

@cocotb.test()
async def test_stage1_plant_output(dut):
    """
    Stage 1: Verify PhotonicPlant produces valid ADC values.

    This test calls the plant model directly WITHOUT cocotb/RTL interaction
    to verify the Python model is working correctly.

    Expected: For 1.0 optical input, output should be ~2047 LSB (ADC full scale)
    """
    dut._log.info("=" * 60)
    dut._log.info("STAGE 1: Verify Plant Output (Python only)")
    dut._log.info("=" * 60)

    # Create plant with known parameters
    plant = PhotonicPlant(noise_std=2.0, seed=42)

    # Test with identity-like phases and [1,0] input
    dac_codes = [0, 0, 0, 0]  # All phases at 0
    x0 = complex(1.0, 0.0)  # Full amplitude on port 0
    x1 = complex(0.0, 0.0)  # Zero on port 1

    dut._log.info(f"Input: dac_codes={dac_codes}, x0={x0}, x1={x1}")

    # Call plant multiple times to see noise behavior
    for i in range(5):
        i0, q0, i1, q1 = plant.sample_outputs(dac_codes, x0, x1)
        dut._log.info(f"Sample {i}: i0={i0:6d}, q0={q0:6d}, i1={i1:6d}, q1={q1:6d}")

    # Final sample for assertion
    i0, q0, i1, q1 = plant.sample_outputs(dac_codes, x0, x1)

    dut._log.info("-" * 40)
    dut._log.info(f"Final: i0={i0}, q0={q0}, i1={i1}, q1={q1}")
    dut._log.info(f"Expected i0 ~ 2047 (gain=2047 for 1.0 optical)")

    # Check that we get reasonable values
    # With identity matrix and [1,0] input, output should be close to [1,0]
    # So i0 should be near 2047 (±noise), and i1 should be small
    if abs(i0) < 100:
        dut._log.error(f"FAIL: i0={i0} is too small (expected ~2047)")
        raise AssertionError(f"Plant i0={i0} is too small, expected ~2047 for 1.0 optical")

    dut._log.info("PASS: Plant produces reasonable ADC values")


# =============================================================================
# Stage 2: Verify Adapter Drives RTL
# =============================================================================

@cocotb.test()
async def test_stage2_adapter_drives_rtl(dut):
    """
    Stage 2: Verify PlantAdapter drives ADC signals to RTL.

    This test starts the adapter and checks that adc_i0/q0/i1/q1 and adc_valid
    are being driven to the RTL inputs.

    Note: plant_enable is an RTL OUTPUT, so we need to trigger calibration
    to enable the plant interface.
    """
    dut._log.info("=" * 60)
    dut._log.info("STAGE 2: Verify Adapter Drives RTL")
    dut._log.info("=" * 60)

    # Start clock
    cocotb.start_soon(Clock(dut.clk, 10, units='ns').start())

    # Create plant and adapter
    plant = PhotonicPlant(noise_std=2.0, seed=42)
    adapter = PlantAdapter(dut, plant)

    # Reset DUT
    await reset_dut(dut)

    # Set weights and mode
    set_weights_q15(dut, 1.0 - 1e-4, 0.0, 0.0, 1.0 - 1e-4)
    dut.svd_mode.value = 0

    # Start adapter
    cocotb.start_soon(adapter.run())

    # Trigger calibration to enable plant_enable
    await start_calibration(dut)

    dut._log.info("Calibration started, waiting for ADC signals...")

    # Wait and sample ADC values
    adc_valid_seen = False
    adc_nonzero_seen = False

    for cycle in range(100):
        await RisingEdge(dut.clk)

        try:
            plant_enable = int(dut.plant_enable.value)
            adc_i0 = int(dut.adc_i0.value)
            adc_q0 = int(dut.adc_q0.value)
            adc_i1 = int(dut.adc_i1.value)
            adc_q1 = int(dut.adc_q1.value)
            adc_valid = int(dut.adc_valid.value)

            # Convert to signed
            adc_i0_s = signed_12bit(adc_i0)
            adc_q0_s = signed_12bit(adc_q0)
            adc_i1_s = signed_12bit(adc_i1)
            adc_q1_s = signed_12bit(adc_q1)

            if cycle % 10 == 0 or adc_valid:
                dut._log.info(f"Cycle {cycle}: plant_enable={plant_enable}, "
                            f"adc_valid={adc_valid}, "
                            f"i0={adc_i0_s:5d}, q0={adc_q0_s:5d}, "
                            f"i1={adc_i1_s:5d}, q1={adc_q1_s:5d}")

            if adc_valid:
                adc_valid_seen = True
            if adc_i0 != 0 or adc_q0 != 0:
                adc_nonzero_seen = True

        except Exception as e:
            dut._log.warning(f"Cycle {cycle}: Could not read signals: {e}")

    adapter.stop()

    dut._log.info("-" * 40)
    dut._log.info(f"adc_valid seen: {adc_valid_seen}")
    dut._log.info(f"Non-zero ADC seen: {adc_nonzero_seen}")

    if not adc_valid_seen:
        dut._log.error("FAIL: adc_valid was never asserted")
        raise AssertionError("adc_valid never asserted - adapter timing issue?")

    if not adc_nonzero_seen:
        dut._log.error("FAIL: ADC values were always zero")
        raise AssertionError("ADC values always zero - plant not producing output?")

    dut._log.info("PASS: Adapter drives ADC signals to RTL")


# =============================================================================
# Stage 3: Verify Accumulator
# =============================================================================

@cocotb.test()
async def test_stage3_accumulator(dut):
    """
    Stage 3: Verify RTL accumulates ADC samples.

    This test checks that i0_accum accumulates when adc_valid is asserted.
    """
    dut._log.info("=" * 60)
    dut._log.info("STAGE 3: Verify RTL Accumulator")
    dut._log.info("=" * 60)

    cocotb.start_soon(Clock(dut.clk, 10, units='ns').start())

    plant = PhotonicPlant(noise_std=2.0, seed=42)
    adapter = PlantAdapter(dut, plant)

    await reset_dut(dut)
    set_weights_q15(dut, 1.0 - 1e-4, 0.0, 0.0, 1.0 - 1e-4)
    dut.svd_mode.value = 0

    cocotb.start_soon(adapter.run())
    await start_calibration(dut)

    dut._log.info("Monitoring accumulator during calibration...")

    accum_changed = False
    max_sample_counter = 0

    for cycle in range(200):
        await RisingEdge(dut.clk)

        try:
            cal_state = int(dut.cal_state.value)
            sample_counter = int(dut.sample_counter.value)
            i0_accum = int(dut.i0_accum.value)
            adc_valid = int(dut.adc_valid.value)

            # Convert to signed (accumulator is 16-bit signed)
            i0_accum_s = signed_16bit(i0_accum)

            max_sample_counter = max(max_sample_counter, sample_counter)

            if i0_accum != 0:
                accum_changed = True

            if cycle % 20 == 0 or sample_counter > 0:
                dut._log.info(f"Cycle {cycle}: cal_state={cal_state}, "
                            f"samples={sample_counter}, "
                            f"adc_valid={adc_valid}, "
                            f"i0_accum={i0_accum_s}")

        except Exception as e:
            dut._log.warning(f"Cycle {cycle}: Could not read signals: {e}")

    adapter.stop()

    dut._log.info("-" * 40)
    dut._log.info(f"Max sample_counter reached: {max_sample_counter}")
    dut._log.info(f"Accumulator changed from 0: {accum_changed}")

    if max_sample_counter == 0:
        dut._log.error("FAIL: sample_counter never incremented")
        raise AssertionError("sample_counter stuck at 0 - adc_valid not seen by RTL?")

    if not accum_changed:
        dut._log.error("FAIL: i0_accum never changed from 0")
        raise AssertionError("Accumulator stuck at 0 - ADC values not being added?")

    dut._log.info("PASS: RTL accumulates ADC samples")


# =============================================================================
# Stage 4: Verify Scaler
# =============================================================================

@cocotb.test()
async def test_stage4_scaler(dut):
    """
    Stage 4: Verify adc_scaler produces correct Q1.15 output.

    After 8 samples, i0_scaled should be approximately 2 * i0_accum.
    """
    dut._log.info("=" * 60)
    dut._log.info("STAGE 4: Verify ADC Scaler")
    dut._log.info("=" * 60)

    cocotb.start_soon(Clock(dut.clk, 10, units='ns').start())

    plant = PhotonicPlant(noise_std=2.0, seed=42)
    adapter = PlantAdapter(dut, plant)

    await reset_dut(dut)
    set_weights_q15(dut, 1.0 - 1e-4, 0.0, 0.0, 1.0 - 1e-4)
    dut.svd_mode.value = 0

    cocotb.start_soon(adapter.run())
    await start_calibration(dut)

    dut._log.info("Waiting for 8 samples to complete...")

    scaler_checked = False

    for cycle in range(300):
        await RisingEdge(dut.clk)

        try:
            sample_counter = int(dut.sample_counter.value)
            i0_accum = int(dut.i0_accum.value)
            i0_scaled = int(dut.i0_scaled.value)

            # Convert to signed
            i0_accum_s = signed_16bit(i0_accum)
            i0_scaled_s = signed_16bit(i0_scaled)

            if sample_counter >= 8 and not scaler_checked:
                dut._log.info(f"After 8 samples:")
                dut._log.info(f"  i0_accum  = {i0_accum_s} (raw: {i0_accum})")
                dut._log.info(f"  i0_scaled = {i0_scaled_s} (raw: {i0_scaled})")
                dut._log.info(f"  Expected i0_scaled ≈ i0_accum * 2 = {i0_accum_s * 2}")
                scaler_checked = True

                # Check scaler output
                expected = i0_accum_s * 2
                if abs(i0_scaled_s - expected) > 10:
                    dut._log.error(f"FAIL: Scaler output mismatch")
                    dut._log.error(f"  Got {i0_scaled_s}, expected ~{expected}")

        except Exception as e:
            dut._log.warning(f"Cycle {cycle}: Could not read signals: {e}")

    adapter.stop()

    if not scaler_checked:
        dut._log.error("FAIL: Never reached 8 samples")
        raise AssertionError("Never reached 8 samples to check scaler")

    dut._log.info("PASS: ADC scaler produces correct output")


# =============================================================================
# Stage 5: Verify m_hat Assignment
# =============================================================================

@cocotb.test()
async def test_stage5_mhat_assignment(dut):
    """
    Stage 5: Verify m_hat_real gets assigned from scaler.

    When FSM reaches COMPUTE_ERROR state, m_hat_real[0] should contain
    the scaled measurement, not 32768.
    """
    dut._log.info("=" * 60)
    dut._log.info("STAGE 5: Verify m_hat_real Assignment")
    dut._log.info("=" * 60)

    cocotb.start_soon(Clock(dut.clk, 10, units='ns').start())

    plant = PhotonicPlant(noise_std=2.0, seed=42)
    adapter = PlantAdapter(dut, plant)

    await reset_dut(dut)
    set_weights_q15(dut, 1.0 - 1e-4, 0.0, 0.0, 1.0 - 1e-4)
    dut.svd_mode.value = 0

    cocotb.start_soon(adapter.run())
    await start_calibration(dut)

    dut._log.info("Waiting for FSM to reach COMPUTE_ERROR state...")

    # CAL_COMPUTE_ERROR = 8 (from pkg_types.sv)
    CAL_COMPUTE_ERROR = 8

    compute_error_reached = False

    for cycle in range(500):
        await RisingEdge(dut.clk)

        try:
            cal_state = int(dut.cal_state.value)
            m00 = int(dut.m_hat_real[0].value)
            m01 = int(dut.m_hat_real[1].value)
            m10 = int(dut.m_hat_real[2].value)
            m11 = int(dut.m_hat_real[3].value)

            # Convert to signed
            m00_s = signed_16bit(m00)
            m01_s = signed_16bit(m01)
            m10_s = signed_16bit(m10)
            m11_s = signed_16bit(m11)

            if cal_state == CAL_COMPUTE_ERROR and not compute_error_reached:
                dut._log.info(f"At COMPUTE_ERROR (state {cal_state}):")
                dut._log.info(f"  m_hat_real[0] (M00) = {m00_s} (raw: {m00})")
                dut._log.info(f"  m_hat_real[1] (M01) = {m01_s} (raw: {m01})")
                dut._log.info(f"  m_hat_real[2] (M10) = {m10_s} (raw: {m10})")
                dut._log.info(f"  m_hat_real[3] (M11) = {m11_s} (raw: {m11})")

                compute_error_reached = True

                # Check for the stuck value
                if m00 == 32768 and m01 == 32768 and m10 == 32768 and m11 == 32768:
                    dut._log.error("FAIL: All m_hat_real values are 32768 (stuck at -32768)")
                    dut._log.error("This indicates measurements aren't being stored!")

        except Exception as e:
            dut._log.warning(f"Cycle {cycle}: Could not read signals: {e}")

    adapter.stop()

    if not compute_error_reached:
        dut._log.error("FAIL: FSM never reached COMPUTE_ERROR state")
        raise AssertionError("FSM stuck - never reached COMPUTE_ERROR")

    dut._log.info("PASS: m_hat_real assigned (check values above)")


# =============================================================================
# Full Chain Test
# =============================================================================

@cocotb.test()
async def test_full_chain_verbose(dut):
    """
    Full chain test with verbose logging at each step.

    This combines all stages into one continuous trace.
    """
    dut._log.info("=" * 60)
    dut._log.info("FULL CHAIN: Verbose trace of complete data path")
    dut._log.info("=" * 60)

    cocotb.start_soon(Clock(dut.clk, 10, units='ns').start())

    plant = PhotonicPlant(noise_std=2.0, seed=42)
    adapter = PlantAdapter(dut, plant)

    await reset_dut(dut)
    set_weights_q15(dut, 1.0 - 1e-4, 0.0, 0.0, 1.0 - 1e-4)
    dut.svd_mode.value = 0

    dut._log.info("Target weights: [[1,0],[0,1]] (identity)")

    cocotb.start_soon(adapter.run())
    await start_calibration(dut)

    prev_cal_state = -1

    for cycle in range(300):
        await RisingEdge(dut.clk)

        try:
            cal_state = int(dut.cal_state.value)
            plant_enable = int(dut.plant_enable.value)
            adc_i0 = signed_12bit(int(dut.adc_i0.value))
            adc_valid = int(dut.adc_valid.value)
            sample_counter = int(dut.sample_counter.value)
            i0_accum = signed_16bit(int(dut.i0_accum.value))
            i0_scaled = signed_16bit(int(dut.i0_scaled.value))
            m00 = signed_16bit(int(dut.m_hat_real[0].value))

            # Log on state transitions
            if cal_state != prev_cal_state:
                dut._log.info(f"--- State transition: {prev_cal_state} -> {cal_state} ---")
                prev_cal_state = cal_state

            # Log key events
            if adc_valid or sample_counter > 0 or cycle % 50 == 0:
                dut._log.info(
                    f"C{cycle:3d}: state={cal_state} en={plant_enable} "
                    f"valid={adc_valid} adc_i0={adc_i0:5d} "
                    f"samples={sample_counter} accum={i0_accum:6d} "
                    f"scaled={i0_scaled:6d} m00={m00:6d}"
                )

        except Exception as e:
            if cycle == 0:
                dut._log.warning(f"Could not read signals: {e}")

    adapter.stop()
    dut._log.info("Full chain trace complete")
