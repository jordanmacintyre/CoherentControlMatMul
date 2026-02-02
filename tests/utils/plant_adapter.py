"""
Plant-in-the-loop adapter for cocotb testbench.

Bridges cocotb signals to the Python photonic plant model.
"""

import sys
from pathlib import Path

# Add plant package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cocotb
from cocotb.triggers import RisingEdge, Timer

from plant import PhotonicPlant
from .stimulus import q1_15_to_float


class PlantInLoop:
    """
    Bridges cocotb signals to Python plant model.

    Runs as a coroutine alongside the RTL simulation, reading phase DAC
    outputs and input drives from the RTL, stepping the plant model,
    and driving ADC signals back to the RTL.
    """

    def __init__(
        self,
        dut,
        plant: PhotonicPlant,
        num_phases: int = 4,
    ):
        """
        Initialize plant-in-loop adapter.

        Args:
            dut: cocotb DUT handle
            plant: PhotonicPlant instance
            num_phases: Number of phase shifters
        """
        self.dut = dut
        self.plant = plant
        self.num_phases = num_phases
        self._running = False
        self._cycle_count = 0

    async def run(self):
        """
        Main loop: read RTL outputs, step plant, drive ADC inputs.

        This coroutine runs indefinitely until stopped.
        """
        self._running = True
        self._cycle_count = 0

        while self._running:
            await RisingEdge(self.dut.clk)
            self._cycle_count += 1

            # Check if plant is enabled
            plant_enable = bool(self.dut.plant_enable.value)

            if plant_enable:
                # Read phase DACs from RTL
                phi_dac = []
                for i in range(self.num_phases):
                    try:
                        phi_dac.append(int(self.dut.phi_dac[i].value))
                    except (ValueError, AttributeError):
                        phi_dac.append(0)

                # Read input drives (Q1.15 format)
                try:
                    x0_code = int(self.dut.x_drive0.value.signed_integer)
                    x1_code = int(self.dut.x_drive1.value.signed_integer)
                except (ValueError, AttributeError):
                    x0_code = 0
                    x1_code = 0

                x0 = q1_15_to_float(x0_code)
                x1 = q1_15_to_float(x1_code)

                # Step plant model (inputs are real, convert to complex)
                i0, q0, i1, q1 = self.plant.sample_outputs(
                    phi_dac, complex(x0, 0), complex(x1, 0)
                )

                # Drive ADC signals back to RTL
                self.dut.adc_i0.value = i0
                self.dut.adc_q0.value = q0
                self.dut.adc_i1.value = i1
                self.dut.adc_q1.value = q1
                self.dut.adc_valid.value = 1
            else:
                # Plant disabled, set ADC invalid
                self.dut.adc_i0.value = 0
                self.dut.adc_q0.value = 0
                self.dut.adc_i1.value = 0
                self.dut.adc_q1.value = 0
                self.dut.adc_valid.value = 0

    def stop(self):
        """Stop the plant-in-loop coroutine."""
        self._running = False

    @property
    def cycle_count(self) -> int:
        """Number of simulation cycles executed."""
        return self._cycle_count


async def run_clock(dut, period_ns: float = 10.0, max_cycles: int | None = None):
    """
    Generate clock signal.

    Args:
        dut: cocotb DUT handle
        period_ns: Clock period in nanoseconds
        max_cycles: Maximum number of cycles (None for infinite)
    """
    half_period = period_ns / 2
    cycles = 0

    while max_cycles is None or cycles < max_cycles:
        dut.clk.value = 0
        await Timer(half_period, units="ns")
        dut.clk.value = 1
        await Timer(half_period, units="ns")
        cycles += 1


async def reset_dut(dut, duration_cycles: int = 5):
    """
    Reset the DUT.

    Args:
        dut: cocotb DUT handle
        duration_cycles: Number of clock cycles to hold reset
    """
    dut.rst_n.value = 0

    # Initialize all inputs to safe values
    dut.w0.value = 0
    dut.w1.value = 0
    dut.w2.value = 0
    dut.w3.value = 0
    dut.x0.value = 0
    dut.x1.value = 0
    dut.start_cal.value = 0
    dut.start_eval.value = 0
    dut.adc_i0.value = 0
    dut.adc_q0.value = 0
    dut.adc_i1.value = 0
    dut.adc_q1.value = 0
    dut.adc_valid.value = 0

    for _ in range(duration_cycles):
        await RisingEdge(dut.clk)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


def set_weights(dut, w0: float, w1: float, w2: float, w3: float):
    """
    Set target matrix weights on DUT.

    Args:
        dut: cocotb DUT handle
        w0, w1, w2, w3: Target weights in [-1, 1]
    """
    from .stimulus import float_to_q1_15, validate_weights

    validate_weights(w0, w1, w2, w3)

    dut.w0.value = float_to_q1_15(w0)
    dut.w1.value = float_to_q1_15(w1)
    dut.w2.value = float_to_q1_15(w2)
    dut.w3.value = float_to_q1_15(w3)


def set_inputs(dut, x0: float, x1: float):
    """
    Set input values on DUT.

    Args:
        dut: cocotb DUT handle
        x0, x1: Input values in [-1, 1]
    """
    from .stimulus import float_to_q1_15

    dut.x0.value = float_to_q1_15(x0)
    dut.x1.value = float_to_q1_15(x1)


async def start_calibration(dut):
    """
    Trigger calibration start.

    Args:
        dut: cocotb DUT handle
    """
    dut.start_cal.value = 1
    await RisingEdge(dut.clk)
    dut.start_cal.value = 0


async def start_evaluation(dut):
    """
    Trigger evaluation start.

    Args:
        dut: cocotb DUT handle
    """
    dut.start_eval.value = 1
    await RisingEdge(dut.clk)
    dut.start_eval.value = 0


async def wait_for_calibration(
    dut, max_cycles: int = 10000
) -> tuple[bool, int]:
    """
    Wait for calibration to complete.

    Args:
        dut: cocotb DUT handle
        max_cycles: Maximum cycles to wait

    Returns:
        Tuple of (locked, cycles) where locked is True if calibration
        succeeded, and cycles is the number of cycles taken
    """
    for cycle in range(max_cycles):
        await RisingEdge(dut.clk)

        if dut.cal_locked.value:
            return True, cycle + 1

        # Check for error state
        status = int(dut.status_flags.value)
        if status & 0x10:  # error_weights flag
            return False, cycle + 1
        if status & 0x08:  # error_timeout flag
            return False, cycle + 1

    return False, max_cycles


async def wait_for_evaluation(dut, max_cycles: int = 100) -> tuple[bool, int]:
    """
    Wait for evaluation to complete.

    Args:
        dut: cocotb DUT handle
        max_cycles: Maximum cycles to wait

    Returns:
        Tuple of (done, cycles) where done is True if evaluation
        completed, and cycles is the number of cycles taken
    """
    for cycle in range(max_cycles):
        await RisingEdge(dut.clk)

        if dut.eval_done.value:
            return True, cycle + 1

    return False, max_cycles
