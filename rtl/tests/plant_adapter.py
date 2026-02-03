"""
Plant adapter for RTL ↔ Python mixed-signal co-simulation.

This module bridges the RTL controller (coherent_matmul_top.sv) with the
Python photonic plant models (PhotonicPlant, SVDPhotonicPlant). It reads
DAC codes and input drives from RTL, steps the plant model, and drives
ADC outputs back to RTL with proper timing.

Signal mapping:
    RTL Output          →  Python Plant
    ───────────────────────────────────
    phi_dac[4]          →  dac_codes[0:4]  (unitary mode)
    phi_dac_v[4]        →  dac_codes[0:4]  (SVD mode)
    voa_dac[2]          →  dac_codes[4:6]  (SVD mode)
    phi_dac_u[4]        →  dac_codes[6:10] (SVD mode)
    x_drive0/1          →  x0, x1 (Q1.15 → complex)

    Python Plant        →  RTL Input
    ───────────────────────────────────
    sample_outputs()    →  adc_i0/q0/i1/q1
    timing_valid()      →  adc_valid
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import cocotb
from cocotb.triggers import RisingEdge, FallingEdge
from cocotb.handle import SimHandleBase

import sys
from pathlib import Path

# Add project root to path for plant imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from plant.plant_wrapper import PhotonicPlant, q1_15_to_float
from plant.svd_plant import SVDPhotonicPlant


def signed_to_int(val: int, bits: int) -> int:
    """Convert unsigned SimHandleBase value to signed Python int."""
    if val >= (1 << (bits - 1)):
        return val - (1 << bits)
    return val


def int_to_unsigned(val: int, bits: int) -> int:
    """Convert signed Python int to unsigned value for RTL."""
    if val < 0:
        return val + (1 << bits)
    return val


class PlantAdapter:
    """
    Bridges RTL signals to Python PhotonicPlant model for unitary mode.

    This adapter:
    1. Reads phi_dac[0:4] phase codes from RTL
    2. Reads x_drive0/1 input vectors (Q1.15 format)
    3. Steps the PhotonicPlant model
    4. Drives adc_i0/q0/i1/q1 with plant outputs
    5. Controls adc_valid timing per RTL protocol

    ADC Timing Protocol:
    - RTL waits CAL_SETTLE_CYCLES (16) after phase change before sampling
    - RTL accumulates CAL_AVG_SAMPLES (8) adc_valid pulses per measurement
    - This adapter pulses adc_valid each cycle after the settle period
    """

    # Constants from pkg_types.sv
    NUM_PHASES = 4
    DATA_WIDTH = 16
    ADC_WIDTH = 12
    PHASE_WIDTH = 16
    CAL_SETTLE_CYCLES = 16
    CAL_AVG_SAMPLES = 8

    def __init__(
        self,
        dut: SimHandleBase,
        plant: PhotonicPlant,
        settle_cycles: int = 16,
    ):
        """
        Initialize plant adapter.

        Args:
            dut: cocotb DUT handle (coherent_matmul_top)
            plant: PhotonicPlant instance
            settle_cycles: Cycles to wait after phase change before asserting adc_valid
        """
        self.dut = dut
        self.plant = plant
        self.settle_cycles = settle_cycles
        self._running = False
        self._cycles_since_phase_change = 0
        self._last_dac_codes: list[int] = [0] * self.NUM_PHASES

    def _read_dac_codes(self) -> list[int]:
        """Read phase DAC codes from RTL."""
        codes = []
        for i in range(self.NUM_PHASES):
            codes.append(int(self.dut.phi_dac[i].value))
        return codes

    def _read_input_drives(self) -> tuple[complex, complex]:
        """Read x_drive0/1 from RTL and convert to complex inputs."""
        # Read Q1.15 values
        x0_q15 = int(self.dut.x_drive0.value)
        x1_q15 = int(self.dut.x_drive1.value)

        # Convert signed values
        x0_q15 = signed_to_int(x0_q15, self.DATA_WIDTH)
        x1_q15 = signed_to_int(x1_q15, self.DATA_WIDTH)

        # Convert to float and treat as complex (real only for now)
        x0 = complex(q1_15_to_float(x0_q15), 0.0)
        x1 = complex(q1_15_to_float(x1_q15), 0.0)

        return x0, x1

    def _detect_phase_change(self, current_codes: list[int]) -> bool:
        """Detect if any phase DAC code changed."""
        changed = current_codes != self._last_dac_codes
        self._last_dac_codes = current_codes.copy()
        return changed

    def _should_assert_valid(self) -> bool:
        """Determine if adc_valid should be asserted this cycle."""
        return self._cycles_since_phase_change >= self.settle_cycles

    def _drive_adc(self, i0: int, q0: int, i1: int, q1: int, valid: bool) -> None:
        """Drive ADC signals to RTL."""
        # Clamp to ADC range
        adc_max = (1 << (self.ADC_WIDTH - 1)) - 1
        adc_min = -(1 << (self.ADC_WIDTH - 1))

        i0 = max(adc_min, min(adc_max, i0))
        q0 = max(adc_min, min(adc_max, q0))
        i1 = max(adc_min, min(adc_max, i1))
        q1 = max(adc_min, min(adc_max, q1))

        # Convert to unsigned for RTL
        self.dut.adc_i0.value = int_to_unsigned(i0, self.ADC_WIDTH)
        self.dut.adc_q0.value = int_to_unsigned(q0, self.ADC_WIDTH)
        self.dut.adc_i1.value = int_to_unsigned(i1, self.ADC_WIDTH)
        self.dut.adc_q1.value = int_to_unsigned(q1, self.ADC_WIDTH)
        self.dut.adc_valid.value = 1 if valid else 0

    async def run(self) -> None:
        """
        Main adapter loop: continuously bridge RTL ↔ plant.

        Should be started with cocotb.start_soon() and will run until stop() is called.
        """
        self._running = True
        self._cycles_since_phase_change = 0
        self._last_dac_codes = [0] * self.NUM_PHASES

        while self._running:
            await RisingEdge(self.dut.clk)

            # Check if plant should be active
            if int(self.dut.plant_enable.value) == 0:
                self._drive_adc(0, 0, 0, 0, False)
                continue

            # Read RTL outputs
            dac_codes = self._read_dac_codes()
            x0, x1 = self._read_input_drives()

            # Detect phase changes and reset settle counter
            if self._detect_phase_change(dac_codes):
                self._cycles_since_phase_change = 0
            else:
                self._cycles_since_phase_change += 1

            # Step plant model and get I/Q outputs
            i0, q0, i1, q1 = self.plant.sample_outputs(dac_codes, x0, x1)

            # Drive ADC with proper timing
            valid = self._should_assert_valid()
            self._drive_adc(i0, q0, i1, q1, valid)

    def stop(self) -> None:
        """Stop the adapter loop."""
        self._running = False


class SVDPlantAdapter:
    """
    Bridges RTL signals to Python SVDPhotonicPlant model for SVD mode.

    This adapter handles the 10-parameter SVD interface:
    - phi_dac_v[4]: V† mesh phases
    - voa_dac[2]: Σ diagonal attenuators
    - phi_dac_u[4]: U mesh phases

    DAC code layout: [v0, v1, v2, v3, σ0, σ1, u0, u1, u2, u3]
    """

    # Constants
    NUM_PHASES_V = 4
    NUM_PHASES_U = 4
    NUM_VOAS = 2
    TOTAL_PARAMS = 10
    DATA_WIDTH = 16
    ADC_WIDTH = 12
    PHASE_WIDTH = 16
    VOA_WIDTH = 16
    CAL_SETTLE_CYCLES = 16

    def __init__(
        self,
        dut: SimHandleBase,
        plant: SVDPhotonicPlant,
        settle_cycles: int = 16,
    ):
        """
        Initialize SVD plant adapter.

        Args:
            dut: cocotb DUT handle
            plant: SVDPhotonicPlant instance
            settle_cycles: Cycles to wait after parameter change
        """
        self.dut = dut
        self.plant = plant
        self.settle_cycles = settle_cycles
        self._running = False
        self._cycles_since_param_change = 0
        self._last_dac_codes: list[int] = [0] * self.TOTAL_PARAMS

    def _read_dac_codes(self) -> list[int]:
        """Read all 10 DAC codes from RTL."""
        codes = []

        # V† phases [0:4]
        for i in range(self.NUM_PHASES_V):
            codes.append(int(self.dut.phi_dac_v[i].value))

        # VOA codes [4:6]
        for i in range(self.NUM_VOAS):
            codes.append(int(self.dut.voa_dac[i].value))

        # U phases [6:10]
        for i in range(self.NUM_PHASES_U):
            codes.append(int(self.dut.phi_dac_u[i].value))

        return codes

    def _read_input_drives(self) -> tuple[complex, complex]:
        """Read x_drive0/1 from RTL and convert to complex inputs."""
        x0_q15 = int(self.dut.x_drive0.value)
        x1_q15 = int(self.dut.x_drive1.value)

        x0_q15 = signed_to_int(x0_q15, self.DATA_WIDTH)
        x1_q15 = signed_to_int(x1_q15, self.DATA_WIDTH)

        x0 = complex(q1_15_to_float(x0_q15), 0.0)
        x1 = complex(q1_15_to_float(x1_q15), 0.0)

        return x0, x1

    def _detect_param_change(self, current_codes: list[int]) -> bool:
        """Detect if any DAC code changed."""
        changed = current_codes != self._last_dac_codes
        self._last_dac_codes = current_codes.copy()
        return changed

    def _should_assert_valid(self) -> bool:
        """Determine if adc_valid should be asserted."""
        return self._cycles_since_param_change >= self.settle_cycles

    def _drive_adc(self, i0: int, q0: int, i1: int, q1: int, valid: bool) -> None:
        """Drive ADC signals to RTL."""
        adc_max = (1 << (self.ADC_WIDTH - 1)) - 1
        adc_min = -(1 << (self.ADC_WIDTH - 1))

        i0 = max(adc_min, min(adc_max, i0))
        q0 = max(adc_min, min(adc_max, q0))
        i1 = max(adc_min, min(adc_max, i1))
        q1 = max(adc_min, min(adc_max, q1))

        self.dut.adc_i0.value = int_to_unsigned(i0, self.ADC_WIDTH)
        self.dut.adc_q0.value = int_to_unsigned(q0, self.ADC_WIDTH)
        self.dut.adc_i1.value = int_to_unsigned(i1, self.ADC_WIDTH)
        self.dut.adc_q1.value = int_to_unsigned(q1, self.ADC_WIDTH)
        self.dut.adc_valid.value = 1 if valid else 0

    async def run(self) -> None:
        """Main adapter loop for SVD mode."""
        self._running = True
        self._cycles_since_param_change = 0
        self._last_dac_codes = [0] * self.TOTAL_PARAMS

        while self._running:
            await RisingEdge(self.dut.clk)

            if int(self.dut.plant_enable.value) == 0:
                self._drive_adc(0, 0, 0, 0, False)
                continue

            # Read RTL outputs (all 10 parameters)
            dac_codes = self._read_dac_codes()
            x0, x1 = self._read_input_drives()

            # Track parameter changes for settle timing
            if self._detect_param_change(dac_codes):
                self._cycles_since_param_change = 0
            else:
                self._cycles_since_param_change += 1

            # Step plant model
            i0, q0, i1, q1 = self.plant.sample_outputs(dac_codes, x0, x1)

            # Drive ADC with timing
            valid = self._should_assert_valid()
            self._drive_adc(i0, q0, i1, q1, valid)

    def stop(self) -> None:
        """Stop the adapter loop."""
        self._running = False


class AutoSelectPlantAdapter:
    """
    Automatically selects between unitary and SVD plant adapters based on svd_mode signal.

    This wrapper creates both plant types and switches between them based on the
    RTL svd_mode signal, allowing seamless testing of both modes.
    """

    def __init__(
        self,
        dut: SimHandleBase,
        noise_std: float = 2.0,
        seed: int | None = 42,
        settle_cycles: int = 16,
    ):
        """
        Initialize auto-selecting adapter.

        Args:
            dut: cocotb DUT handle
            noise_std: Receiver noise standard deviation
            seed: Random seed for reproducibility
            settle_cycles: Settle cycles before adc_valid
        """
        self.dut = dut

        # Create both plant types
        self.unitary_plant = PhotonicPlant(noise_std=noise_std, seed=seed)
        self.svd_plant = SVDPhotonicPlant(noise_std=noise_std, seed=seed)

        # Create both adapters
        self.unitary_adapter = PlantAdapter(dut, self.unitary_plant, settle_cycles)
        self.svd_adapter = SVDPlantAdapter(dut, self.svd_plant, settle_cycles)

        self._running = False

    async def run(self) -> None:
        """
        Main loop that delegates to appropriate adapter based on svd_mode.
        """
        self._running = True

        while self._running:
            await RisingEdge(self.dut.clk)

            # Determine mode and delegate
            svd_mode = int(self.dut.svd_mode.value) if hasattr(self.dut, 'svd_mode') else 0

            if int(self.dut.plant_enable.value) == 0:
                # Plant disabled
                self.dut.adc_i0.value = 0
                self.dut.adc_q0.value = 0
                self.dut.adc_i1.value = 0
                self.dut.adc_q1.value = 0
                self.dut.adc_valid.value = 0
                continue

            if svd_mode:
                # SVD mode
                dac_codes = self.svd_adapter._read_dac_codes()
                x0, x1 = self.svd_adapter._read_input_drives()

                if self.svd_adapter._detect_param_change(dac_codes):
                    self.svd_adapter._cycles_since_param_change = 0
                else:
                    self.svd_adapter._cycles_since_param_change += 1

                i0, q0, i1, q1 = self.svd_plant.sample_outputs(dac_codes, x0, x1)
                valid = self.svd_adapter._should_assert_valid()
                self.svd_adapter._drive_adc(i0, q0, i1, q1, valid)
            else:
                # Unitary mode
                dac_codes = self.unitary_adapter._read_dac_codes()
                x0, x1 = self.unitary_adapter._read_input_drives()

                if self.unitary_adapter._detect_phase_change(dac_codes):
                    self.unitary_adapter._cycles_since_phase_change = 0
                else:
                    self.unitary_adapter._cycles_since_phase_change += 1

                i0, q0, i1, q1 = self.unitary_plant.sample_outputs(dac_codes, x0, x1)
                valid = self.unitary_adapter._should_assert_valid()
                self.unitary_adapter._drive_adc(i0, q0, i1, q1, valid)

    def stop(self) -> None:
        """Stop the adapter."""
        self._running = False
