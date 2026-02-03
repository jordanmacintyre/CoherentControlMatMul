# Coherent Photonic Matrix Multiply

Mixed-signal co-simulation of a closed-loop coherent photonic 2×2 matrix multiplier. RTL calibration controller + Python photonic plant models via cocotb.

## What This Does

This system implements **hardware-in-the-loop calibration** for a photonic matrix multiplier:

```
User provides: Target matrix M = [[w₀, w₁], [w₂, w₃]]
RTL does:      Coordinate descent to find phases via I/Q feedback
Plant model:   Python photonic simulation (MZI, thermal, noise)
Result:        Verified RTL that computes y = M·x
```

The key insight is that **users specify what they want (matrix weights), not how to get it (phase values)**. The RTL calibration FSM finds the phases automatically using only I/Q measurements from the Python plant model.

## Quick Start

```bash
# Install dependencies
pip install -e .
pip install cocotb
brew install icarus-verilog  # macOS
# or: apt install iverilog   # Linux

# Check dependencies
python run.py --check-deps

# Run co-simulation
python run.py --weights 0.707 0.707 0.707 -0.707 --input 1.0 0.5

# Run all RTL tests
cd rtl && make test-all

# View waveforms
cd rtl && DUMP_WAVES=1 make test-all && make waves
```

## Architecture

### Mixed-Signal Co-Simulation

```
┌─────────────────────────────────────────────────────────────────┐
│  Python (cocotb)                                                │
│  ┌─────────────────┐      ┌──────────────────────────────────┐  │
│  │  PlantAdapter   │◄────►│ PhotonicPlant / SVDPhotonicPlant │  │
│  │  (RTL bridge)   │      │ (thermal, optics, receiver)      │  │
│  └────────┬────────┘      └──────────────────────────────────┘  │
│           │                                                     │
│           │ DAC codes, x_drive ↓   ↑ ADC I/Q, adc_valid         │
│           │                                                     │
├───────────┼─────────────────────────────────────────────────────┤
│  RTL      │                                                     │
│  ┌────────▼────────────────────────────────────────────────┐    │
│  │  coherent_matmul_top.sv                                 │    │
│  │  (calibration FSM, coordinate descent, error compute)   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Signal Flow

| RTL Output | Direction | Python Plant |
|------------|-----------|--------------|
| `phi_dac[4]` | RTL→Plant | Phase DAC codes (unitary) |
| `phi_dac_v[4]` | RTL→Plant | V† mesh phases (SVD) |
| `voa_dac[2]` | RTL→Plant | Σ VOA codes (SVD) |
| `phi_dac_u[4]` | RTL→Plant | U mesh phases (SVD) |
| `x_drive0/1` | RTL→Plant | Input vector (Q1.15) |
| `adc_i0/q0/i1/q1` | Plant→RTL | I/Q measurements |
| `adc_valid` | Plant→RTL | Sample valid pulse |

### Optical Architecture

**Standard Mode (Unitary):**
```
Input x → [MZI (4 phases)] → Output y = M̂·x
```

**SVD Mode (Any Matrix):**
```
Input x → [MZI V†] → [VOA Σ] → [MZI U] → Output y = U·Σ·V†·x
         4 phases   2 gains   4 phases
```

## Two Operating Modes

### Standard Mode (Unitary Matrices)

Single MZI mesh with 4 phase controls. Works for matrices where M†M ≈ I:

```bash
python run.py --weights 0.707 0.707 0.707 -0.707 --input 1.0 0.5
```

Suitable for: Identity, Swap, Hadamard, rotations, permutations.

### SVD Mode (Any Matrix)

Two MZI meshes + variable attenuators. Works for **any** 2×2 matrix:

```bash
python run.py --svd --weights 0.7 -0.3 0.2 -0.5 --input 1.0 0.5
python run.py --svd --weights 2.0 -1.5 3.0 -2.5 --input 1.0 0.5  # auto-scaled
```

Uses SVD decomposition: M = U·Σ·V† where U, V† are unitary (MZI meshes) and Σ is diagonal (VOAs).

## Command Reference

```
python run.py [OPTIONS]

Required:
  --weights W0 W1 W2 W3    Target matrix [[W0,W1],[W2,W3]]

Optional:
  --input X0 X1            Input vector (can specify multiple times)
  --svd                    Use SVD architecture for arbitrary matrices
  --noise N                Receiver noise std dev (default: 2.0)
  --seed N                 Random seed for reproducibility
  --verbose, -v            Show detailed output
  --timeout N              Simulation timeout in seconds (default: 120)
  --check-deps             Check dependencies and exit
```

## RTL Test Suite

```bash
cd rtl

# Run all tests
make test-all

# Run specific tests
make TESTCASE=test_unitary_identity
make TESTCASE=test_unitary_hadamard
make TESTCASE=test_svd_diagonal
make TESTCASE=test_svd_arbitrary

# Shortcuts
make test-unitary    # All unitary mode tests
make test-svd        # All SVD mode tests

# With waveforms
DUMP_WAVES=1 make test-all
make waves           # Opens GTKWave
```

### Test Cases

| Test | Mode | Description |
|------|------|-------------|
| `test_unitary_identity` | Unitary | Identity matrix calibration |
| `test_unitary_hadamard` | Unitary | Hadamard (balanced splitter) |
| `test_unitary_rotation` | Unitary | 30° rotation matrix |
| `test_svd_diagonal` | SVD | Diagonal scaling [[0.5,0],[0,0.3]] |
| `test_svd_arbitrary` | SVD | Arbitrary matrix [[0.7,-0.3],[0.2,-0.5]] |
| `test_convergence_monitoring` | Unitary | Verify error decreases |
| `test_reset_during_calibration` | Unitary | FSM recovery after reset |
| `test_mode_switching` | Both | Unitary → SVD transitions |
| `test_evaluation_after_lock` | Unitary | y = M·x computation |

## Understanding the Output

### Convergence Plots

The convergence plot shows RMS error (%) over calibration cycles:

- **Blue line**: RMS error percentage at each sample point
- **Green star**: Final calibration point (minimum error achieved)
- **Red dashed line**: Lock threshold (~1.6% RMS)
- **Orange/Green horizontal lines**: 5% and 1% accuracy targets

**Why does error go back up?** The optimizer continues searching after finding a good solution. The "Final Cal" marker shows where the best calibration was achieved—this is what gets used.

### Phase Evolution Plots

Shows the 4 phase DAC codes over time:

- **Vertical green line**: Cycle where minimum error occurred
- **Colored dots**: Final phase values at calibration point
- **Text box**: "Final Cal Values" showing exact DAC codes

### What Are "Final Cal Values"?

The phase values shown (e.g., `theta: 23842`) are **16-bit DAC codes**, not radians:

| DAC Code | Phase (radians) | Phase (degrees) |
|----------|-----------------|-----------------|
| 0 | 0 | 0° |
| 16384 | π/2 | 90° |
| 32768 | π | 180° |
| 49152 | 3π/2 | 270° |
| 65535 | 2π − ε | ~360° |

**Conversion:** `phase_rad = (dac_code / 65535) × 2π`

Example: `theta: 23842` → (23842/65535) × 2π = **2.29 radians (131°)**

These large numbers are the raw hardware control values. The MZI mesh uses these 4 phases to create the target 2×2 matrix transformation.

## How It Works

### The Calibration Problem

Given a target matrix M, find phase values φ = [θ, φ₀, φ₁, φ_out] such that the MZI transfer matrix matches M.

The MZI implements:
```
M̂(φ) = [[e^{j(φ₀+φ₁)}cos(θ/2), -e^{jφ₁}sin(θ/2)],
         [e^{j(φ₀+φ_out)}sin(θ/2), e^{jφ_out}cos(θ/2)]]
```

For **real** target matrices, we want M̂_real ≈ M and M̂_imag ≈ 0.

### Calibration Algorithm (RTL)

**Coordinate descent** with adaptive step sizing:

1. Apply basis inputs [1,0] and [0,1]
2. Measure I/Q outputs via coherent receiver (Python plant)
3. Compute error: ‖M̂_real - M_target‖² + ‖M̂_imag‖²
4. For each phase, try ±Δ and keep the direction that reduces error
5. Decay step size when no improvement
6. Lock when error stays below threshold for N consecutive iterations

### ADC Timing Protocol

The PlantAdapter respects RTL timing requirements:

```python
# 16-cycle settle period after phase change
if self._detect_phase_change(dac_codes):
    self._cycles_since_phase_change = 0
else:
    self._cycles_since_phase_change += 1

# Assert adc_valid only after settling
valid = self._cycles_since_phase_change >= self.settle_cycles
```

## Physical Models (Python)

### MZI Mesh
- Reck/Clements decomposition structure
- 4 thermal phase shifters
- Can realize any 2×2 unitary

### Thermal Dynamics
- RC time constant: τ = 100μs (configurable)
- Random drift: Brownian motion model
- Crosstalk: 5% coupling to adjacent heaters

### Coherent Receiver

```
                    ┌─────────────┐
Signal E(t) ───────►│ 90° Hybrid  │──► I = Re{E} = |E|cos(φ)
                    │             │
LO (reference) ────►│             │──► Q = Im{E} = |E|sin(φ)
                    └─────────────┘
```

- Dual I/Q detection (both output ports)
- Transimpedance gain: 2047 (fills ADC range)
- Gaussian noise model
- 12-bit ADC quantization

## Signal Scaling

### ADC Interface Contract

The coherent receiver outputs are quantized by 12-bit ADCs. The interface contract:

> **Full-scale ADC (±2047 LSB) = ±1.0 optical field amplitude**

This is achieved by setting receiver TIA gain = 2047, matching typical coherent optical system design where the analog front-end fills the ADC dynamic range.

### Fixed-Point Formats

| Signal | Format | Range | Resolution |
|--------|--------|-------|------------|
| ADC output | 12-bit signed | ±2047 | 1 LSB |
| Weights/Inputs | Q1.15 | ±1.0 | 3.05×10⁻⁵ |
| Phase DAC | 16-bit unsigned | [0, 2π) | 96 µrad |
| Error metric | 32-bit unsigned | [0, 4.3×10⁹] | 1 |

### ADC to Q1.15 Conversion

The `adc_scaler` module converts accumulated ADC samples to Q1.15:

1. Accumulate 8 ADC samples (adds ~3 bits of precision)
2. Scale by 2 (left shift 1) to reach Q1.15 range

**Math:** `q15_out = (Σ adc_samples) << 1`

For 1.0 optical: 8 × 2047 = 16376 → 16376 << 1 = 32752 ≈ 32767 (Q1.15 for 1.0)

## Calibration

### Error Metric

The calibration FSM minimizes sum-of-squared error between measured and target matrix elements:

```
error = Σᵢ (measured[i] - target[i])²
```

Both measured and target are Q1.15, so error is in squared-LSB units.

### Accuracy Targets

| Accuracy | Per-element diff | Threshold |
|----------|-----------------|-----------|
| 1% RMS | 328 LSB | 430,000 |
| 0.5% RMS | 164 LSB | 100,000 |
| 0.1% RMS | 33 LSB | 4,000 |

Default threshold (100,000) targets **0.5% RMS error**.

### Convergence

The coordinate descent algorithm adjusts one phase at a time:

1. Probe φ ± Δφ, measure error
2. Move in direction that reduces error
3. If no improvement, reduce step size
4. Repeat until error < threshold for N consecutive iterations

Typical convergence: 500-1500 iterations (~50-150 µs at 100 MHz).

## Calibration Algorithm Details

### Momentum-Based Coordinate Descent

The RTL implements gradient-free optimization with momentum acceleration:

1. **Probing**: For each phase, try φ+Δ and φ−Δ
2. **Direction**: Move toward lower error
3. **Momentum**: `v_new = β·v_old + step` where β ≈ 0.3125
4. **Direction reversal**: Reset velocity when gradient flips (prevents overshoot)

```systemverilog
// Momentum decay: v >> 2 + v >> 4 ≈ 0.3125 × v
v_decayed = (velocity >>> 2) + (velocity >>> 4);

// If direction changed, reset velocity
if (gradient_positive != velocity_positive)
    v_new = ±step;  // Reset
else
    v_new = v_decayed ± step;  // Accumulate
```

### Hysteresis Lock Detection

Two thresholds prevent oscillation near the lock boundary:

| Threshold | Value | Purpose |
|-----------|-------|---------|
| CAL_LOCK_THRESHOLD | 1,048,576 (~1.6% RMS) | Lock when error drops below |
| CAL_UNLOCK_THRESHOLD | 1,572,864 (~2.4% RMS) | Only reset if error exceeds |

**Hysteresis band**: If error is between thresholds, lock_counter is maintained (not reset). This prevents the calibration from "bouncing" around the threshold.

### Step Size Adaptation

| Parameter | Value | Purpose |
|-----------|-------|---------|
| PHASE_STEP_INITIAL | 0x1600 (~8.6% of 2π) | Coarse search |
| PHASE_STEP_MIN | 0x0028 (~0.06% of 2π) | Fine convergence |
| Decay | ÷2 every 64 iterations | Automatic refinement |

### Variable Optical Attenuators (SVD mode)
- 16-bit DAC control
- 40 dB attenuation range
- Maps singular values σ ∈ [0,1] to optical transmission

## Project Structure

```
CoherentControlMatMul/
├── run.py                     # Main entry point (runs co-simulation)
├── plant/                     # Photonic hardware models
│   ├── mzi_mesh.py           # MZI transfer matrix
│   ├── thermal_dynamics.py   # Heater RC model + drift
│   ├── coherent_receiver.py  # I/Q detection + noise
│   ├── voa.py                # Variable attenuators
│   ├── svd_plant.py          # SVD architecture wrapper
│   └── plant_wrapper.py      # Unified interface
├── rtl/                       # RTL + co-simulation tests
│   ├── pkg_types.sv          # Type definitions, FSM states
│   ├── coherent_matmul_top.sv # Calibration controller
│   ├── Makefile              # cocotb build rules
│   └── tests/                # Co-simulation tests
│       ├── plant_adapter.py  # RTL ↔ Python bridge
│       ├── test_cosim.py     # cocotb test cases
│       └── conftest.py       # pytest fixtures
└── sim/results/               # Experiment outputs
```

## RTL Implementation

### Software/Hardware Boundary

| Component | Software (Python) | Hardware (RTL) |
|-----------|-------------------|----------------|
| SVD decomposition | ✓ `np.linalg.svd(M)` | - |
| σ → DAC conversion | ✓ `voa.sigma_to_dac()` | - |
| Photonic simulation | ✓ MZI, thermal, noise | - |
| Calibration FSM | - | ✓ coordinate descent |
| I/Q measurement | - | ✓ ADC sampling |
| Phase DAC output | - | ✓ 10 DAC channels |
| Error computation | - | ✓ fixed-point |

### SVD Handoff Protocol

```
Software                          Hardware (RTL)
────────                          ──────────────
1. Compute SVD: M = U·Σ·V†
2. Convert σ → DAC codes
3. Write sigma_dac0, sigma_dac1  →  Stores in voa_reg[]
4. Set svd_mode = 1              →  Enters SVD calibration
5. Assert start_cal              →  Begins calibration
                                    ...calibrates V† (4 phases)...
                                    Sets VOA from sigma_dac inputs
                                    ...calibrates U (4 phases)...
                                 ←  cal_locked asserts
6. Read cal_locked
7. Assert start_eval with x0,x1  →  Computes y = M·x
                                 ←  y0_out, y1_out, y_valid
```

### RTL Interface (SVD Mode)

```systemverilog
// Inputs
input  logic        svd_mode,      // 1 = SVD, 0 = unitary
input  logic [15:0] sigma_dac0,    // Pre-computed σ₀ VOA code
input  logic [15:0] sigma_dac1,    // Pre-computed σ₁ VOA code

// Outputs (10 control parameters)
output logic [15:0] phi_dac_v[4],  // V† mesh phases
output logic [15:0] voa_dac[2],    // Σ attenuators
output logic [15:0] phi_dac_u[4],  // U mesh phases

// Status
output logic        cal_locked,    // Calibration complete
output logic [7:0]  status_flags,  // Includes v_locked, u_locked
```

### DAC Code Layout

```
Index:   0   1   2   3   4   5   6   7   8   9
         v0  v1  v2  v3  σ0  σ1  u0  u1  u2  u3
         └── V† mesh ──┘  └VOA┘  └── U mesh ──┘
```

## Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Matrix size | 2×2 | Simplest case; scales via Clements/Reck |
| Verification | RTL + Python co-sim | Realistic closed-loop testing |
| Calibration | Coordinate descent in RTL | Noise-robust, hardware-friendly |
| Plant model | Python | Easy to modify, realistic physics |
| Detection | Coherent I/Q | Preserves phase, enables signed values |
| Thermal model | Full RC + drift | Realistic control challenges |

## Writing Custom Tests

```python
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

from tests.plant_adapter import PlantAdapter
from plant.plant_wrapper import PhotonicPlant, float_to_q1_15

@cocotb.test()
async def test_my_matrix(dut):
    """Test calibration to custom matrix."""
    cocotb.start_soon(Clock(dut.clk, 10, units='ns').start())

    plant = PhotonicPlant(noise_std=2.0, seed=42)
    adapter = PlantAdapter(dut, plant)

    # Reset
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    # Set target weights
    dut.w0.value = float_to_q1_15(0.8)
    dut.w1.value = float_to_q1_15(0.6)
    dut.w2.value = float_to_q1_15(-0.6)
    dut.w3.value = float_to_q1_15(0.8)

    # Start adapter and calibration
    cocotb.start_soon(adapter.run())
    dut.start_cal.value = 1
    await RisingEdge(dut.clk)
    dut.start_cal.value = 0

    # Wait for lock
    for _ in range(50000):
        await RisingEdge(dut.clk)
        if dut.cal_locked.value:
            break

    adapter.stop()
    assert dut.cal_locked.value, "Calibration failed"
```

## Glossary

| Term | Definition |
|------|------------|
| **DAC code** | 16-bit value (0–65535) controlling phase shifter voltage |
| **Q1.15** | Fixed-point format: 1 sign bit, 15 fractional bits, range [−1, 1) |
| **MZI** | Mach-Zehnder Interferometer—optical 2×2 coupler with phase control |
| **I/Q** | In-phase/Quadrature—complex signal decomposition |
| **RMS error** | Root-mean-square error as percentage of full scale |
| **Lock** | Calibration complete—error below threshold for N cycles |
| **Hysteresis** | Different lock/unlock thresholds to prevent oscillation |
| **Momentum** | Velocity accumulation in optimizer to accelerate convergence |
| **Unitary** | Matrix where M†M = I (preserves signal power) |
| **SVD** | Singular Value Decomposition: M = U·Σ·V† |
| **VOA** | Variable Optical Attenuator—controls signal amplitude |
| **Basis input** | Test vectors [1,0] and [0,1] to measure matrix columns |
| **Settling time** | 16 cycles after phase change for thermal equilibrium |
| **Coordinate descent** | Optimization that adjusts one parameter at a time |
| **Coherent receiver** | Optical detector that measures both amplitude and phase via I/Q |

## License

MIT
