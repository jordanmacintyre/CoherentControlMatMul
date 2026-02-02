# Coherent Photonic Matrix Multiply

Closed-loop coherent photonic 2×2 matrix multiplication with output-driven phase control.

## What This Does

This system implements **hardware-in-the-loop calibration** for a photonic matrix multiplier:

```
User provides: Target matrix M = [[w₀, w₁], [w₂, w₃]]
System does:   Adjusts optical phases until photonic output matches M·x
Result:        Hardware computes y = M·x at the speed of light
```

The key insight is that **users specify what they want (matrix weights), not how to get it (phase values)**. The calibration loop finds the phases automatically using only I/Q measurements as feedback.

## Quick Start

```bash
# Install
pip install -e .

# Run with a Hadamard matrix
python run.py --weights 0.707 0.707 0.707 -0.707 --input 1.0 0.5

# Use SVD mode for non-unitary matrices
python run.py --svd --weights 0.5 0 0 0.3 --input 1.0 1.0

# Save results with plots
python run.py --weights 1 0 0 1 --input 1.0 0.0 --save-results --verbose
```

## Two Operating Modes

### Standard Mode (Unitary Matrices)

Single MZI mesh with 4 phase controls. Works for matrices where M†M = I:

```bash
python run.py --weights 0.707 0.707 0.707 -0.707 --input 1.0 0.5
```

Suitable for: Identity, Swap, Hadamard, rotations, permutations.

### SVD Mode (Any Matrix)

Two MZI meshes + variable attenuators. Works for **any** 2×2 matrix:

```bash
# Standard SVD (weights in [-1, 1])
python run.py --svd --weights 0.7 -0.3 0.2 -0.5 --input 1.0 0.5

# Large weights (auto-scaled)
python run.py --svd --weights 2.0 -1.5 3.0 -2.5 --input 1.0 0.5
```

Uses SVD decomposition: M = U·Σ·V† where U, V† are unitary (MZI meshes) and Σ is diagonal (VOAs).

**Auto-scaling**: When singular values exceed 1.0, the system automatically normalizes the matrix and scales the outputs in software. This enables weights of any magnitude.

## Command Reference

```
python run.py [OPTIONS]

Required:
  --weights W0 W1 W2 W3    Target matrix [[W0,W1],[W2,W3]]
                           Range [-1,1] for unitary mode; any values for --svd

Optional:
  --input X0 X1            Input vector (can specify multiple times)
  --svd                    Use SVD architecture for arbitrary matrices (auto-scales)
  --noise N                Receiver noise std dev (default: 2.0)
  --max-iterations N       Max calibration iterations (default: 300)
  --error-threshold E      Lock threshold (default: 2e-3)
  --verbose, -v            Show detailed output
  --save-results, -s       Save to sim/results/<timestamp>/
  --seed N                 Random seed for reproducibility
```

## How It Works

### The Calibration Problem

Given a target matrix M, find phase values φ = [θ, φ₀, φ₁, φ_out] such that the MZI transfer matrix matches M.

The MZI implements:
```
M̂(φ) = [[e^{j(φ₀+φ₁)}cos(θ/2), -e^{jφ₁}sin(θ/2)],
         [e^{j(φ₀+φ_out)}sin(θ/2), e^{jφ_out}cos(θ/2)]]
```

For **real** target matrices, we want M̂_real ≈ M and M̂_imag ≈ 0.

### Calibration Algorithm

**Coordinate descent** with adaptive step sizing:

1. Apply basis inputs [1,0] and [0,1]
2. Measure I/Q outputs via coherent receiver
3. Compute error: ‖M̂_real - M_target‖² + ‖M̂_imag‖²
4. For each phase, try ±Δ and keep the direction that reduces error
5. Decay step size when no improvement
6. Lock when error stays below threshold for N consecutive iterations

### Why Coordinate Descent?

- **Noise robust**: Discrete ±Δ probing handles measurement noise better than gradient estimation
- **No derivatives**: Avoids numerical differentiation issues
- **Hardware-friendly**: Simple comparisons, easy to implement in RTL
- **Adaptive**: Step decay prevents oscillation near optimum

### Why Not Analytical?

The mapping from target matrix to phases is non-linear and affected by:
- Manufacturing variations
- Thermal drift
- Heater crosstalk

Closed-loop calibration adapts to real conditions; analytical solutions assume ideal hardware.

## Architecture

### Standard Mode
```
Input x → [MZI (4 phases)] → Output y = M̂·x
```

### SVD Mode
```
Input x → [MZI V†] → [VOA Σ] → [MZI U] → Output y = U·Σ·V†·x
         4 phases   2 gains   4 phases
```

The SVD architecture can realize **any** matrix with singular values ≤ 1.

## Physical Models

### MZI Mesh
- Reck/Clements decomposition structure
- 4 thermal phase shifters
- Can realize any 2×2 unitary

### Thermal Dynamics
- RC time constant: τ = 100μs (configurable)
- Random drift: Brownian motion model
- Crosstalk: 5% coupling to adjacent heaters

### Coherent Receiver

The coherent receiver extracts both the **amplitude** and **phase** of the optical signal using I/Q (In-phase/Quadrature) detection:

```
                    ┌─────────────┐
Signal E(t) ───────►│ 90° Hybrid  │──► I = Re{E} = |E|cos(φ)
                    │             │
LO (reference) ────►│             │──► Q = Im{E} = |E|sin(φ)
                    └─────────────┘
```

- **I (In-phase)**: The real component of the complex field. Measures projection onto the reference.
- **Q (Quadrature)**: The imaginary component. Measures projection 90° out of phase.
- **Together**: I + jQ = E recovers the full complex field, including sign information.

This is essential because:
- Direct detection (photodiode) only measures |E|², losing phase
- With I/Q, we can distinguish +0.5 from -0.5 (same power, opposite sign)
- Matrix calibration needs signed values to achieve negative weights

Hardware:
- Dual I/Q detection (both output ports)
- Transimpedance gain: 1000
- Gaussian noise model
- 12-bit ADC quantization

### Variable Optical Attenuators (SVD mode)
- 16-bit DAC control
- 40 dB attenuation range
- Maps singular values σ ∈ [0,1] to optical transmission

## Project Structure

```
CoherentControlMatMul/
├── run.py                     # Main entry point
├── plant/                     # Photonic hardware models
│   ├── mzi_mesh.py           # MZI transfer matrix
│   ├── thermal_dynamics.py   # Heater RC model + drift
│   ├── coherent_receiver.py  # I/Q detection + noise
│   ├── voa.py                # Variable attenuators
│   ├── svd_plant.py          # SVD architecture wrapper
│   └── plant_wrapper.py      # Unified interface
├── demo/                      # Control algorithms
│   ├── control_loop.py       # Standard (unitary) controller
│   ├── svd_control_loop.py   # SVD controller
│   ├── visualizer.py         # Plotting utilities
│   └── run_demo.py           # Demo with visualizations
├── rtl/                       # SystemVerilog (for cocotb)
└── sim/results/               # Experiment outputs
```

## Example Output

```
╔══════════════════════════════════════════════════════════╗
║     Coherent Photonic Matrix Multiply - Run Script       ║
╚══════════════════════════════════════════════════════════╝

Target Matrix:
  [[0.7000, -0.3000],
   [0.2000, -0.5000]]

Using SVD architecture (M = U·Σ·V†)

SVD Decomposition:
  Singular values: σ = [0.8713, 0.3328]

Calibrating photonic system...
  ✓ LOCKED in 5 iterations (error: 1.04e-05)

Computing y = M·x for input vectors:
────────────────────────────────────────────────────────────
  Input x          Photonic y       Reference y      Error
────────────────────────────────────────────────────────────
  [1.00, 0.50]     [0.55, -0.05]    [0.55, -0.05]    7.9e-04
────────────────────────────────────────────────────────────

Summary:
  Max absolute error: 7.5e-04
  RMS error: 5.6e-04
```

## Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Matrix size | 2×2 | Simplest case; scales via Clements/Reck |
| Weights | Real, any value | SVD + auto-scaling handles arbitrary matrices |
| Architecture | MZI + optional SVD | Unitary fast, SVD universal |
| Calibration | Coordinate descent | Noise-robust, hardware-friendly |
| Detection | Coherent I/Q | Preserves phase, enables signed values |
| Thermal model | Full RC + drift | Realistic control challenges |
| Large matrices | Auto-scale + software multiply | Photonic attenuation only; gain in software |

### Why Coherent Detection?

Direct detection (photodiode) measures power |E|², losing phase information. Coherent detection preserves the complex field, enabling:
- Signed matrix elements (positive and negative)
- Full complex matrix recovery
- Linear relationship between field and output

### Why Output-Driven Control?

**Alternative**: Pre-compute phase setpoints from target matrix.

**Problem**: Real hardware has:
- Manufacturing variations
- Temperature drift
- Aging effects
- Crosstalk

Output-driven control adapts to these automatically. No per-device calibration tables needed.

## Realizability

### Standard Mode (Unitary Only)

A matrix M is realizable if M†M = I (unitary). Check:
```python
np.allclose(M @ M.T, np.eye(2))  # Should be True
```

Examples:
- ✓ Identity, Swap, Hadamard, rotations
- ✗ [[0.5, 0], [0, 0.3]] (not unitary)

### SVD Mode (Universal with Auto-Scaling)

SVD mode can realize **any** 2×2 real matrix through automatic scaling:

```python
# Any matrix works - auto-scaling handles large values
python run.py --svd --weights 2.0 -1.5 3.0 -2.5 --input 1.0 0.5
```

**How it works:**
1. Compute SVD: M = U·Σ·V†
2. If max(σ) > 1.0, normalize: M_norm = M / max(σ)
3. Calibrate photonic system to realize M_norm
4. Scale outputs in software: y = max(σ) × y_photonic

**Example with auto-scaling:**
```
Target Matrix:
  [[2.0000, -1.5000],
   [3.0000, -2.5000]]

SVD Decomposition:
  Singular values: σ = [4.6356, 0.1079]

  ╔═══════════════════════════════════════════════════════╗
  ║  AUTO-SCALING APPLIED                                 ║
  ╠═══════════════════════════════════════════════════════╣
  ║  Original σ_max = 4.6356 > 1.0                        ║
  ║  Matrix normalized by factor: 4.6356                  ║
  ║  Photonic outputs will be scaled by 4.6356×           ║
  ╚═══════════════════════════════════════════════════════╝

Computing y = M·x for input vectors:
  (Photonic outputs scaled by 4.6356× to match original matrix)
```

This approach:
- Works for any real matrix (no restrictions on weight values)
- Preserves full precision of the photonic system
- Applies scaling in software (simple multiply)

## RTL Implementation

The `rtl/` directory contains SystemVerilog implementing the calibration controller for both unitary and SVD modes.

### Software/Hardware Boundary

| Component | Software (Python) | Hardware (RTL) |
|-----------|-------------------|----------------|
| SVD decomposition | ✓ `np.linalg.svd(M)` | - |
| σ → DAC conversion | ✓ `voa.sigma_to_dac()` | - |
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

### Why This Split?

- **SVD in software**: Matrix decomposition requires floating-point; doing it off-chip keeps RTL simple
- **Calibration in hardware**: Real-time feedback loop needs low latency; RTL handles coordinate descent
- **σ passed as DAC codes**: Avoids floating-point in RTL; software pre-computes the mapping

## License

MIT
