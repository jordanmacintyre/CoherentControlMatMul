# CoherentControlMatMul

Closed-loop coherent photonic matrix multiplication with output-driven phase control.

## Overview

This project implements a simulation of a **coherent (I/Q) 2×2 photonic matrix multiply block**:

- User specifies a target matrix **M** (weights w₀..w₃ in [-1, 1])
- A photonic fabric (MZI mesh with thermal phase shifters) implements a transform **M̂(φ)**
- A digital controller programs phases using **only coherent I/Q measurements** as feedback
- Once "locked," the system processes inputs **x** and produces outputs **y = M̂·x**

**Key principle**: User weights define the desired behavior, not phase setpoints. Phase setpoints are internal control variables continuously adjusted based on output error.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         cocotb Testbench                            │
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐   │
│  │ Test Driver  │───▶│  RTL Controller │◀──▶│ Python Plant     │   │
│  │ (weights, x) │    │   (Verilator)   │    │ (photonic model) │   │
│  └──────────────┘    └─────────────────┘    └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- Verilator 5.0+
- conda (for environment management)

### Setup

```bash
# Activate the photonics conda environment
conda activate photonics

# Install dependencies
pip install -e .
```

### Run the Demo

```bash
# Run the Python control loop demo with visualizations
python demo/run_demo.py

# Interactive mode
python demo/run_demo.py --interactive

# Animated demo (shows real-time calibration)
python demo/run_demo.py --animated

# Save plots without display (headless)
python demo/run_demo.py --no-display --save-plots
```

### Run RTL Simulation

```bash
# Run calibration tests with Verilator
make sim

# Run specific test modules
make test-cal      # Calibration tests
make test-compute  # Compute correctness tests
make test-drift    # Drift robustness tests

# Clean simulation artifacts
make clean-sim
```

## Project Structure

```
CoherentControlMatMul/
├── rtl/                           # SystemVerilog RTL
│   ├── pkg_types.sv               # Type definitions & parameters
│   └── coherent_matmul_top.sv     # Top-level controller
│
├── plant/                         # Python photonic plant model
│   ├── mzi_mesh.py                # MZI transfer functions
│   ├── thermal_dynamics.py        # Heater thermal model
│   ├── coherent_receiver.py       # I/Q detection + noise
│   └── plant_wrapper.py           # Unified interface
│
├── demo/                          # Interactive demonstration
│   ├── control_loop.py            # Python control algorithm
│   ├── visualizer.py              # Plotting utilities
│   └── run_demo.py                # Main demo script
│
├── tests/                         # cocotb testbenches
│   ├── test_calibration.py        # Calibration convergence
│   ├── test_compute.py            # Compute correctness
│   ├── test_drift.py              # Drift robustness
│   └── utils/                     # Test utilities
│
├── sim/                           # Simulation artifacts
│   ├── obj_dir/                   # Verilator build
│   └── results/                   # Test results & plots
│
├── Makefile                       # cocotb + Verilator build
└── pyproject.toml                 # Python package config
```

## How It Works

### 1. Target Matrix

The user specifies a 2×2 real matrix:

```
M = [[w₀, w₁],
     [w₂, w₃]]
```

where each weight wᵢ ∈ [-1, 1].

### 2. MZI Mesh

The photonic fabric uses an MZI (Mach-Zehnder Interferometer) mesh with 4 phase shifters:

- **θ**: Internal splitting angle (controls power distribution)
- **φ₀, φ₁**: Input port phases
- **φ_out**: Global output phase

The transfer matrix is:

```
M̂(φ) = e^{jφ_out} · [[e^{jφ₀}·cos(θ/2),  j·sin(θ/2)],
                      [j·sin(θ/2),         e^{jφ₁}·cos(θ/2)]]
```

### 3. Calibration Algorithm

The controller uses **coordinate descent** optimization:

1. **Measure matrix**: Apply basis inputs [1,0] and [0,1], measure I/Q outputs
2. **Compute error**: ‖M̂_real - M_target‖² + ‖M̂_imag‖²
3. **Update phases**: For each phase, try ±Δ and keep the direction that reduces error
4. **Converge**: Lock when error stays below threshold for N iterations

### 4. Thermal Dynamics

The plant model includes realistic thermal effects:

- **RC time constant**: First-order exponential response (~100μs)
- **Random drift**: Brownian motion phase drift
- **Crosstalk**: Coupling between adjacent heaters

### 5. Coherent Receiver

The I/Q detection model includes:

- **Transimpedance gain**: Converts optical field to voltage
- **Noise**: Thermal/shot noise
- **ADC quantization**: 12-bit resolution with clipping

## Standard Target Matrices

The demo (`demo/run_demo.py`) calibrates to three standard matrices that represent common linear transformations:

### Identity Matrix
```
M = [[1, 0],
     [0, 1]]
```
**Effect**: y = x (pass-through)
- Output equals input: y₀ = x₀, y₁ = x₁
- Useful as a baseline test - if calibration works, the system should reproduce inputs exactly
- **Weights**: `--weights 1 0 0 1`

### Swap Matrix
```
M = [[0, 1],
     [1, 0]]
```
**Effect**: Exchanges the two inputs
- y₀ = x₁, y₁ = x₀
- Tests the system's ability to route signals between ports
- Requires significant phase adjustment to redirect light paths
- **Weights**: `--weights 0 1 1 0`

### Hadamard Matrix
```
M = [[0.707,  0.707],
     [0.707, -0.707]]
```
**Effect**: Creates equal superposition with phase difference
- y₀ = (x₀ + x₁)/√2 (sum)
- y₁ = (x₀ - x₁)/√2 (difference)
- Fundamental operation in signal processing and quantum computing
- Tests precise amplitude control (50/50 splitting)
- **Weights**: `--weights 0.707 0.707 0.707 -0.707`

### Custom Matrices

The `run.py` script accepts **any** user-specified matrix via `--weights`. Examples:

```bash
# Attenuator (50% gain)
python run.py --weights 0.5 0 0 0.5 --input 1.0 1.0

# Rotation-like matrix
python run.py --weights 0.866 -0.5 0.5 0.866 --input 1.0 0.0

# Asymmetric mixing
python run.py --weights 0.8 0.2 -0.3 0.9 --input 0.5 0.5
```

### Realizability Constraints

**Not all 2×2 matrices can be exactly realized** by the MZI mesh due to physical constraints.

The MZI mesh implements **unitary** (or scaled unitary) transformations. A matrix is unitary if M†M = I, meaning:
- Columns are orthonormal: |col₀|² = |col₁|² = 1, and col₀ · col₁ = 0
- Rows are orthonormal: |row₀|² = |row₁|² = 1, and row₀ · row₁ = 0

**Examples of realizable matrices** (unitary or scaled unitary):
```
Identity:  [[1, 0], [0, 1]]           ✓ Unitary
Swap:      [[0, 1], [1, 0]]           ✓ Unitary
Hadamard:  [[0.707, 0.707],           ✓ Unitary
            [0.707, -0.707]]
Rotation:  [[cos θ, -sin θ],          ✓ Unitary
            [sin θ,  cos θ]]
```

**Examples of non-realizable matrices**:
```
[[0.91, -0.32],    ✗ Not unitary (rows not orthogonal)
 [0.15, -0.74]]

[[0.5, 0], [0, 0.5]]  ✓ Scaled identity (realizable with loss)
[[0.8, 0.2], [0.3, 0.9]]  ✗ Not unitary
```

**How to check**: A matrix M is approximately unitary if:
- `np.allclose(M @ M.T, np.eye(2))` for real matrices
- Row/column norms ≈ 1

When calibration fails to lock, the target matrix likely violates unitarity constraints. The system will find the **closest achievable** approximation, but with high residual error.

## Main Run Script

The `run.py` script provides a complete workflow: specify a target matrix, calibrate the photonic system, compute matrix-vector products, and compare against floating-point reference.

```bash
# Basic usage: specify matrix weights and input vector
python run.py --weights 0.707 0.707 0.707 -0.707 --input 1.0 0.5

# Multiple input vectors
python run.py --weights 1 0 0 1 --input 1.0 0.0 --input 0.5 0.5 --input -1.0 1.0

# Custom noise and save results
python run.py --weights 0.5 -0.3 0.8 0.2 --input 1.0 1.0 --noise 5.0 --save-results

# Verbose output with all intermediate values
python run.py --weights 0 1 1 0 --input 0.5 0.5 --verbose
```

### Output Example

```
╔══════════════════════════════════════════════════════════╗
║     Coherent Photonic Matrix Multiply - Run Script       ║
╚══════════════════════════════════════════════════════════╝

Target Matrix:
  [[ 0.7070,  0.7070],
   [ 0.7070, -0.7070]]

Calibrating photonic system...
  ✓ LOCKED in 25 iterations (error: 1.43e-03)

Computing y = M·x for input vectors:
──────────────────────────────────────────────────────────
  Input x         Photonic y       Reference y      Error
──────────────────────────────────────────────────────────
  [1.00, 0.50]    [1.06, 0.35]     [1.06, 0.35]    2.1e-03
──────────────────────────────────────────────────────────

Summary:
  Max absolute error: 2.1e-03
  RMS error: 1.8e-03
```

## Output Plots

When running with `--save-plots`, the demo generates several visualization plots saved to timestamped experiment directories under `sim/results/`.

### Summary Plot (6-panel overview)

The main summary plot contains six panels providing a complete view of the calibration:

```
┌─────────────────┬─────────────────┬─────────────────┐
│   Convergence   │  Phases φ₀-φ₁  │  Phases φ₂-φ₃  │
├─────────────────┼─────────────────┼─────────────────┤
│  Matrix Heatmap │   M₀₀ I/Q      │   M₁₁ I/Q      │
└─────────────────┴─────────────────┴─────────────────┘
```

**Panel 1: Convergence** (top-left)
- **Y-axis (log scale)**: Calibration error = ‖M̂_real - M_target‖² + ‖M̂_imag‖²
- **X-axis**: Iteration number
- **Red dashed line**: Error threshold for lock condition
- **What to look for**: Exponential decay followed by plateau below threshold indicates successful calibration

**Panels 2-3: Phase Trajectories** (top-center, top-right)
- **Y-axis**: Phase value in radians
- **X-axis**: Iteration number
- **Four phases**: θ (splitting angle), φ₀ (input 0), φ₁ (input 1), φ_out (global output)
- **What to look for**: Phases should stabilize as error decreases; oscillations indicate step size issues

**Panel 4: Final Matrix Heatmap** (bottom-left)
- **Color scale**: Red-Blue diverging colormap, range [-1, 1]
- **Each cell shows**: Measured value (top) and target value (bottom)
- **What to look for**: Colors should match target; near-zero error means good calibration

**Panels 5-6: I/Q Constellation Diagrams** (bottom-center, bottom-right)
- **Axes**: In-phase (I) vs Quadrature (Q) components
- **Green circle**: Starting point (random initial phases)
- **Red square**: Final converged point
- **Blue star**: Target value (always on real axis since targets are real)
- **Gray dashed circle**: Unit circle (|z| = 1)
- **Color gradient**: Time progression (purple → yellow)
- **What to look for**: Trajectory should spiral toward the blue target star; final Q component should be near zero

### Understanding the I/Q Plane

The coherent receiver measures both In-phase (I) and Quadrature (Q) components of the optical field. For a **real** target matrix:
- **Target position**: Always on the real axis (Q = 0)
- **Ideal result**: Final measurement lands on target with Q ≈ 0
- **Imaginary residual**: Non-zero Q indicates phase error in the photonic circuit

### Experiment Data Files

Each calibration run saves:
- `calibration_<name>.png`: Summary plot for that target matrix
- `calibration_<name>.json`: Machine-readable data including:
  - Target weights and name
  - Final phases (φ₀, φ₁, φ₂, φ₃)
  - Convergence iterations and final error
  - Controller configuration
- `summary.json`: Aggregated results for all matrices in the run

## Example Results

Running the demo produces calibration results like:

```
Target: Identity
  M = [[ 1.0000,  0.0000],
       [ 0.0000,  1.0000]]
  ✓ LOCKED in 24 iterations
    Final error: 6.47e-04

Target: Hadamard
  M = [[ 0.7070,  0.7070],
       [ 0.7070, -0.7070]]
  ✓ LOCKED in 25 iterations
    Final error: 1.43e-03
```

## Design Decisions

### Summary Table

| Decision | Choice | Alternatives Considered |
|----------|--------|------------------------|
| Matrix size | 2×2 only | N×N generalizable |
| Weight type | Real-only [-1, 1] | Complex weights |
| Architecture | Single MZI mesh | SVD (U·Σ·V†) decomposition |
| Calibration | Coordinate descent | Gradient descent, analytical |
| Detection | Coherent I/Q | Direct detection (power only) |
| Thermal model | Full dynamics | Instantaneous or simplified |
| Fixed-point | Q1.15 weights, Q2.14 phase | Floating-point |

### Detailed Rationale

#### Why 2×2 Matrix Size?

**Choice**: Fixed 2×2 matrix multiplication

**Rationale**:
- Simplest non-trivial case that demonstrates all key concepts
- Single MZI naturally implements 2×2 unitary transforms
- Scales to N×N via Clements/Reck decomposition (cascaded 2×2 blocks)
- Keeps RTL complexity manageable for demonstration

**Trade-off**: Larger matrices (4×4, 8×8) would be more practical for neural network inference but require O(N²) MZIs and significantly more complex calibration.

#### Why Real-Only Weights?

**Choice**: Weights constrained to w ∈ [-1, 1] (real numbers only)

**Rationale**:
- Most neural network weights are real-valued
- Simplifies user interface (4 floats vs 8 for complex)
- Calibration targets real axis in I/Q plane (Q = 0)
- Reduces calibration degrees of freedom

**Trade-off**: Cannot implement arbitrary phase shifts in the matrix. For applications requiring complex weights (e.g., Fourier transforms), would need to extend to complex targets with separate I/Q weight inputs.

#### Why Single MZI Mesh (Not SVD Decomposition)?

**Choice**: Single MZI mesh that can only realize unitary matrices

**Alternatives**:
1. **SVD decomposition**: M = U·Σ·V† using two MZI meshes + variable attenuators
2. **Coherent + incoherent hybrid**: MZI for unitary, photodetector/modulator for scaling

**Rationale**:
- Demonstrates core coherent control principles without added complexity
- Many practical matrices (rotations, Hadamard, permutations) are already unitary
- For neural networks, weights can be trained with unitarity constraints
- Avoids need for variable optical attenuators (VOAs) in the model

**Trade-off**: Cannot exactly realize non-unitary matrices. System approximates with residual error. Production systems typically use SVD decomposition for arbitrary matrix support.

#### Why Coordinate Descent (Not Gradient Descent)?

**Choice**: Coordinate descent with adaptive step sizing

**Alternatives**:
1. **Gradient descent**: Update all phases simultaneously based on gradient
2. **Analytical solution**: Directly compute phases from target matrix
3. **Particle swarm / genetic algorithms**: Global optimization

**Rationale**:
- **Robust to noise**: Discrete ±Δ probing is less sensitive to measurement noise than gradient estimation
- **No gradient computation**: Avoids numerical differentiation issues
- **Hardware-friendly**: Simple comparisons, no floating-point division
- **Adaptive**: Step size decay prevents oscillation near optimum

**Why not analytical?** The mapping from target matrix to phases is non-linear and affected by manufacturing variations, thermal drift, and crosstalk. Analytical solutions assume ideal hardware; closed-loop calibration adapts to real conditions.

**Why not gradient descent?** Gradient estimation via finite differences requires 2N measurements per iteration (for N phases). Coordinate descent with multi-step probing achieves similar convergence with better noise immunity.

#### Why Coherent (I/Q) Detection?

**Choice**: Coherent receiver measuring both In-phase (I) and Quadrature (Q) components

**Alternative**: Direct detection (photodiode measuring optical power only)

**Rationale**:
- **Phase information**: Coherent detection preserves phase, essential for complex matrix elements
- **Linear in field**: Output proportional to electric field amplitude, not intensity
- **Sign preservation**: Can distinguish positive and negative matrix elements
- **Full matrix recovery**: Can measure all 4 complex elements of 2×2 matrix

**Trade-off**: Requires local oscillator (LO) laser and more complex receiver. Direct detection is simpler but loses phase information—would require interferometric techniques to recover matrix elements.

#### Why Full Thermal Dynamics?

**Choice**: Complete thermal model with RC time constant, drift, and crosstalk

**Alternatives**:
1. **Instantaneous**: Phase follows DAC immediately
2. **RC only**: First-order response, no drift or crosstalk
3. **Lookup table**: Pre-characterized phase-vs-DAC curves

**Rationale**:
- **Realistic control challenges**: Exposes timing constraints and stability issues
- **Drift compensation**: Tests controller's ability to track slow variations
- **Crosstalk handling**: Adjacent heater coupling is significant in integrated photonics (~5%)
- **Verification value**: Catches bugs that wouldn't appear with ideal models

**Parameters chosen**:
- τ_thermal = 100μs (typical for silicon photonic heaters)
- Drift rate = 0.001 rad/s (slow environmental changes)
- Crosstalk = 5% (typical for closely-spaced heaters)

#### Why Q1.15 and Q2.14 Fixed-Point?

**Choice**:
- Weights: Q1.15 (1 sign bit, 15 fractional bits) → range [-1, 1)
- Phases: Q2.14 (1 sign bit, 1 integer bit, 14 fractional bits) → range [-2, 2)

**Rationale**:
- **Hardware efficiency**: Fixed-point arithmetic is smaller/faster than floating-point in RTL
- **Sufficient precision**: Q1.15 resolution is 3×10⁻⁵, adequate for ~0.01% weight accuracy
- **Natural mapping**: Q1.15 directly represents [-1, 1] weight range
- **Phase range**: Q2.14 covers [0, 2π) with ~0.4 mrad resolution

**Trade-off**: Limited dynamic range compared to floating-point. For weights near zero, relative precision decreases. Acceptable for neural network inference where weights are typically O(1).

#### Why Output-Driven (Not Setpoint-Based) Control?

**Choice**: Controller adjusts phases based on measured output error, not pre-computed phase setpoints

**Alternative**: Characterize MZI transfer function, compute phase setpoints analytically

**Rationale**:
- **Adapts to hardware variations**: No need for per-device calibration tables
- **Tracks drift**: Continuous feedback compensates environmental changes
- **Handles crosstalk**: Multi-variable optimization accounts for phase coupling
- **Robust to aging**: Re-calibration automatically adjusts to device degradation

**Trade-off**: Requires calibration time before each use (or after disturbance). Setpoint-based control would be faster for static conditions but fails when hardware deviates from model.

### Industry Context

**How production systems handle these trade-offs**:

| Aspect | This Project | Production Systems |
|--------|--------------|-------------------|
| Matrix size | 2×2 | 64×64 to 256×256 |
| Architecture | Single MZI | SVD (Clements/Reck mesh) |
| Arbitrary matrices | Unitary only | Full via SVD + VOAs |
| Calibration | Software coordinate descent | Hardware-accelerated or hybrid |
| Detection | Coherent | Often coherent with balanced detection |
| Control bandwidth | ~kHz (demo) | ~MHz (production) |

**Key insight**: This project demonstrates the fundamental principles. Scaling to production requires:
1. Larger MZI meshes with SVD decomposition
2. Hardware-accelerated calibration loops
3. Higher-bandwidth thermal control
4. Manufacturing process optimization for lower crosstalk

## License

MIT License - see [LICENSE](LICENSE)
