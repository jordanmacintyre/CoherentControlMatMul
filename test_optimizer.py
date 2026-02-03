#!/usr/bin/env python3
"""
Test the optimizer algorithm in pure Python to verify it can converge.

This removes the RTL from the equation to isolate the optimization algorithm.
"""

import numpy as np
from plant.mzi_mesh import MZIMesh
from plant.plant_wrapper import PhotonicPlant


def compute_error_q15(plant: PhotonicPlant, dac_codes: list[int], target_q15: np.ndarray) -> int:
    """
    Compute error exactly as the RTL does.

    Args:
        plant: Photonic plant model
        dac_codes: Current DAC codes for phases
        target_q15: Target matrix in Q1.15 format (4 elements)

    Returns:
        Sum of squared errors in Q1.15² units
    """
    # Measure transfer matrix
    # Column 0: input [1, 0]
    i0_c0, q0_c0, i1_c0, q1_c0 = plant.sample_outputs(dac_codes, complex(1, 0), complex(0, 0))
    # Column 1: input [0, 1]
    i0_c1, q0_c1, i1_c1, q1_c1 = plant.sample_outputs(dac_codes, complex(0, 0), complex(1, 0))

    # Scale ADC to Q1.15 (same as RTL: accumulate 8 samples, shift left 1)
    # For simplicity, assume single sample, so just multiply by 16
    m_q15 = np.array([
        i0_c0 * 16,  # M00
        i0_c1 * 16,  # M01
        i1_c0 * 16,  # M10
        i1_c1 * 16,  # M11
    ], dtype=np.int32)

    # Saturate to Q1.15 range
    m_q15 = np.clip(m_q15, -32768, 32767)

    # Compute error
    diff = m_q15 - target_q15
    error = np.sum(diff.astype(np.int64) ** 2)

    return int(error)


def bang_bang_optimizer(
    plant: PhotonicPlant,
    target_q15: np.ndarray,
    max_iterations: int = 2000,
    threshold: int = 1_000_000,
    step_initial: int = 0x2000,
    step_min: int = 0x0020,
    verbose: bool = True,
) -> tuple[list[int], list[int], bool]:
    """
    Simple bang-bang coordinate descent optimizer.

    No momentum, just probe ±delta and move in better direction.

    Key fix: Don't reduce step too aggressively. Keep searching at larger
    step sizes before reducing.
    """
    dac_codes = [0, 0, 0, 0]  # Start at identity
    step = step_initial

    errors = []

    # Let plant settle initially
    for _ in range(1000):
        plant.step(dac_codes, complex(0, 0), complex(0, 0))

    error_best = compute_error_q15(plant, dac_codes, target_q15)
    errors.append(error_best)

    if verbose:
        print(f"Initial error: {error_best:,}")

    no_improvement_count = 0
    iterations_at_step = 0  # Track how many iterations at this step size

    for iteration in range(max_iterations):
        improved_this_round = False
        iterations_at_step += 1

        for phase_idx in range(4):
            # Probe plus
            dac_plus = dac_codes.copy()
            dac_plus[phase_idx] = (dac_plus[phase_idx] + step) & 0xFFFF
            error_plus = compute_error_q15(plant, dac_plus, target_q15)

            # Probe minus
            dac_minus = dac_codes.copy()
            dac_minus[phase_idx] = (dac_minus[phase_idx] - step) & 0xFFFF
            error_minus = compute_error_q15(plant, dac_minus, target_q15)

            # Move in better direction
            if error_plus < error_minus and error_plus < error_best:
                dac_codes[phase_idx] = dac_plus[phase_idx]
                error_best = error_plus
                improved_this_round = True
            elif error_minus < error_best:
                dac_codes[phase_idx] = dac_minus[phase_idx]
                error_best = error_minus
                improved_this_round = True

        errors.append(error_best)

        if not improved_this_round:
            no_improvement_count += 1
            # Only reduce step if we've been stuck for a while AND
            # we've tried enough iterations at this step size
            if no_improvement_count >= 8 and step > step_min and iterations_at_step >= 16:
                step = step >> 1
                no_improvement_count = 0
                iterations_at_step = 0
                if verbose:
                    print(f"  Iteration {iteration}: reducing step to {step} (0x{step:04X}), error = {error_best:,}")
        else:
            no_improvement_count = 0

        if verbose and iteration % 200 == 0:
            print(f"  Iteration {iteration}: error = {error_best:,}, step = {step} (0x{step:04X})")

        if error_best < threshold:
            if verbose:
                print(f"✓ Converged at iteration {iteration}! Error = {error_best:,}")
            return dac_codes, errors, True

    if verbose:
        print(f"✗ Did not converge. Final error = {error_best:,}")
    return dac_codes, errors, False


def analyze_error_landscape():
    """Analyze why the optimizer gets stuck."""
    print("\n" + "=" * 60)
    print("ERROR LANDSCAPE ANALYSIS")
    print("=" * 60)

    mzi = MZIMesh(seed=42)

    # Hadamard decomposition
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    ideal_phases = MZIMesh.decompose_unitary(H)
    print(f"\nIdeal phases for Hadamard (radians):")
    print(f"  theta={ideal_phases[0]:.4f}, phi_in0={ideal_phases[1]:.4f}, "
          f"phi_in1={ideal_phases[2]:.4f}, phi_out={ideal_phases[3]:.4f}")

    # Convert to DAC codes
    ideal_dac = []
    for p in ideal_phases:
        p_wrapped = p % (2 * np.pi)
        dac = int(p_wrapped / (2 * np.pi) * 65535)
        ideal_dac.append(dac)
    print(f"Ideal DAC codes: {[f'0x{c:04X}' for c in ideal_dac]}")

    # Check what matrix we get at ideal phases
    M_ideal = mzi.compute_transfer_matrix(ideal_phases)
    print(f"\nMatrix at ideal phases:")
    print(f"  [[{M_ideal[0,0]:.4f}, {M_ideal[0,1]:.4f}],")
    print(f"   [{M_ideal[1,0]:.4f}, {M_ideal[1,1]:.4f}]]")

    # Check what matrix we get at zero phases
    M_zero = mzi.compute_transfer_matrix(np.array([0, 0, 0, 0]))
    print(f"\nMatrix at zero phases:")
    print(f"  [[{M_zero[0,0]:.4f}, {M_zero[0,1]:.4f}],")
    print(f"   [{M_zero[1,0]:.4f}, {M_zero[1,1]:.4f}]]")

    # The issue: we measure REAL part only, but need to match MAGNITUDE
    print(f"\n--- The Issue ---")
    print(f"Hadamard element magnitude: |0.7071| = 0.7071")
    print(f"Hadamard element (real): Re(0.7071) = 0.7071")
    print()
    print(f"MZI at zero phases produces identity:")
    print(f"  M[0,0] = 1.0 (real=1.0, magnitude=1.0)")
    print(f"  Target M[0,0] = 0.7071 (real=0.7071)")
    print()
    print(f"The optimizer uses I channel (real part) for error.")
    print(f"But MZI phases can produce complex values!")
    print()

    # Show how the matrix changes with theta only
    print(f"\n--- Effect of theta (rotation angle) ---")
    for theta in [0, 0.5, 1.0, 1.5, 1.5708]:
        M = mzi.compute_transfer_matrix(np.array([theta, 0, 0, 0]))
        print(f"theta={theta:.3f}: M[0,0]={M[0,0].real:+.3f}{M[0,0].imag:+.3f}j, "
              f"|M[0,0]|={np.abs(M[0,0]):.3f}")


def main():
    # First analyze the error landscape
    analyze_error_landscape()

    print("\n" + "=" * 60)
    print("OPTIMIZER VERIFICATION")
    print("=" * 60)
    print("Testing bang-bang optimizer in pure Python")
    print("(No RTL, just algorithm + plant model)")
    print("=" * 60)

    # Create plant with minimal noise for testing
    plant = PhotonicPlant(
        tau_thermal=100e-6,
        drift_rate=0.0,
        crosstalk_coeff=0.0,
        receiver_gain=2047.0,
        noise_std=0.0,
        seed=42,
    )

    # Target: Hadamard matrix in Q1.15
    # H = 1/sqrt(2) * [[+1, +1], [+1, -1]]
    # H[0,0] = +0.707 → Q1.15 = +23170
    # H[0,1] = +0.707 → Q1.15 = +23170
    # H[1,0] = +0.707 → Q1.15 = +23170
    # H[1,1] = -0.707 → Q1.15 = -23170  <-- NOTE THE SIGN!
    hadamard_q15 = np.array([23170, 23170, 23170, -23170], dtype=np.int32)

    print(f"\nTarget: Hadamard matrix")
    print(f"Q1.15 targets: {hadamard_q15}")
    print(f"Threshold: 1,000,000 (~3% RMS error)")

    print(f"\n--- Running optimizer ---")
    final_dac, errors, converged = bang_bang_optimizer(
        plant,
        hadamard_q15,
        max_iterations=2000,
        threshold=1_000_000,
    )

    print(f"\n--- Results ---")
    print(f"Final DAC codes: {[f'0x{c:04X}' for c in final_dac]}")
    print(f"Final error: {errors[-1]:,}")
    print(f"Converged: {converged}")

    # Verify the final matrix
    print(f"\n--- Verification ---")
    for _ in range(10000):
        plant.step(final_dac, complex(0, 0), complex(0, 0))

    i0_c0, _, i1_c0, _ = plant.sample_outputs(final_dac, complex(1, 0), complex(0, 0))
    i0_c1, _, i1_c1, _ = plant.sample_outputs(final_dac, complex(0, 0), complex(1, 0))

    print(f"Final measured matrix (ADC values):")
    print(f"  [[{i0_c0}, {i0_c1}],")
    print(f"   [{i1_c0}, {i1_c1}]]")

    # Convert to float
    m_float = np.array([[i0_c0, i0_c1], [i1_c0, i1_c1]]) / 2047.0
    print(f"\nFinal measured matrix (normalized):")
    print(f"  [[{m_float[0,0]:.4f}, {m_float[0,1]:.4f}],")
    print(f"   [{m_float[1,0]:.4f}, {m_float[1,1]:.4f}]]")

    print(f"\nTarget Hadamard matrix:")
    print(f"  [[0.7071, 0.7071],")
    print(f"   [0.7071, 0.7071]]  (magnitudes)")


if __name__ == "__main__":
    main()
