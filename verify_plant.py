#!/usr/bin/env python3
"""
Verify that the plant model can achieve target matrices.

This script tests the simulation correctness by:
1. Computing the ideal MZI phases for a target unitary (e.g., Hadamard)
2. Setting those phases directly in the plant
3. Measuring the resulting transfer matrix
4. Comparing against the target

If this fails, the simulation is fundamentally broken.
If this passes, the problem is controller tuning.
"""

import numpy as np
from plant.mzi_mesh import MZIMesh
from plant.plant_wrapper import PhotonicPlant


def test_mzi_decomposition():
    """Test that MZI can decompose and reconstruct unitaries."""
    print("=" * 60)
    print("Test 1: MZI Decomposition/Reconstruction")
    print("=" * 60)

    mzi = MZIMesh(seed=42)

    # Hadamard matrix (normalized)
    H = np.array([
        [1, 1],
        [1, -1]
    ], dtype=np.complex128) / np.sqrt(2)

    print(f"\nTarget Hadamard matrix:")
    print(f"  [[{H[0,0].real:+.4f}, {H[0,1].real:+.4f}],")
    print(f"   [{H[1,0].real:+.4f}, {H[1,1].real:+.4f}]]")

    # Decompose into MZI phases
    phases = MZIMesh.decompose_unitary(H)
    print(f"\nDecomposed phases (radians):")
    print(f"  theta   = {phases[0]:.4f} ({np.degrees(phases[0]):.1f}°)")
    print(f"  phi_in0 = {phases[1]:.4f} ({np.degrees(phases[1]):.1f}°)")
    print(f"  phi_in1 = {phases[2]:.4f} ({np.degrees(phases[2]):.1f}°)")
    print(f"  phi_out = {phases[3]:.4f} ({np.degrees(phases[3]):.1f}°)")

    # Reconstruct matrix from phases
    M_reconstructed = mzi.compute_transfer_matrix(phases)

    print(f"\nReconstructed matrix:")
    print(f"  [[{M_reconstructed[0,0].real:+.4f}{M_reconstructed[0,0].imag:+.4f}j, "
          f"{M_reconstructed[0,1].real:+.4f}{M_reconstructed[0,1].imag:+.4f}j],")
    print(f"   [{M_reconstructed[1,0].real:+.4f}{M_reconstructed[1,0].imag:+.4f}j, "
          f"{M_reconstructed[1,1].real:+.4f}{M_reconstructed[1,1].imag:+.4f}j]]")

    # Compute error (Frobenius norm)
    error = np.linalg.norm(M_reconstructed - H, 'fro')
    print(f"\nFrobenius norm error: {error:.6f}")

    # Check if magnitudes match (ignoring global phase)
    mag_target = np.abs(H)
    mag_reconstructed = np.abs(M_reconstructed)
    mag_error = np.linalg.norm(mag_target - mag_reconstructed, 'fro')
    print(f"Magnitude-only error: {mag_error:.6f}")

    if error < 0.01:
        print("\n✓ PASS: MZI can exactly represent Hadamard")
        return True
    elif mag_error < 0.01:
        print("\n✓ PASS: MZI achieves correct magnitudes (global phase differs)")
        return True
    else:
        print("\n✗ FAIL: MZI cannot represent Hadamard")
        return False


def test_plant_with_ideal_phases():
    """Test plant with ideal phases (no thermal dynamics)."""
    print("\n" + "=" * 60)
    print("Test 2: Plant with Ideal Phases")
    print("=" * 60)

    # Create plant with NO noise for this test
    plant = PhotonicPlant(
        tau_thermal=100e-6,
        drift_rate=0.0,  # No drift
        crosstalk_coeff=0.0,  # No crosstalk
        receiver_gain=2047.0,
        noise_std=0.0,  # No receiver noise
        adc_bits=12,
        dac_bits=16,
        seed=42
    )

    # Hadamard matrix
    H = np.array([
        [1, 1],
        [1, -1]
    ], dtype=np.complex128) / np.sqrt(2)

    # Get ideal phases
    phases = MZIMesh.decompose_unitary(H)

    # Convert phases to DAC codes
    # phase_range = 2*pi, dac_max = 65535
    # Handle negative phases by wrapping to [0, 2pi)
    dac_max = (1 << 16) - 1
    dac_codes = []
    for p in phases:
        # Wrap phase to [0, 2*pi)
        p_wrapped = p % (2 * np.pi)
        dac_code = int(p_wrapped / (2 * np.pi) * dac_max)
        dac_codes.append(dac_code)

    print(f"\nDAC codes for Hadamard:")
    print(f"  theta   = {dac_codes[0]} (0x{dac_codes[0]:04X})")
    print(f"  phi_in0 = {dac_codes[1]} (0x{dac_codes[1]:04X})")
    print(f"  phi_in1 = {dac_codes[2]} (0x{dac_codes[2]:04X})")
    print(f"  phi_out = {dac_codes[3]} (0x{dac_codes[3]:04X})")

    # Let thermal dynamics settle (run many steps)
    # tau = 100us, dt = 10ns, so 1 tau = 10,000 steps
    # Need ~5 tau for 99% settling = 50,000 steps
    print(f"\nSettling thermal dynamics (5 tau = 50,000 steps)...")
    for _ in range(50000):
        plant.step(dac_codes, complex(0, 0), complex(0, 0))

    # Print actual phases after settling
    actual_phases = plant.thermal.get_phases()
    print(f"\nActual phases after settling:")
    print(f"  theta   = {actual_phases[0]:.4f} (target: {phases[0]:.4f})")
    print(f"  phi_in0 = {actual_phases[1]:.4f} (target: {phases[1]:.4f})")
    print(f"  phi_in1 = {actual_phases[2]:.4f} (target: {phases[2]:.4f})")
    print(f"  phi_out = {actual_phases[3]:.4f} (target: {phases[3]:.4f})")

    # Measure transfer matrix by probing with basis vectors
    print(f"\nMeasuring transfer matrix...")

    # Probe with [1, 0]
    y0_col0, y1_col0 = plant.step(dac_codes, complex(1, 0), complex(0, 0))

    # Probe with [0, 1]
    y0_col1, y1_col1 = plant.step(dac_codes, complex(0, 0), complex(1, 0))

    M_measured = np.array([
        [y0_col0, y0_col1],
        [y1_col0, y1_col1]
    ], dtype=np.complex128)

    print(f"\nMeasured matrix (complex):")
    print(f"  [[{M_measured[0,0].real:+.4f}{M_measured[0,0].imag:+.4f}j, "
          f"{M_measured[0,1].real:+.4f}{M_measured[0,1].imag:+.4f}j],")
    print(f"   [{M_measured[1,0].real:+.4f}{M_measured[1,0].imag:+.4f}j, "
          f"{M_measured[1,1].real:+.4f}{M_measured[1,1].imag:+.4f}j]]")

    print(f"\nMeasured matrix (magnitudes):")
    print(f"  [[{np.abs(M_measured[0,0]):.4f}, {np.abs(M_measured[0,1]):.4f}],")
    print(f"   [{np.abs(M_measured[1,0]):.4f}, {np.abs(M_measured[1,1]):.4f}]]")

    print(f"\nTarget Hadamard (magnitudes):")
    print(f"  [[{np.abs(H[0,0]):.4f}, {np.abs(H[0,1]):.4f}],")
    print(f"   [{np.abs(H[1,0]):.4f}, {np.abs(H[1,1]):.4f}]]")

    # Compute error
    mag_error = np.linalg.norm(np.abs(M_measured) - np.abs(H), 'fro')
    print(f"\nMagnitude error (Frobenius): {mag_error:.6f}")

    # Expected: 0.707 per element
    expected_mag = 1.0 / np.sqrt(2)
    per_element_errors = np.abs(np.abs(M_measured) - expected_mag)
    print(f"Per-element magnitude errors: {per_element_errors.flatten()}")

    if mag_error < 0.05:
        print("\n✓ PASS: Plant achieves Hadamard with ideal phases")
        return True
    else:
        print("\n✗ FAIL: Plant does not achieve Hadamard even with ideal phases")
        return False


def test_adc_scaling():
    """Test that ADC scaling produces correct Q1.15 values."""
    print("\n" + "=" * 60)
    print("Test 3: ADC Scaling Chain")
    print("=" * 60)

    # Create plant
    plant = PhotonicPlant(
        receiver_gain=2047.0,
        noise_std=0.0,
        seed=42
    )

    # Identity matrix (theta=0, all other phases=0)
    # For identity: output = input
    dac_codes = [0, 0, 0, 0]

    # Settle
    for _ in range(1000):
        plant.step(dac_codes, complex(0, 0), complex(0, 0))

    # Input [1, 0] - should get output [1, 0] (roughly)
    # With identity matrix and input (1+0j, 0+0j):
    #   output should be approximately (1+0j, 0+0j)

    # Get ADC samples
    i0, q0, i1, q1 = plant.sample_outputs(dac_codes, complex(1, 0), complex(0, 0))

    print(f"\nInput: [1.0+0j, 0.0+0j]")
    print(f"ADC outputs: i0={i0}, q0={q0}, i1={i1}, q1={q1}")

    # Expected: i0 ≈ 2047 (1.0 optical → full scale ADC)
    #           q0 ≈ 0 (no imaginary component)
    #           i1, q1 ≈ 0 (no power in output 1)

    print(f"\nExpected for 1.0 optical: i0 ≈ 2047")
    print(f"Actual i0: {i0}")
    print(f"Ratio: {i0 / 2047:.4f}")

    # What Q1.15 value would this produce?
    # With 8-sample accumulation and <<1 scaling:
    # q15 = (8 * adc) << 1 = 16 * adc
    q15_expected = 16 * i0
    print(f"\nExpected Q1.15 (16 * ADC): {q15_expected}")
    print(f"Q1.15 full scale: 32767")
    print(f"Ratio to full scale: {q15_expected / 32767:.4f}")

    # For identity matrix, we expect m00 = 1.0, m01 = 0, m10 = 0, m11 = 1.0
    # So Q1.15 for m00 should be ~32767

    if abs(i0 - 2047) < 100:
        print("\n✓ PASS: ADC scaling is correct")
        return True
    else:
        print(f"\n✗ FAIL: ADC value {i0} far from expected 2047")
        return False


def test_error_computation():
    """Test the error computation math."""
    print("\n" + "=" * 60)
    print("Test 4: Error Computation")
    print("=" * 60)

    # Simulate what the RTL does:
    # error = sum((measured_q15[i] - target_q15[i])^2)

    # Target: Hadamard ≈ 0.707 per element
    # Q1.15 for 0.707: int(0.707 * 32767) = 23170
    target_q15 = 23170

    # If measured is perfect:
    measured_q15 = 23170
    error_perfect = 4 * (measured_q15 - target_q15) ** 2
    print(f"Perfect match error: {error_perfect}")

    # If measured is 1% off:
    measured_q15_1pct = int(23170 * 1.01)
    error_1pct = 4 * (measured_q15_1pct - target_q15) ** 2
    print(f"1% error: {error_1pct}")

    # If measured is 10% off:
    measured_q15_10pct = int(23170 * 1.10)
    error_10pct = 4 * (measured_q15_10pct - target_q15) ** 2
    print(f"10% error: {error_10pct}")

    # If measured is 50% off (e.g., 0.35 instead of 0.707):
    measured_q15_50pct = int(23170 * 0.5)
    error_50pct = 4 * (measured_q15_50pct - target_q15) ** 2
    print(f"50% error: {error_50pct}")

    # If measured is completely wrong (e.g., 0 instead of 0.707):
    measured_q15_wrong = 0
    error_wrong = 4 * (measured_q15_wrong - target_q15) ** 2
    print(f"Completely wrong (0 vs 0.707): {error_wrong}")

    # What does 500M error mean?
    print(f"\n--- Interpreting 500M error ---")
    # error = 4 * diff^2 = 500M
    # diff^2 = 125M
    # diff = 11180 LSB
    diff = np.sqrt(500_000_000 / 4)
    print(f"500M error implies per-element diff: {diff:.0f} LSB")
    print(f"As fraction of target (23170): {diff/23170:.1%}")

    # What about 2B error (what we're seeing)?
    diff_2b = np.sqrt(2_000_000_000 / 4)
    print(f"\n2B error implies per-element diff: {diff_2b:.0f} LSB")
    print(f"As fraction of full scale (32767): {diff_2b/32767:.1%}")

    print("\n✓ Error computation analysis complete")
    return True


def test_initial_phases():
    """Test what matrix we get with all phases at zero."""
    print("\n" + "=" * 60)
    print("Test 5: Initial State (All Phases Zero)")
    print("=" * 60)

    mzi = MZIMesh(seed=42)

    # All phases at zero
    phases = np.array([0.0, 0.0, 0.0, 0.0])
    M = mzi.compute_transfer_matrix(phases)

    print(f"\nTransfer matrix with phases = [0, 0, 0, 0]:")
    print(f"  [[{M[0,0].real:+.4f}{M[0,0].imag:+.4f}j, {M[0,1].real:+.4f}{M[0,1].imag:+.4f}j],")
    print(f"   [{M[1,0].real:+.4f}{M[1,0].imag:+.4f}j, {M[1,1].real:+.4f}{M[1,1].imag:+.4f}j]]")

    print(f"\nMagnitudes:")
    print(f"  [[{np.abs(M[0,0]):.4f}, {np.abs(M[0,1]):.4f}],")
    print(f"   [{np.abs(M[1,0]):.4f}, {np.abs(M[1,1]):.4f}]]")

    # Expected for Hadamard: all ~0.707
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)

    # Compute initial error to Hadamard
    # Error = Σ(M[i,j] - H[i,j])² in Q1.15
    print(f"\nError to Hadamard target:")
    error = 0
    for i in range(2):
        for j in range(2):
            m_q15 = int(np.abs(M[i,j]) * 32767)
            h_q15 = int(np.abs(H[i,j]) * 32767)
            diff = m_q15 - h_q15
            error += diff * diff
            print(f"  M[{i},{j}]: measured_q15={m_q15}, target_q15={h_q15}, diff={diff}")

    print(f"\nInitial error (phases=0, target=Hadamard): {error:,}")
    print(f"Threshold is 1,048,576 (1M)")
    print(f"This is {'above' if error > 1048576 else 'below'} the threshold")

    return True


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("PLANT MODEL VERIFICATION")
    print("=" * 60)
    print("Testing whether the simulation is fundamentally correct")
    print("=" * 60)

    results = []

    results.append(("MZI Decomposition", test_mzi_decomposition()))
    results.append(("Plant with Ideal Phases", test_plant_with_ideal_phases()))
    results.append(("ADC Scaling", test_adc_scaling()))
    results.append(("Error Computation", test_error_computation()))
    results.append(("Initial State", test_initial_phases()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed. Simulation is correct.")
        print("Problem is controller tuning, not simulation.")
    else:
        print("Some tests failed. Simulation has bugs that need fixing.")

    return all_passed


if __name__ == "__main__":
    main()
