#!/usr/bin/env python3
"""
Main run script for coherent photonic matrix multiply.

Workflow:
1. User specifies target matrix weights (w0, w1, w2, w3)
2. System calibrates the photonic circuit to realize the target matrix
3. User provides input vectors x
4. System computes y = M·x using the photonic circuit
5. Results are compared against floating-point reference

Usage:
    python run.py --weights 0.707 0.707 0.707 -0.707 --input 1.0 0.5
    python run.py --weights 1 0 0 1 --input 1.0 0.0 --input 0.5 0.5
    python run.py --weights 0.5 -0.3 0.8 0.2 --input 1.0 1.0 --verbose --save-results
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from plant import PhotonicPlant, SVDPhotonicPlant


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types and complex numbers."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        if isinstance(obj, np.complexfloating):
            return {"real": float(obj.real), "imag": float(obj.imag)}
        return super().default(obj)
from demo.control_loop import CoherentController, ControlConfig
from demo.svd_control_loop import SVDCoherentController, SVDControlConfig
from demo.visualizer import ControlVisualizer


def print_banner():
    """Print script banner."""
    print()
    print("\u2554" + "\u2550" * 58 + "\u2557")
    print("\u2551     Coherent Photonic Matrix Multiply - Run Script       \u2551")
    print("\u255a" + "\u2550" * 58 + "\u255d")
    print()


def validate_weights(weights: list[float], svd_mode: bool = False) -> bool:
    """
    Validate weights for the given mode.

    In standard (unitary) mode, weights must be in [-1, 1].
    In SVD mode, any weights are allowed (auto-scaling handles large values).
    """
    if svd_mode:
        # SVD mode allows any weights - auto-scaling will handle normalization
        return True

    for i, w in enumerate(weights):
        if not -1.0 <= w <= 1.0:
            print(f"Error: Weight w{i} = {w} is outside valid range [-1, 1]")
            print("  Use --svd mode for matrices with weights outside [-1, 1]")
            return False
    return True


def format_vector(v: list[float] | np.ndarray, precision: int = 4) -> str:
    """Format a vector for display."""
    return "[" + ", ".join(f"{x:.{precision}f}" for x in v) + "]"


def format_matrix(m: np.ndarray, precision: int = 4, indent: int = 2) -> str:
    """Format a matrix for display."""
    lines = []
    prefix = " " * indent
    lines.append(f"{prefix}[[{m[0, 0]:.{precision}f}, {m[0, 1]:.{precision}f}],")
    lines.append(f"{prefix} [{m[1, 0]:.{precision}f}, {m[1, 1]:.{precision}f}]]")
    return "\n".join(lines)


def run_calibration(
    weights: list[float],
    plant: PhotonicPlant | SVDPhotonicPlant,
    config: ControlConfig | SVDControlConfig,
    verbose: bool = False,
) -> tuple[bool, CoherentController | SVDCoherentController]:
    """
    Calibrate the photonic system to realize the target matrix.

    Args:
        weights: Target matrix weights [w0, w1, w2, w3]
        plant: PhotonicPlant or SVDPhotonicPlant instance
        config: Control configuration
        verbose: Print detailed progress

    Returns:
        Tuple of (locked, controller)
    """
    # Choose controller type based on plant type
    if isinstance(plant, SVDPhotonicPlant):
        controller = SVDCoherentController(plant, config)

        def progress_callback(state):
            if verbose and state.stage_iteration % 10 == 0:
                err = state.error_history[-1] if state.error_history else float("inf")
                print(f"    [{state.stage}] Iter {state.stage_iteration:3d}: error = {err:.4e}")
    else:
        controller = CoherentController(plant, config)

        def progress_callback(state):
            if verbose and state.iteration % 10 == 0:
                print(f"    Iteration {state.iteration:3d}: error = {state.error:.4e}")

    controller.set_target(*weights)
    locked = controller.run_calibration(callback=progress_callback if verbose else None)

    return locked, controller


def compute_outputs(
    controller: CoherentController,
    inputs: list[list[float]],
    verbose: bool = False,
    scale_factor: float = 1.0,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute outputs for given input vectors.

    Args:
        controller: Calibrated controller
        inputs: List of input vectors [x0, x1]
        verbose: Print detailed output
        scale_factor: Factor to scale photonic outputs (for auto-normalized matrices)

    Returns:
        List of (input, photonic_output, reference_output) tuples
    """
    target = controller.target_matrix
    results = []

    for x in inputs:
        x_arr = np.array(x)

        # Photonic computation (scaled back to original matrix space)
        y_photonic_raw = np.array(controller.evaluate(x[0], x[1]))
        y_photonic = y_photonic_raw * scale_factor

        # Reference (floating-point) computation using ORIGINAL target matrix
        # Note: controller.target_matrix is the normalized matrix,
        # so we multiply by scale_factor to get the original
        y_reference = (target * scale_factor) @ x_arr

        results.append((x_arr, y_photonic, y_reference))

        if verbose:
            error = np.abs(y_photonic - y_reference)
            print(f"  Input: {format_vector(x_arr)}")
            print(f"    Photonic:  {format_vector(y_photonic)}")
            print(f"    Reference: {format_vector(y_reference)}")
            print(f"    Error:     {format_vector(error)}")
            print()

    return results


def compute_error_metrics(
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> dict:
    """Compute error metrics from results."""
    all_errors = []

    for x, y_photonic, y_reference in results:
        error = np.abs(y_photonic - y_reference)
        all_errors.extend(error.tolist())

    all_errors = np.array(all_errors)

    return {
        "max_error": float(np.max(all_errors)),
        "mean_error": float(np.mean(all_errors)),
        "rms_error": float(np.sqrt(np.mean(all_errors**2))),
        "num_samples": len(all_errors),
    }


def create_experiment_dir(base_name: str = "run") -> Path:
    """Create a timestamped experiment directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_base = Path(__file__).parent / "sim" / "results"
    experiment_dir = results_base / f"{base_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def save_results(
    weights: list[float],
    inputs: list[list[float]],
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    controller: CoherentController | SVDCoherentController,
    error_metrics: dict,
    experiment_dir: Path,
    save_plot: bool = True,
    is_svd: bool = False,
    scale_factor: float = 1.0,
) -> Path:
    """Save results to JSON file and plot in the experiment directory."""
    import matplotlib.pyplot as plt

    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save calibration plot (SVD has different visualization)
    if save_plot:
        visualizer = ControlVisualizer()
        if is_svd:
            fig = visualizer.plot_svd_summary(controller, "SVD Calibration Summary")
        else:
            fig = visualizer.plot_summary(controller, "Calibration Summary")
        plot_file = experiment_dir / "calibration_plot.png"
        fig.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

    filename = experiment_dir / "results.json"

    # Build calibration data based on controller type
    if is_svd:
        calibration_data = {
            "locked": controller.state.locked,
            "iterations": controller.state.iteration,
            "final_error": controller.state.error,
            "architecture": "svd",
            "phases_v": controller.state.phases_v.tolist(),
            "phases_u": controller.state.phases_u.tolist(),
            "sigma_dac": controller.state.sigma_dac,
            "v_locked": controller.state.v_locked,
            "u_locked": controller.state.u_locked,
        }
        # Add SVD target info
        U, sigma, Vh = controller.target_svd
        svd_info = {
            "singular_values": sigma.tolist(),
            "U_matrix": U.tolist(),
            "Vh_matrix": Vh.tolist(),
        }
    else:
        calibration_data = {
            "locked": controller.state.locked,
            "iterations": controller.state.iteration,
            "final_error": controller.state.error,
            "architecture": "unitary",
            "final_phases": controller.state.phases.tolist(),
        }
        svd_info = None

    # Build target matrix info including auto-scaling
    target_matrix_info = {
        "weights": weights,
        "matrix": controller.target_matrix.tolist(),
    }
    if scale_factor != 1.0:
        target_matrix_info["auto_scaled"] = True
        target_matrix_info["scale_factor"] = scale_factor
        target_matrix_info["normalized_weights"] = [w / scale_factor for w in weights]

    data = {
        "timestamp": datetime.now().isoformat(),
        "target_matrix": target_matrix_info,
        "calibration": calibration_data,
        "computations": [
            {
                "input": x.tolist(),
                "photonic_output": y_p.tolist(),
                "reference_output": y_r.tolist(),
                "error": (np.abs(y_p - y_r)).tolist(),
            }
            for x, y_p, y_r in results
        ],
        "error_metrics": error_metrics,
    }

    if svd_info:
        data["svd_decomposition"] = svd_info

    with open(filename, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyJSONEncoder)

    return filename


def main():
    parser = argparse.ArgumentParser(
        description="Run coherent photonic matrix multiply",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --weights 1 0 0 1 --input 1.0 0.5
  python run.py --weights 0.707 0.707 0.707 -0.707 --input 1.0 0.0 --input 0.5 0.5
  python run.py --weights 0.5 -0.3 0.8 0.2 --input 1.0 1.0 --verbose --save-results
        """,
    )

    parser.add_argument(
        "--weights",
        "-w",
        type=float,
        nargs=4,
        required=True,
        metavar=("W0", "W1", "W2", "W3"),
        help="Target matrix weights: M = [[w0, w1], [w2, w3]]. Range [-1,1] for unitary mode; any values for --svd mode (auto-scaled)",
    )

    parser.add_argument(
        "--input",
        "-i",
        type=float,
        nargs=2,
        action="append",
        metavar=("X0", "X1"),
        help="Input vector [x0, x1]. Can be specified multiple times.",
    )

    parser.add_argument(
        "--noise",
        "-n",
        type=float,
        default=2.0,
        help="Receiver noise standard deviation (default: 2.0)",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=300,
        help="Maximum calibration iterations (default: 300)",
    )

    parser.add_argument(
        "--error-threshold",
        type=float,
        default=2e-3,
        help="Calibration error threshold (default: 2e-3)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed output",
    )

    parser.add_argument(
        "--save-results",
        "-s",
        action="store_true",
        help="Save results to JSON file in sim/results/",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--svd",
        action="store_true",
        help="Use SVD decomposition for arbitrary matrices. Supports any weight values (auto-scales if needed)",
    )

    args = parser.parse_args()

    # Validate weights (SVD mode allows any weights)
    if not validate_weights(args.weights, svd_mode=args.svd):
        sys.exit(1)

    # Default input if none provided
    if args.input is None:
        args.input = [[1.0, 0.0]]

    print_banner()

    # Display target matrix
    w0, w1, w2, w3 = args.weights
    target = np.array([[w0, w1], [w2, w3]])
    print("Target Matrix:")
    print(format_matrix(target))
    print()

    # Create plant and controller based on mode
    seed = args.seed if args.seed is not None else 42

    if args.svd:
        # SVD mode for arbitrary matrices
        plant = SVDPhotonicPlant(
            drift_rate=0.0,
            noise_std=args.noise,
            tau_thermal=1e-9,
            crosstalk_coeff=0.0,
            seed=seed,
        )

        config = SVDControlConfig(
            initial_step=0.5,
            min_step=0.005,
            error_threshold=args.error_threshold,
            max_iterations=args.max_iterations,
            lock_count=3,
            num_averages=4,
            joint_refinement=True,
            refinement_iterations=100,
        )

        print("Using SVD architecture (M = U·Σ·V†)")
        print()

        # Show SVD decomposition
        U, sigma, Vh = np.linalg.svd(target)
        print("SVD Decomposition:")
        print(f"  Singular values: σ = [{sigma[0]:.4f}, {sigma[1]:.4f}]")

        # Auto-scaling: normalize if singular values > 1
        scale_factor = 1.0
        if np.max(sigma) > 1.0:
            scale_factor = np.max(sigma)
            normalized_weights = [w / scale_factor for w in args.weights]
            target = np.array([[normalized_weights[0], normalized_weights[1]],
                              [normalized_weights[2], normalized_weights[3]]])

            # Recompute SVD for normalized matrix
            U, sigma_norm, Vh = np.linalg.svd(target)

            print()
            print("  ╔═══════════════════════════════════════════════════════╗")
            print("  ║  AUTO-SCALING APPLIED                                 ║")
            print("  ╠═══════════════════════════════════════════════════════╣")
            print(f"  ║  Original σ_max = {scale_factor:.4f} > 1.0                        ║")
            print(f"  ║  Matrix normalized by factor: {scale_factor:.4f}                  ║")
            print(f"  ║  Photonic outputs will be scaled by {scale_factor:.4f}×           ║")
            print("  ╚═══════════════════════════════════════════════════════╝")
            print()
            print("Normalized Matrix (realized by hardware):")
            print(format_matrix(target))
            print()
            print(f"  Normalized σ = [{sigma_norm[0]:.4f}, {sigma_norm[1]:.4f}]")

        print()
    else:
        # Standard unitary-only mode
        plant = PhotonicPlant(
            drift_rate=0.0,
            noise_std=args.noise,
            tau_thermal=1e-9,
            crosstalk_coeff=0.0,
            seed=seed,
        )

        config = ControlConfig(
            initial_step=0.5,
            min_step=0.005,
            error_threshold=args.error_threshold,
            max_iterations=args.max_iterations,
            lock_count=3,
            num_averages=4,
        )

    # Calibrate
    print("Calibrating photonic system...")
    if args.verbose:
        print()

    # Use normalized weights for calibration if auto-scaling was applied
    if args.svd and scale_factor != 1.0:
        calibration_weights = [w / scale_factor for w in args.weights]
    else:
        calibration_weights = args.weights
        scale_factor = 1.0  # Ensure scale_factor is defined for non-SVD mode

    locked, controller = run_calibration(calibration_weights, plant, config, args.verbose)

    if locked:
        print(
            f"  \u2713 LOCKED in {controller.state.iteration} iterations "
            f"(error: {controller.state.error:.2e})"
        )
    else:
        print(
            f"  \u2717 FAILED to lock after {controller.state.iteration} iterations "
            f"(error: {controller.state.error:.2e})"
        )
        print("  Consider increasing --max-iterations or --error-threshold")

    print()

    # Compute outputs (with scaling for auto-normalized matrices)
    print("Computing y = M·x for input vectors:")
    if scale_factor != 1.0:
        print(f"  (Photonic outputs scaled by {scale_factor:.4f}× to match original matrix)")
    print()

    results = compute_outputs(controller, args.input, verbose=False, scale_factor=scale_factor)

    # Use original weights for display
    for x, y_photonic, y_reference in results:
        error = np.linalg.norm(y_photonic - y_reference)
        print(f"  Input x = {format_vector(x, 4)}")
        print()
        print(f"    Computation: y = M · x")
        print(f"      | y0 |   | {w0:7.4f}  {w1:7.4f} |   | {x[0]:.4f} |")
        print(f"      |    | = |                   | · |        |")
        print(f"      | y1 |   | {w2:7.4f}  {w3:7.4f} |   | {x[1]:.4f} |")
        print()
        print(f"    Expected (reference):  y = [{y_reference[0]:8.4f}, {y_reference[1]:8.4f}]")
        print(f"    Actual (photonic):     y = [{y_photonic[0]:8.4f}, {y_photonic[1]:8.4f}]")
        print(f"    Difference:           Δy = [{y_photonic[0] - y_reference[0]:+8.4f}, {y_photonic[1] - y_reference[1]:+8.4f}]")
        print(f"    Error (L2 norm):       {error:.2e}")
        print()
        print("─" * 60)

    print()

    # Error metrics
    metrics = compute_error_metrics(results)
    print("Summary:")
    print(f"  Max absolute error: {metrics['max_error']:.1e}")
    print(f"  RMS error: {metrics['rms_error']:.1e}")

    # Verbose output
    if args.verbose:
        print()
        print("Detailed Results:")
        print("-" * 40)
        compute_outputs(controller, args.input, verbose=True, scale_factor=scale_factor)

        print("Final Phase Settings:")
        if args.svd:
            print("  V† mesh phases:")
            phases_v = controller.state.phases_v
            print(f"    θ_v:    {phases_v[0]:.4f} rad ({np.degrees(phases_v[0]):.1f}°)")
            print(f"    φ_v0:   {phases_v[1]:.4f} rad ({np.degrees(phases_v[1]):.1f}°)")
            print(f"    φ_v1:   {phases_v[2]:.4f} rad ({np.degrees(phases_v[2]):.1f}°)")
            print(f"    φ_vout: {phases_v[3]:.4f} rad ({np.degrees(phases_v[3]):.1f}°)")
            print("  Σ (singular values via VOA):")
            sigma = controller.plant.voas.dac_to_sigma(controller.state.sigma_dac)
            print(f"    σ₀: {sigma[0]:.4f}")
            print(f"    σ₁: {sigma[1]:.4f}")
            print("  U mesh phases:")
            phases_u = controller.state.phases_u
            print(f"    θ_u:    {phases_u[0]:.4f} rad ({np.degrees(phases_u[0]):.1f}°)")
            print(f"    φ_u0:   {phases_u[1]:.4f} rad ({np.degrees(phases_u[1]):.1f}°)")
            print(f"    φ_u1:   {phases_u[2]:.4f} rad ({np.degrees(phases_u[2]):.1f}°)")
            print(f"    φ_uout: {phases_u[3]:.4f} rad ({np.degrees(phases_u[3]):.1f}°)")
        else:
            phases = controller.state.phases
            print(f"  θ (splitting):  {phases[0]:.4f} rad ({np.degrees(phases[0]):.1f}°)")
            print(f"  φ₀ (input 0):   {phases[1]:.4f} rad ({np.degrees(phases[1]):.1f}°)")
            print(f"  φ₁ (input 1):   {phases[2]:.4f} rad ({np.degrees(phases[2]):.1f}°)")
            print(f"  φ_out (global): {phases[3]:.4f} rad ({np.degrees(phases[3]):.1f}°)")

    # Save results
    if args.save_results:
        base_name = "svd_run" if args.svd else "run"
        experiment_dir = create_experiment_dir(base_name)
        filename = save_results(
            args.weights, args.input, results, controller, metrics, experiment_dir,
            is_svd=args.svd, scale_factor=scale_factor
        )
        print()
        print(f"Results saved to: {experiment_dir}")

    print()

    # Return exit code based on calibration success
    return 0 if locked else 1


if __name__ == "__main__":
    sys.exit(main())
