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

from plant import PhotonicPlant
from demo.control_loop import CoherentController, ControlConfig
from demo.visualizer import ControlVisualizer


def print_banner():
    """Print script banner."""
    print()
    print("\u2554" + "\u2550" * 58 + "\u2557")
    print("\u2551     Coherent Photonic Matrix Multiply - Run Script       \u2551")
    print("\u255a" + "\u2550" * 58 + "\u255d")
    print()


def validate_weights(weights: list[float]) -> bool:
    """Validate that all weights are in [-1, 1]."""
    for i, w in enumerate(weights):
        if not -1.0 <= w <= 1.0:
            print(f"Error: Weight w{i} = {w} is outside valid range [-1, 1]")
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
    plant: PhotonicPlant,
    config: ControlConfig,
    verbose: bool = False,
) -> tuple[bool, CoherentController]:
    """
    Calibrate the photonic system to realize the target matrix.

    Args:
        weights: Target matrix weights [w0, w1, w2, w3]
        plant: PhotonicPlant instance
        config: Control configuration
        verbose: Print detailed progress

    Returns:
        Tuple of (locked, controller)
    """
    controller = CoherentController(plant, config)
    controller.set_target(*weights)

    def progress_callback(state):
        if verbose and state.iteration % 10 == 0:
            print(f"    Iteration {state.iteration:3d}: error = {state.error:.4e}")

    locked = controller.run_calibration(callback=progress_callback if verbose else None)

    return locked, controller


def compute_outputs(
    controller: CoherentController,
    inputs: list[list[float]],
    verbose: bool = False,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute outputs for given input vectors.

    Args:
        controller: Calibrated controller
        inputs: List of input vectors [x0, x1]
        verbose: Print detailed output

    Returns:
        List of (input, photonic_output, reference_output) tuples
    """
    target = controller.target_matrix
    results = []

    for x in inputs:
        x_arr = np.array(x)

        # Photonic computation
        y_photonic = np.array(controller.evaluate(x[0], x[1]))

        # Reference (floating-point) computation
        y_reference = target @ x_arr

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
    controller: CoherentController,
    error_metrics: dict,
    experiment_dir: Path,
    save_plot: bool = True,
) -> Path:
    """Save results to JSON file and plot in the experiment directory."""
    import matplotlib.pyplot as plt

    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save calibration plot
    if save_plot:
        visualizer = ControlVisualizer()
        fig = visualizer.plot_summary(controller, "Calibration Summary")
        plot_file = experiment_dir / "calibration_plot.png"
        fig.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

    filename = experiment_dir / "results.json"

    data = {
        "timestamp": datetime.now().isoformat(),
        "target_matrix": {
            "weights": weights,
            "matrix": controller.target_matrix.tolist(),
        },
        "calibration": {
            "locked": controller.state.locked,
            "iterations": controller.state.iteration,
            "final_error": controller.state.error,
            "final_phases": controller.state.phases.tolist(),
        },
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

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

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
        help="Target matrix weights: M = [[w0, w1], [w2, w3]], each in [-1, 1]",
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

    args = parser.parse_args()

    # Validate weights
    if not validate_weights(args.weights):
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

    # Create plant
    seed = args.seed if args.seed is not None else 42
    plant = PhotonicPlant(
        drift_rate=0.0,
        noise_std=args.noise,
        tau_thermal=1e-9,
        crosstalk_coeff=0.0,
        seed=seed,
    )

    # Create config
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
    locked, controller = run_calibration(args.weights, plant, config, args.verbose)

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

    # Compute outputs
    print("Computing y = M\u00b7x for input vectors:")
    print("\u2500" * 60)
    print(f"  {'Input x':<16} {'Photonic y':<16} {'Reference y':<16} {'Error':<10}")
    print("\u2500" * 60)

    results = compute_outputs(controller, args.input, verbose=False)

    for x, y_photonic, y_reference in results:
        error = np.linalg.norm(y_photonic - y_reference)
        print(
            f"  {format_vector(x, 2):<16} "
            f"{format_vector(y_photonic, 2):<16} "
            f"{format_vector(y_reference, 2):<16} "
            f"{error:.1e}"
        )

    print("\u2500" * 60)
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
        compute_outputs(controller, args.input, verbose=True)

        print("Final Phase Settings:")
        phases = controller.state.phases
        print(f"  \u03b8 (splitting):  {phases[0]:.4f} rad ({np.degrees(phases[0]):.1f}\u00b0)")
        print(f"  \u03c6\u2080 (input 0):   {phases[1]:.4f} rad ({np.degrees(phases[1]):.1f}\u00b0)")
        print(f"  \u03c6\u2081 (input 1):   {phases[2]:.4f} rad ({np.degrees(phases[2]):.1f}\u00b0)")
        print(f"  \u03c6_out (global): {phases[3]:.4f} rad ({np.degrees(phases[3]):.1f}\u00b0)")

    # Save results
    if args.save_results:
        experiment_dir = create_experiment_dir("run")
        filename = save_results(
            args.weights, args.input, results, controller, metrics, experiment_dir
        )
        print()
        print(f"Results saved to: {experiment_dir}")

    print()

    # Return exit code based on calibration success
    return 0 if locked else 1


if __name__ == "__main__":
    sys.exit(main())
