#!/usr/bin/env python3
"""
Interactive Demo: Coherent Photonic Matrix Multiply Control Loop

This script demonstrates the closed-loop control of a photonic 2x2 matrix
multiply block using coherent (I/Q) measurements for feedback.

Usage:
    python demo/run_demo.py [--interactive] [--save-plots] [--no-display]

The demo shows:
1. Calibration to various target matrices (identity, swap, Hadamard, custom)
2. Real-time visualization of phase trajectories and error convergence
3. Matrix element evolution in the I/Q plane
4. Evaluation of the locked matrix on test inputs
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from plant import PhotonicPlant
from demo.control_loop import CoherentController, ControlConfig, ControlState
from demo.visualizer import ControlVisualizer


def create_experiment_dir(base_name: str = "experiment") -> Path:
    """Create a timestamped experiment directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_base = Path(__file__).parent.parent / "sim" / "results"
    experiment_dir = results_base / f"{base_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def print_banner():
    """Print demo banner."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  Coherent Photonic 2×2 Matrix Multiply Control Demo    ║")
    print("╠" + "═" * 58 + "╣")
    print("║  • User specifies target matrix M (weights w₀..w₃)     ║")
    print("║  • Controller adjusts phases using I/Q feedback        ║")
    print("║  • Once locked, system computes y = M̂·x               ║")
    print("╚" + "═" * 58 + "╝")
    print()


def run_single_calibration(
    target_name: str,
    w0: float,
    w1: float,
    w2: float,
    w3: float,
    plant: PhotonicPlant,
    config: ControlConfig,
    visualizer: ControlVisualizer,
    save_plots: bool = False,
    show_plots: bool = True,
    experiment_dir: Path | None = None,
) -> tuple[bool, CoherentController]:
    """
    Run calibration for a single target matrix.

    Returns:
        Tuple of (locked, controller)
    """
    print(f"\n{'─' * 60}")
    print(f"Target: {target_name}")
    print(f"  M = [[{w0:7.4f}, {w1:7.4f}],")
    print(f"       [{w2:7.4f}, {w3:7.4f}]]")
    print(f"{'─' * 60}")

    # Create fresh controller
    controller = CoherentController(plant, config)
    controller.set_target(w0, w1, w2, w3)

    # Track progress
    iteration_marks = [10, 25, 50, 100, 150]
    mark_idx = 0

    def progress_callback(state: ControlState):
        nonlocal mark_idx
        if mark_idx < len(iteration_marks) and state.iteration >= iteration_marks[mark_idx]:
            print(
                f"  Iter {state.iteration:3d}: error = {state.error:.4e}, "
                f"step = {state.step_size:.4f}"
            )
            mark_idx += 1

    # Run calibration
    print("  Starting calibration...")
    locked = controller.run_calibration(callback=progress_callback)

    # Report results
    if locked:
        print(f"\n  ✓ LOCKED in {controller.state.iteration} iterations")
        print(f"    Final error: {controller.state.error:.4e}")

        # Show achieved matrix
        measured = controller.measure_matrix()
        print(f"\n  Achieved matrix (real part):")
        print(f"    [[{np.real(measured[0, 0]):7.4f}, {np.real(measured[0, 1]):7.4f}],")
        print(f"     [{np.real(measured[1, 0]):7.4f}, {np.real(measured[1, 1]):7.4f}]]")

        # Show imaginary residual
        imag_energy = np.sum(np.imag(measured) ** 2)
        print(f"\n  Imaginary residual (Q-energy): {imag_energy:.4e}")

        # Test evaluation
        print("\n  Evaluation tests:")
        test_inputs = [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5)]
        target = np.array([[w0, w1], [w2, w3]])

        for x0, x1 in test_inputs:
            y0, y1 = controller.evaluate(x0, x1)
            expected = target @ np.array([x0, x1])
            error = np.sqrt((y0 - expected[0]) ** 2 + (y1 - expected[1]) ** 2)
            print(
                f"    x=({x0:4.1f}, {x1:4.1f}) → "
                f"y=({y0:6.3f}, {y1:6.3f}) "
                f"[expected ({expected[0]:6.3f}, {expected[1]:6.3f})] "
                f"err={error:.4f}"
            )
    else:
        print(f"\n  ✗ Failed to lock after {config.max_iterations} iterations")
        print(f"    Final error: {controller.state.error:.4e}")

    # Generate plots
    if show_plots or save_plots:
        fig = visualizer.plot_summary(
            controller, f"{target_name} Matrix Calibration"
        )

        if save_plots and experiment_dir is not None:
            filename = experiment_dir / f"calibration_{target_name.lower().replace(' ', '_')}.png"
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            print(f"\n  Plot saved: {filename}")

            # Also save experiment data as JSON
            data = {
                "target_name": target_name,
                "target_weights": [w0, w1, w2, w3],
                "locked": locked,
                "iterations": controller.state.iteration,
                "final_error": controller.state.error,
                "config": {
                    "initial_step": config.initial_step,
                    "min_step": config.min_step,
                    "error_threshold": config.error_threshold,
                    "max_iterations": config.max_iterations,
                },
                "final_phases": controller.state.phases.tolist(),
            }
            json_file = experiment_dir / f"calibration_{target_name.lower().replace(' ', '_')}.json"
            with open(json_file, "w") as f:
                json.dump(data, f, indent=2)

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    return locked, controller


def run_animated_demo(
    w0: float,
    w1: float,
    w2: float,
    w3: float,
    plant: PhotonicPlant,
    config: ControlConfig,
):
    """
    Run calibration with live animation.
    """
    print("\n" + "=" * 60)
    print("ANIMATED CALIBRATION DEMO")
    print("=" * 60)

    controller = CoherentController(plant, config)
    controller.set_target(w0, w1, w2, w3)
    controller.reset()

    # Set up figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Live Calibration Progress", fontsize=14)

    # Initialize plot elements
    (error_line,) = axes[0, 0].semilogy([], [], "b-", linewidth=2)
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Error (log)")
    axes[0, 0].set_title("Convergence")
    axes[0, 0].set_xlim(0, config.max_iterations)
    axes[0, 0].set_ylim(1e-6, 10)
    axes[0, 0].axhline(y=config.error_threshold, color="r", linestyle="--")
    axes[0, 0].grid(True, alpha=0.3)

    phase_lines = []
    for i in range(4):
        (line,) = axes[0, 1].plot([], [], linewidth=2, label=f"φ{i}")
        phase_lines.append(line)
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Phase (rad)")
    axes[0, 1].set_title("Phase Trajectories")
    axes[0, 1].set_xlim(0, config.max_iterations)
    axes[0, 1].set_ylim(-1, 8)
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].grid(True, alpha=0.3)

    # Matrix display
    matrix_im = axes[1, 0].imshow(
        np.zeros((2, 2)), cmap="RdBu", vmin=-1, vmax=1, aspect="equal"
    )
    axes[1, 0].set_title("Measured Matrix (Real)")
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    matrix_texts = []
    for i in range(2):
        row = []
        for j in range(2):
            txt = axes[1, 0].text(j, i, "", ha="center", va="center", fontsize=12)
            row.append(txt)
        matrix_texts.append(row)
    plt.colorbar(matrix_im, ax=axes[1, 0])

    # Status text
    status_text = axes[1, 1].text(
        0.5,
        0.5,
        "",
        ha="center",
        va="center",
        fontsize=14,
        transform=axes[1, 1].transAxes,
    )
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Status")

    # Animation data
    error_data = []
    phase_data = [[] for _ in range(4)]
    target = np.array([[w0, w1], [w2, w3]])

    def init():
        error_line.set_data([], [])
        for line in phase_lines:
            line.set_data([], [])
        matrix_im.set_array(np.zeros((2, 2)))
        status_text.set_text("")
        return [error_line] + phase_lines + [matrix_im, status_text]

    def animate(frame):
        if controller.state.locked or controller.state.iteration >= config.max_iterations:
            return [error_line] + phase_lines + [matrix_im, status_text]

        # Step controller
        controller.step()

        # Update data
        error_data.append(controller.state.error)
        for i in range(4):
            phase_data[i].append(controller.state.phases[i])

        iterations = list(range(len(error_data)))

        # Update plots
        error_line.set_data(iterations, error_data)

        for i, line in enumerate(phase_lines):
            line.set_data(iterations, phase_data[i])

        # Update matrix display
        measured = controller.measure_matrix()
        measured_real = np.real(measured)
        matrix_im.set_array(measured_real)

        for i in range(2):
            for j in range(2):
                matrix_texts[i][j].set_text(
                    f"{measured_real[i, j]:.3f}\n(t:{target[i, j]:.3f})"
                )

        # Update status
        status = (
            f"Iteration: {controller.state.iteration}\n"
            f"Error: {controller.state.error:.4e}\n"
            f"Step size: {controller.state.step_size:.4f}\n"
            f"Lock counter: {controller.state.lock_counter}/{config.lock_count}\n\n"
        )
        if controller.state.locked:
            status += "✓ LOCKED!"
        elif controller.state.iteration >= config.max_iterations:
            status += "✗ Max iterations reached"
        else:
            status += "Calibrating..."

        status_text.set_text(status)

        return [error_line] + phase_lines + [matrix_im, status_text]

    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=config.max_iterations + 10,
        interval=50,  # 50ms per frame
        blit=False,
        repeat=False,
    )

    plt.tight_layout()
    plt.show()

    return controller


def interactive_menu():
    """Run interactive menu for choosing targets."""
    print_banner()

    # Create shared components
    plant = PhotonicPlant(
        drift_rate=0.0,
        noise_std=2.0,
        tau_thermal=1e-9,
        crosstalk_coeff=0.0,
        seed=42,
    )

    config = ControlConfig(
        initial_step=0.5,
        min_step=0.005,
        error_threshold=2e-3,
        max_iterations=300,
        lock_count=3,
        num_averages=4,
    )

    visualizer = ControlVisualizer()

    predefined_targets = {
        "1": ("Identity", 1.0, 0.0, 0.0, 1.0),
        "2": ("Swap", 0.0, 1.0, 1.0, 0.0),
        "3": ("Hadamard", 0.707, 0.707, 0.707, -0.707),
        "4": ("50-50 Splitter", 0.707, 0.707, -0.707, 0.707),
        "5": ("Attenuator (0.5)", 0.5, 0.0, 0.0, 0.5),
    }

    while True:
        print("\n" + "=" * 60)
        print("MAIN MENU")
        print("=" * 60)
        print("\nPredefined targets:")
        for key, (name, w0, w1, w2, w3) in predefined_targets.items():
            print(f"  [{key}] {name}: [[{w0:.3f}, {w1:.3f}], [{w2:.3f}, {w3:.3f}]]")
        print("\nOptions:")
        print("  [c] Custom matrix")
        print("  [a] Animated demo (Identity)")
        print("  [r] Run all predefined targets")
        print("  [q] Quit")
        print()

        choice = input("Enter choice: ").strip().lower()

        if choice == "q":
            print("\nGoodbye!")
            break

        elif choice == "a":
            plant.reset()
            run_animated_demo(1.0, 0.0, 0.0, 1.0, plant, config)

        elif choice == "r":
            print("\nRunning all predefined targets...")
            results = []
            for key, (name, w0, w1, w2, w3) in predefined_targets.items():
                plant.reset()
                locked, _ = run_single_calibration(
                    name, w0, w1, w2, w3, plant, config, visualizer,
                    save_plots=True, show_plots=False
                )
                results.append((name, locked))

            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            for name, locked in results:
                status = "✓ LOCKED" if locked else "✗ FAILED"
                print(f"  {name:20s}: {status}")

        elif choice == "c":
            print("\nEnter custom matrix weights (values in [-1, 1]):")
            try:
                w0 = float(input("  w0 (M[0,0]): "))
                w1 = float(input("  w1 (M[0,1]): "))
                w2 = float(input("  w2 (M[1,0]): "))
                w3 = float(input("  w3 (M[1,1]): "))

                if any(abs(w) > 1 for w in [w0, w1, w2, w3]):
                    print("  Error: Weights must be in [-1, 1]")
                    continue

                plant.reset()
                run_single_calibration(
                    "Custom", w0, w1, w2, w3, plant, config, visualizer,
                    save_plots=False, show_plots=True
                )
            except ValueError:
                print("  Invalid input. Please enter numeric values.")

        elif choice in predefined_targets:
            name, w0, w1, w2, w3 = predefined_targets[choice]
            plant.reset()
            run_single_calibration(
                name, w0, w1, w2, w3, plant, config, visualizer,
                save_plots=False, show_plots=True
            )

        else:
            print("  Invalid choice. Please try again.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Coherent Photonic Matrix Multiply Control Demo"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive menu",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to files",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display plots (for headless runs)",
    )
    parser.add_argument(
        "--animated",
        action="store_true",
        help="Run animated demo",
    )
    args = parser.parse_args()

    if args.no_display:
        plt.switch_backend("Agg")

    if args.interactive:
        interactive_menu()
    elif args.animated:
        print_banner()
        plant = PhotonicPlant(
            drift_rate=0.0, noise_std=2.0, tau_thermal=1e-9, seed=42
        )
        config = ControlConfig(
            initial_step=0.5, min_step=0.005, error_threshold=2e-3, max_iterations=300
        )
        run_animated_demo(1.0, 0.0, 0.0, 1.0, plant, config)
    else:
        # Default: run all demos
        print_banner()

        plant = PhotonicPlant(
            drift_rate=0.0, noise_std=2.0, tau_thermal=1e-9, seed=42
        )
        config = ControlConfig(
            initial_step=0.5, min_step=0.005, error_threshold=2e-3, max_iterations=300
        )
        visualizer = ControlVisualizer()

        # Create experiment directory if saving plots
        experiment_dir = None
        if args.save_plots:
            experiment_dir = create_experiment_dir("calibration_demo")
            print(f"Experiment directory: {experiment_dir}\n")

        targets = [
            ("Identity", 1.0, 0.0, 0.0, 1.0),
            ("Swap", 0.0, 1.0, 1.0, 0.0),
            ("Hadamard", 0.707, 0.707, 0.707, -0.707),
        ]

        results = []
        for name, w0, w1, w2, w3 in targets:
            plant.reset()
            locked, controller = run_single_calibration(
                name, w0, w1, w2, w3, plant, config, visualizer,
                save_plots=args.save_plots, show_plots=not args.no_display,
                experiment_dir=experiment_dir
            )
            results.append((name, locked, controller))

        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        for name, locked, controller in results:
            status = "✓ LOCKED" if locked else "✗ FAILED"
            iters = controller.state.iteration
            error = controller.state.error
            print(f"  {name:15s}: {status} | {iters:3d} iters | error: {error:.2e}")
        print("=" * 60)

        # Save summary to experiment directory
        if experiment_dir:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "results": [
                    {
                        "name": name,
                        "locked": locked,
                        "iterations": controller.state.iteration,
                        "final_error": controller.state.error,
                    }
                    for name, locked, controller in results
                ],
            }
            with open(experiment_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nResults saved to: {experiment_dir}")


if __name__ == "__main__":
    main()
