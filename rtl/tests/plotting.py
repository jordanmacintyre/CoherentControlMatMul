"""
Plotting utilities for co-simulation calibration results.

Generates convergence graphs and saves them to the artifacts directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Union, Optional

import math
import numpy as np

# Try to import matplotlib, but allow graceful degradation
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def normalize_error(raw_error: int, num_elements: int = 4) -> float:
    """
    Convert raw squared-error to normalized RMS error.

    The calibration FSM computes error as sum of squared differences
    in Q1.15 fixed-point: error = Σ(measured - target)²

    This function converts to a normalized RMS value in [0, 1] range
    for human-readable display.

    Args:
        raw_error: Raw sum-of-squared-error from calibration FSM
        num_elements: Number of matrix elements (default 4 for 2x2)

    Returns:
        Normalized RMS error as fraction of full scale.
        Example: 0.01 = 1% RMS error

    Example:
        >>> normalize_error(100000)  # CAL_LOCK_THRESHOLD default
        0.004832...  # ~0.5% RMS error
    """
    if raw_error <= 0:
        return 0.0

    # MSE = raw_error / num_elements
    mse = raw_error / num_elements

    # RMSE = sqrt(MSE)
    rmse = math.sqrt(mse)

    # Normalize by Q1.15 full scale (32767)
    Q1_15_FULL_SCALE = 32767
    return rmse / Q1_15_FULL_SCALE


def error_to_percent(raw_error: int, num_elements: int = 4) -> float:
    """
    Convert raw squared-error to percentage for display.

    Args:
        raw_error: Raw sum-of-squared-error from calibration FSM
        num_elements: Number of matrix elements (default 4 for 2x2)

    Returns:
        RMS error as percentage (e.g., 0.5 for 0.5%)
    """
    return normalize_error(raw_error, num_elements) * 100


def create_run_directory(artifacts_dir: Path | str = None) -> Path:
    """
    Create a unique run directory within the artifacts folder.

    Args:
        artifacts_dir: Base artifacts directory. Defaults to project_root/artifacts.

    Returns:
        Path to the created run directory (e.g., artifacts/run_20240215_143022)
    """
    if artifacts_dir is None:
        # Default to project root / artifacts
        project_root = Path(__file__).parent.parent.parent
        artifacts_dir = project_root / "artifacts"

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Create unique run folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = artifacts_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def save_calibration_data(
    run_dir: Path,
    test_name: str,
    data: dict,
) -> Path:
    """
    Save calibration data to JSON file.

    Args:
        run_dir: Run directory path
        test_name: Name of the test
        data: Dictionary containing calibration data

    Returns:
        Path to saved JSON file
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / f"{test_name}_data.json"

    # Convert numpy arrays to lists for JSON serialization
    serializable_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            serializable_data[key] = value.tolist()
        elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], np.ndarray):
            serializable_data[key] = [v.tolist() for v in value]
        else:
            serializable_data[key] = value

    with open(json_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)

    return json_path


def plot_convergence(
    cycles: list[int],
    errors: list[float],
    run_dir: Path,
    test_name: str,
    title: str = None,
    threshold: float = None,
    show_normalized: bool = True,
) -> tuple[Path | None, int | None]:
    """
    Plot calibration error convergence over time.

    Args:
        cycles: List of cycle numbers
        errors: List of error values (raw squared-error)
        run_dir: Directory to save plot
        test_name: Test name for filename
        title: Plot title (defaults to test_name)
        threshold: Optional convergence threshold line
        show_normalized: Show normalized RMS % on secondary axis

    Returns:
        Tuple of (path to saved plot, cycle at minimum error) or (None, None) if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        return None, None

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Convert to normalized RMS error (%) for primary display
    normalized_errors = [error_to_percent(int(e)) for e in errors]

    # Find minimum error point (this is the "final calibration" point)
    min_idx = int(np.argmin(normalized_errors))
    min_error = normalized_errors[min_idx]
    min_cycle = cycles[min_idx]

    # Primary axis: normalized RMS error (%) - LINEAR scale for readability
    color1 = 'tab:blue'
    ax1.plot(cycles, normalized_errors, color=color1, linewidth=1.5, label='RMS Error %')
    ax1.set_xlabel('Cycle', fontsize=12)
    ax1.set_ylabel('RMS Error (%)', color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Set Y-axis limits based on data (cap at 120% for readability)
    max_error = max(normalized_errors) if normalized_errors else 100
    ax1.set_ylim(0, min(max_error * 1.1, 120))

    # Mark the minimum error point (final calibration)
    ax1.scatter([min_cycle], [min_error], color='green', s=150, zorder=5,
                marker='*', edgecolors='darkgreen', linewidths=1,
                label=f'Final Cal: {min_error:.2f}% @ cycle {min_cycle}')

    # Add annotation arrow pointing to minimum
    # Position text above and to the right, adjusting based on plot range
    text_x = min(min_cycle + cycles[-1] * 0.05, cycles[-1] * 0.85)
    text_y = min(min_error + max_error * 0.15, max_error * 0.5)
    ax1.annotate(f'{min_error:.2f}%',
                 xy=(min_cycle, min_error),
                 xytext=(text_x, text_y),
                 arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                 fontsize=11, fontweight='bold', color='darkgreen')

    # Add threshold line (convert to %)
    if threshold is not None:
        threshold_pct = error_to_percent(int(threshold))
        ax1.axhline(y=threshold_pct, color='r', linestyle='--', linewidth=1.5,
                    label=f'Threshold ({threshold_pct:.1f}%)')

    ax1.grid(True, alpha=0.3)

    # Add reference lines for common accuracy targets
    if max(normalized_errors) > 1.0:
        ax1.axhline(y=1.0, color='green', linestyle='-.', linewidth=0.8, alpha=0.5)
        ax1.text(cycles[-1] * 0.02, 1.1, '1% target', color='green', fontsize=9, alpha=0.7)
    if max(normalized_errors) > 5.0:
        ax1.axhline(y=5.0, color='orange', linestyle='-.', linewidth=0.8, alpha=0.5)
        ax1.text(cycles[-1] * 0.02, 5.5, '5% target', color='orange', fontsize=9, alpha=0.7)

    ax1.legend(loc='upper right')
    ax1.set_title(title or f'{test_name} - Error Convergence', fontsize=14)

    plt.tight_layout()

    plot_path = run_dir / f"{test_name}_convergence.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    return plot_path, min_cycle


def plot_phase_evolution(
    cycles: list[int],
    phases: list[list[float]],
    run_dir: Path,
    test_name: str,
    phase_labels: list[str] = None,
    min_cycle: int = None,
) -> Path | None:
    """
    Plot phase evolution over calibration.

    Args:
        cycles: List of cycle numbers
        phases: List of phase arrays [cycle][phase_idx]
        run_dir: Directory to save plot
        test_name: Test name for filename
        phase_labels: Labels for each phase
        min_cycle: Cycle at minimum error (final calibration point) to mark on plot

    Returns:
        Path to saved plot, or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    phases_array = np.array(phases)
    num_phases = phases_array.shape[1] if len(phases_array.shape) > 1 else 1

    if phase_labels is None:
        phase_labels = [f'Phase {i}' for i in range(num_phases)]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, num_phases))

    for i in range(num_phases):
        ax.plot(cycles, phases_array[:, i], color=colors[i], linewidth=1.5, label=phase_labels[i])

    # Mark the final calibration point (minimum error cycle)
    if min_cycle is not None:
        # Add vertical line at minimum error cycle
        ax.axvline(x=min_cycle, color='green', linestyle='--', linewidth=2,
                   label=f'Final Cal @ {min_cycle}')

        # Find the index closest to min_cycle and mark phase values
        if min_cycle in cycles:
            idx = cycles.index(min_cycle)
        else:
            # Find closest cycle
            idx = min(range(len(cycles)), key=lambda i: abs(cycles[i] - min_cycle))

        # Add markers at the final calibration point for each phase
        for i in range(num_phases):
            phase_val = phases_array[idx, i]
            ax.scatter([cycles[idx]], [phase_val], color=colors[i], s=100, zorder=5,
                       marker='o', edgecolors='black', linewidths=1)

        # Add text box with final phase values
        final_phases_text = "Final Cal Values:\n"
        for i in range(num_phases):
            final_phases_text += f"  {phase_labels[i]}: {int(phases_array[idx, i])}\n"
        ax.text(0.02, 0.98, final_phases_text.strip(), transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Cycle', fontsize=12)
    ax.set_ylabel('Phase (DAC code)', fontsize=12)
    ax.set_title(f'{test_name} - Phase Evolution', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()

    plot_path = run_dir / f"{test_name}_phases.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    return plot_path


def plot_summary(
    run_dir: Path,
    test_results: dict,
) -> Path | None:
    """
    Create summary plot showing all test results.

    Args:
        run_dir: Directory to save plot
        test_results: Dict of {test_name: {'passed': bool, 'cycles': int, 'final_error': float}}

    Returns:
        Path to saved plot, or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    run_dir = Path(run_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    test_names = list(test_results.keys())
    passed = [1 if r.get('passed', False) else 0 for r in test_results.values()]
    cycles = [r.get('cycles', 0) for r in test_results.values()]

    # Pass/fail bar chart
    colors = ['green' if p else 'red' for p in passed]
    ax1.barh(test_names, passed, color=colors)
    ax1.set_xlabel('Passed')
    ax1.set_title('Test Results')
    ax1.set_xlim(0, 1.2)

    # Cycles to convergence
    ax2.barh(test_names, cycles, color='steelblue')
    ax2.set_xlabel('Cycles to Lock')
    ax2.set_title('Convergence Speed')

    plt.tight_layout()

    plot_path = run_dir / "summary.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    return plot_path


def generate_report(
    run_dir: Path,
    test_results: dict,
    weights: tuple[float, float, float, float] = None,
    mode: str = "unitary",
) -> Path:
    """
    Generate a text report summarizing the run.

    Args:
        run_dir: Directory to save report
        test_results: Dict of test results
        weights: Target matrix weights
        mode: "unitary" or "svd"

    Returns:
        Path to report file
    """
    run_dir = Path(run_dir)

    report_path = run_dir / "report.txt"

    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Coherent Photonic Matrix Multiply - Calibration Report\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Run Directory: {run_dir.name}\n")
        f.write(f"Mode: {mode}\n")

        if weights:
            f.write(f"Target Matrix:\n")
            f.write(f"  [[{weights[0]:+.4f}, {weights[1]:+.4f}],\n")
            f.write(f"   [{weights[2]:+.4f}, {weights[3]:+.4f}]]\n\n")

        f.write("-" * 60 + "\n")
        f.write("Test Results\n")
        f.write("-" * 60 + "\n\n")

        passed_count = 0
        total_count = len(test_results)

        for test_name, result in test_results.items():
            status = "PASS" if result.get('passed', False) else "FAIL"
            if result.get('passed', False):
                passed_count += 1

            f.write(f"  {test_name}:\n")
            f.write(f"    Status: {status}\n")

            if 'cycles' in result:
                f.write(f"    Cycles: {result['cycles']}\n")
            if 'final_error' in result:
                f.write(f"    Final Error: {result['final_error']:.2e}\n")
            f.write("\n")

        f.write("-" * 60 + "\n")
        f.write(f"Summary: {passed_count}/{total_count} tests passed\n")
        f.write("=" * 60 + "\n")

    return report_path
