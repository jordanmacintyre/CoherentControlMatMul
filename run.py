#!/usr/bin/env python3
"""
Main run script for coherent photonic matrix multiply.

This script runs the RTL controller with Python plant models via cocotb
mixed-signal co-simulation.

Workflow:
1. User specifies target matrix weights (w0, w1, w2, w3)
2. RTL calibration FSM finds phase settings via closed-loop control
3. Python plant model provides realistic photonic simulation
4. Results are compared against floating-point reference

Artifacts:
- Each run creates a unique folder in artifacts/ with:
  - Calibration data (JSON)
  - Convergence plots (PNG)
  - Waveforms (VCD, if DUMP_WAVES=1)
  - Summary report

Usage:
    python run.py --weights 0.707 0.707 0.707 -0.707 --input 1.0 0.5
    python run.py --svd --weights 0.5 0 0 0.3 --input 1.0 1.0
    python run.py --weights 1 0 0 1 --input 1.0 0.0 --verbose
"""

import argparse
import subprocess
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np


def print_banner():
    """Print script banner."""
    print()
    print("\u2554" + "\u2550" * 62 + "\u2557")
    print("\u2551  Coherent Photonic Matrix Multiply - RTL Co-Simulation       \u2551")
    print("\u255a" + "\u2550" * 62 + "\u255d")
    print()


def validate_weights(weights: list[float], svd_mode: bool = False) -> bool:
    """
    Validate weights for the given mode.

    In standard (unitary) mode, weights should form a near-unitary matrix.
    In SVD mode, any weights are allowed (auto-scaling handles large values).
    """
    if svd_mode:
        return True

    for i, w in enumerate(weights):
        if not -1.0 <= w <= 1.0:
            print(f"Error: Weight w{i} = {w} is outside valid range [-1, 1]")
            print("  Use --svd mode for matrices with weights outside [-1, 1]")
            return False
    return True


def format_matrix(m: np.ndarray, precision: int = 4, indent: int = 2) -> str:
    """Format a matrix for display."""
    prefix = " " * indent
    lines = [
        f"{prefix}[[{m[0, 0]:.{precision}f}, {m[0, 1]:.{precision}f}],",
        f"{prefix} [{m[1, 0]:.{precision}f}, {m[1, 1]:.{precision}f}]]",
    ]
    return "\n".join(lines)


def check_cocotb_installed() -> bool:
    """Check if cocotb and a simulator are available."""
    try:
        import cocotb
        return True
    except ImportError:
        return False


def check_simulator_installed() -> str | None:
    """Check which RTL simulator is available."""
    # Check for Icarus Verilog
    try:
        result = subprocess.run(
            ["iverilog", "-V"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return "icarus"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Check for Verilator
    try:
        result = subprocess.run(
            ["verilator", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return "verilator"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def create_run_artifacts_dir() -> Path:
    """Create a unique run directory in artifacts/."""
    project_root = Path(__file__).parent
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = artifacts_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def copy_waveforms_to_artifacts(rtl_dir: Path, run_dir: Path) -> None:
    """Copy VCD waveforms to artifacts directory if they exist."""
    # Check for common waveform output locations
    possible_vcd_files = [
        rtl_dir / "sim_build" / "dump.vcd",
        rtl_dir / "dump.vcd",
        rtl_dir / "coherent_matmul_top.vcd",
    ]

    for vcd_file in possible_vcd_files:
        if vcd_file.exists():
            dest = run_dir / vcd_file.name
            shutil.copy2(vcd_file, dest)
            print(f"  Waveforms saved to: {dest.relative_to(Path(__file__).parent)}")
            break


def generate_run_report(
    run_dir: Path,
    weights: list[float],
    mode: str,
    success: bool,
    simulator: str,
) -> Path:
    """Generate a summary report for the run."""
    report_path = run_dir / "report.txt"

    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Coherent Photonic Matrix Multiply - Run Report\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Run Directory: {run_dir.name}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Simulator: {simulator}\n\n")

        f.write("Target Matrix:\n")
        f.write(f"  [[{weights[0]:+.4f}, {weights[1]:+.4f}],\n")
        f.write(f"   [{weights[2]:+.4f}, {weights[3]:+.4f}]]\n\n")

        if mode.lower() == "svd":
            M = np.array([[weights[0], weights[1]], [weights[2], weights[3]]])
            _, sigma, _ = np.linalg.svd(M)
            f.write(f"Singular Values: [{sigma[0]:.4f}, {sigma[1]:.4f}]\n\n")

        f.write("-" * 60 + "\n")
        f.write(f"Result: {'PASS' if success else 'FAIL'}\n")
        f.write("=" * 60 + "\n")

    return report_path


def run_cosim(
    weights: list[float],
    inputs: list[list[float]],
    svd_mode: bool = False,
    noise: float = 2.0,
    seed: int = 42,
    verbose: bool = False,
    timeout: int = 60,
    run_dir: Path = None,
    dump_waves: bool = False,
) -> dict:
    """
    Run RTL co-simulation with given parameters.

    Args:
        weights: Target matrix weights [w0, w1, w2, w3]
        inputs: List of input vectors [[x0, x1], ...]
        svd_mode: Use SVD architecture
        noise: Receiver noise standard deviation
        seed: Random seed for reproducibility
        verbose: Print detailed output
        timeout: Simulation timeout in seconds
        run_dir: Directory for artifacts (plots, data, waveforms)
        dump_waves: Enable VCD waveform dumping

    Returns:
        Dictionary with results
    """
    rtl_dir = Path(__file__).parent / "rtl"

    # Set environment variables for cocotb test
    env = os.environ.copy()
    env["COSIM_WEIGHTS"] = ",".join(str(w) for w in weights)
    env["COSIM_INPUTS"] = ";".join(",".join(str(x) for x in inp) for inp in inputs)
    env["COSIM_SVD_MODE"] = "1" if svd_mode else "0"
    env["COSIM_NOISE"] = str(noise)
    env["COSIM_SEED"] = str(seed)
    env["COSIM_VERBOSE"] = "1" if verbose else "0"

    # Set artifacts directory for plotting
    if run_dir:
        env["ARTIFACTS_DIR"] = str(run_dir)

    # Enable waveform dumping
    if dump_waves:
        env["DUMP_WAVES"] = "1"

    # Build and run simulation
    test_case = "test_svd_arbitrary" if svd_mode else "test_unitary_hadamard"

    cmd = ["make", f"TESTCASE={test_case}"]

    if verbose:
        print(f"Running: {' '.join(cmd)}")
        print(f"  Working directory: {rtl_dir}")
        print()

    try:
        result = subprocess.run(
            cmd,
            cwd=rtl_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        if verbose:
            if result.stdout:
                print("--- Simulation Output ---")
                print(result.stdout)
            if result.stderr:
                print("--- Simulation Errors ---")
                print(result.stderr)

        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "run_dir": run_dir,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "error": f"Simulation timed out after {timeout} seconds",
            "run_dir": run_dir,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run coherent photonic matrix multiply (RTL co-simulation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --weights 1 0 0 1 --input 1.0 0.5
  python run.py --weights 0.707 0.707 0.707 -0.707 --input 1.0 0.0
  python run.py --svd --weights 0.5 0 0 0.3 --input 1.0 1.0 --verbose

Requirements:
  - cocotb: pip install cocotb
  - Icarus Verilog: brew install icarus-verilog (macOS) or apt install iverilog (Linux)
        """,
    )

    parser.add_argument(
        "--weights", "-w",
        type=float,
        nargs=4,
        required=True,
        metavar=("W0", "W1", "W2", "W3"),
        help="Target matrix weights: M = [[w0, w1], [w2, w3]]",
    )

    parser.add_argument(
        "--input", "-i",
        type=float,
        nargs=2,
        action="append",
        metavar=("X0", "X1"),
        help="Input vector [x0, x1]. Can be specified multiple times.",
    )

    parser.add_argument(
        "--svd",
        action="store_true",
        help="Use SVD architecture for arbitrary matrices",
    )

    parser.add_argument(
        "--noise", "-n",
        type=float,
        default=2.0,
        help="Receiver noise standard deviation (default: 2.0)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Simulation timeout in seconds (default: 120)",
    )

    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies and exit",
    )

    parser.add_argument(
        "--dump-waves",
        action="store_true",
        help="Dump VCD waveforms (saved to artifacts folder)",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation",
    )

    args = parser.parse_args()

    # Check dependencies
    cocotb_ok = check_cocotb_installed()
    simulator = check_simulator_installed()

    if args.check_deps:
        print("Dependency Check:")
        print(f"  cocotb: {'OK' if cocotb_ok else 'MISSING (pip install cocotb)'}")
        print(f"  Simulator: {simulator if simulator else 'MISSING'}")
        if not simulator:
            print("    Install Icarus Verilog: brew install icarus-verilog (macOS)")
            print("                         or apt install iverilog (Linux)")
        return 0 if (cocotb_ok and simulator) else 1

    if not cocotb_ok:
        print("Error: cocotb not installed")
        print("  Install with: pip install cocotb")
        return 1

    if not simulator:
        print("Error: No RTL simulator found")
        print("  Install Icarus Verilog: brew install icarus-verilog (macOS)")
        print("                       or apt install iverilog (Linux)")
        return 1

    # Validate weights
    if not validate_weights(args.weights, svd_mode=args.svd):
        return 1

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

    mode_str = "SVD" if args.svd else "Unitary"
    print(f"Mode: {mode_str}")
    print(f"Simulator: {simulator}")
    print()

    if args.svd:
        # Show SVD decomposition
        U, sigma, Vh = np.linalg.svd(target)
        print("SVD Decomposition:")
        print(f"  Singular values: σ = [{sigma[0]:.4f}, {sigma[1]:.4f}]")

        if np.max(sigma) > 1.0:
            scale_factor = np.max(sigma)
            print()
            print("  ╔═══════════════════════════════════════════════════════╗")
            print("  ║  AUTO-SCALING REQUIRED                                ║")
            print("  ╠═══════════════════════════════════════════════════════╣")
            print(f"  ║  σ_max = {scale_factor:.4f} > 1.0                              ║")
            print(f"  ║  Matrix will be normalized by {scale_factor:.4f}               ║")
            print("  ╚═══════════════════════════════════════════════════════╝")
        print()

    # Create artifacts directory for this run
    run_dir = create_run_artifacts_dir()
    print(f"Artifacts directory: {run_dir.relative_to(Path(__file__).parent)}")
    print()

    print("Running RTL co-simulation...")
    print()

    # Run simulation
    result = run_cosim(
        weights=args.weights,
        inputs=args.input,
        svd_mode=args.svd,
        noise=args.noise,
        seed=args.seed,
        verbose=args.verbose,
        timeout=args.timeout,
        run_dir=run_dir if not args.no_plots else None,
        dump_waves=args.dump_waves,
    )

    # Generate report and collect artifacts
    mode_str = "SVD" if args.svd else "Unitary"
    generate_run_report(run_dir, args.weights, mode_str, result["success"], simulator)

    # Copy waveforms if they were generated
    if args.dump_waves:
        copy_waveforms_to_artifacts(Path(__file__).parent / "rtl", run_dir)

    if result["success"]:
        print("✓ Simulation completed successfully")
        print()

        # List generated artifacts
        print("Generated artifacts:")
        for item in sorted(run_dir.iterdir()):
            print(f"  {item.relative_to(Path(__file__).parent)}")
        print()

        print("To run specific tests:")
        print("  cd rtl && make test-all           # Run all tests")
        print("  cd rtl && make test-unitary       # Unitary mode tests")
        print("  cd rtl && make test-svd           # SVD mode tests")
        print()
        print("To view waveforms:")
        print("  cd rtl && DUMP_WAVES=1 make test-all && make waves")
        return 0
    else:
        print("✗ Simulation failed")
        if "error" in result:
            print(f"  {result['error']}")
        if result.get("stderr"):
            print()
            print("Error output:")
            print(result["stderr"][:1000])  # Truncate if too long

        print()
        print(f"Check artifacts in: {run_dir.relative_to(Path(__file__).parent)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
