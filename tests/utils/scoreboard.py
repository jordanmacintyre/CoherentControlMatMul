"""
Metrics scoreboard for test results collection.

Collects and reports test metrics for regression testing.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TestResult:
    """Single test result."""

    test_name: str
    passed: bool
    lock_cycles: int | None = None
    matrix_error: float | None = None
    q_residual: float | None = None
    output_error: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


class MetricsScoreboard:
    """
    Collect and report test metrics.

    Tracks metrics across multiple test runs for regression analysis.
    """

    def __init__(self):
        """Initialize metrics scoreboard."""
        self.results: list[TestResult] = []
        self.metrics: dict[str, list[float]] = {
            "lock_time_cycles": [],
            "final_matrix_error": [],
            "q_residual": [],
            "output_error": [],
            "phase_trajectory": [],
        }

    def record_calibration(
        self,
        test_name: str,
        locked: bool,
        cycles: int,
        matrix_error: float | None = None,
        q_residual: float | None = None,
    ):
        """
        Record calibration test result.

        Args:
            test_name: Name of the test
            locked: Whether calibration achieved lock
            cycles: Number of cycles to lock (or timeout)
            matrix_error: Final matrix error (Frobenius norm)
            q_residual: Final Q-residual (imaginary energy)
        """
        result = TestResult(
            test_name=test_name,
            passed=locked,
            lock_cycles=cycles,
            matrix_error=matrix_error,
            q_residual=q_residual,
        )
        self.results.append(result)

        if locked:
            self.metrics["lock_time_cycles"].append(cycles)
            if matrix_error is not None:
                self.metrics["final_matrix_error"].append(matrix_error)
            if q_residual is not None:
                self.metrics["q_residual"].append(q_residual)

    def record_evaluation(
        self,
        test_name: str,
        passed: bool,
        output_error: float,
        details: dict[str, Any] | None = None,
    ):
        """
        Record evaluation test result.

        Args:
            test_name: Name of the test
            passed: Whether evaluation passed
            output_error: RMS output error
            details: Additional test details
        """
        result = TestResult(
            test_name=test_name,
            passed=passed,
            output_error=output_error,
            details=details or {},
        )
        self.results.append(result)
        self.metrics["output_error"].append(output_error)

    def record_phase_trajectory(
        self, phases: list[list[float]]
    ):
        """
        Record phase trajectory during calibration.

        Args:
            phases: List of phase values at each iteration
        """
        self.metrics["phase_trajectory"].extend(phases)

    def to_json(self, path: str | Path):
        """
        Write metrics to JSON file.

        Args:
            path: Output file path
        """
        output = {
            "summary": self.get_summary(),
            "results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "lock_cycles": r.lock_cycles,
                    "matrix_error": r.matrix_error,
                    "q_residual": r.q_residual,
                    "output_error": r.output_error,
                    "details": r.details,
                }
                for r in self.results
            ],
            "metrics": {
                k: v for k, v in self.metrics.items()
                if k != "phase_trajectory"  # Exclude large trajectory data
            },
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Dictionary of summary statistics
        """
        lock_times = self.metrics["lock_time_cycles"]
        matrix_errors = self.metrics["final_matrix_error"]
        q_residuals = self.metrics["q_residual"]
        output_errors = self.metrics["output_error"]

        return {
            "total_tests": len(self.results),
            "passed_tests": sum(1 for r in self.results if r.passed),
            "failed_tests": sum(1 for r in self.results if not r.passed),
            "avg_lock_cycles": float(np.mean(lock_times)) if lock_times else None,
            "max_lock_cycles": int(np.max(lock_times)) if lock_times else None,
            "avg_matrix_error": float(np.mean(matrix_errors)) if matrix_errors else None,
            "max_matrix_error": float(np.max(matrix_errors)) if matrix_errors else None,
            "avg_q_residual": float(np.mean(q_residuals)) if q_residuals else None,
            "avg_output_error": float(np.mean(output_errors)) if output_errors else None,
            "max_output_error": float(np.max(output_errors)) if output_errors else None,
        }

    def check_regression(
        self,
        thresholds: dict[str, float] | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Check if metrics pass regression thresholds.

        Args:
            thresholds: Dictionary of metric thresholds. Defaults to:
                - max_lock_cycles: 5000
                - max_matrix_error: 5e-3
                - max_q_residual: 1e-2
                - max_output_error: 0.01

        Returns:
            Tuple of (passed, violations) where violations is a list
            of threshold violations
        """
        if thresholds is None:
            thresholds = {
                "max_lock_cycles": 5000,
                "max_matrix_error": 5e-3,
                "max_q_residual": 1e-2,
                "max_output_error": 0.01,
            }

        violations = []
        summary = self.get_summary()

        if (
            "max_lock_cycles" in thresholds
            and summary["max_lock_cycles"] is not None
        ):
            if summary["max_lock_cycles"] > thresholds["max_lock_cycles"]:
                violations.append(
                    f"max_lock_cycles: {summary['max_lock_cycles']} > "
                    f"{thresholds['max_lock_cycles']}"
                )

        if (
            "max_matrix_error" in thresholds
            and summary["max_matrix_error"] is not None
        ):
            if summary["max_matrix_error"] > thresholds["max_matrix_error"]:
                violations.append(
                    f"max_matrix_error: {summary['max_matrix_error']:.2e} > "
                    f"{thresholds['max_matrix_error']:.2e}"
                )

        if (
            "max_q_residual" in thresholds
            and summary["avg_q_residual"] is not None
        ):
            if summary["avg_q_residual"] > thresholds["max_q_residual"]:
                violations.append(
                    f"avg_q_residual: {summary['avg_q_residual']:.2e} > "
                    f"{thresholds['max_q_residual']:.2e}"
                )

        if (
            "max_output_error" in thresholds
            and summary["max_output_error"] is not None
        ):
            if summary["max_output_error"] > thresholds["max_output_error"]:
                violations.append(
                    f"max_output_error: {summary['max_output_error']:.2e} > "
                    f"{thresholds['max_output_error']:.2e}"
                )

        return len(violations) == 0, violations

    def print_summary(self):
        """Print summary to console."""
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total tests:      {summary['total_tests']}")
        print(f"Passed:           {summary['passed_tests']}")
        print(f"Failed:           {summary['failed_tests']}")
        print("-" * 60)

        if summary["avg_lock_cycles"] is not None:
            print(f"Avg lock cycles:  {summary['avg_lock_cycles']:.0f}")
            print(f"Max lock cycles:  {summary['max_lock_cycles']}")

        if summary["avg_matrix_error"] is not None:
            print(f"Avg matrix error: {summary['avg_matrix_error']:.2e}")
            print(f"Max matrix error: {summary['max_matrix_error']:.2e}")

        if summary["avg_q_residual"] is not None:
            print(f"Avg Q-residual:   {summary['avg_q_residual']:.2e}")

        if summary["avg_output_error"] is not None:
            print(f"Avg output error: {summary['avg_output_error']:.2e}")
            print(f"Max output error: {summary['max_output_error']:.2e}")

        print("=" * 60)

        # Check regression
        passed, violations = self.check_regression()
        if passed:
            print("REGRESSION: PASSED")
        else:
            print("REGRESSION: FAILED")
            for v in violations:
                print(f"  - {v}")
        print("=" * 60 + "\n")
