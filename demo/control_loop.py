"""
Python-based control loop for coherent photonic matrix multiply.

This implements the calibration algorithm in Python for demonstration
and debugging, separate from the RTL implementation.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Callable
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from plant import PhotonicPlant


@dataclass
class ControlConfig:
    """Configuration for the control loop."""

    # Phase update parameters
    initial_step: float = 0.3  # Initial phase step in radians
    min_step: float = 0.01  # Minimum phase step
    step_decay: float = 0.8  # Step size decay factor

    # Convergence parameters
    error_threshold: float = 1e-4  # Error threshold for lock
    lock_count: int = 5  # Consecutive iterations below threshold
    max_iterations: int = 500  # Maximum iterations

    # Measurement parameters
    num_averages: int = 4  # Samples to average per measurement

    # Algorithm selection
    algorithm: str = "coordinate_descent"  # or "gradient_descent"
    learning_rate: float = 0.1  # For gradient descent


@dataclass
class ControlState:
    """State of the control loop."""

    phases: NDArray[np.float64] = field(default_factory=lambda: np.zeros(4))
    error: float = float("inf")
    iteration: int = 0
    locked: bool = False
    lock_counter: int = 0
    step_size: float = 0.3

    # History for visualization
    phase_history: list = field(default_factory=list)
    error_history: list = field(default_factory=list)
    matrix_history: list = field(default_factory=list)


class CoherentController:
    """
    Closed-loop controller for coherent photonic matrix multiply.

    Implements coordinate descent optimization to find phase settings
    that realize a target matrix using only I/Q measurements.
    """

    def __init__(
        self,
        plant: PhotonicPlant,
        config: ControlConfig | None = None,
    ):
        """
        Initialize controller.

        Args:
            plant: PhotonicPlant instance to control
            config: Control configuration
        """
        self.plant = plant
        self.config = config or ControlConfig()
        self.state = ControlState(step_size=self.config.initial_step)

        # Target matrix (real weights)
        self._target = np.eye(2, dtype=np.float64)

    def set_target(self, w0: float, w1: float, w2: float, w3: float):
        """
        Set target matrix weights.

        Args:
            w0, w1, w2, w3: Target weights in [-1, 1]
        """
        self._target = np.array([[w0, w1], [w2, w3]], dtype=np.float64)

    @property
    def target_matrix(self) -> NDArray[np.float64]:
        """Get target matrix."""
        return self._target.copy()

    def reset(self):
        """Reset controller state."""
        self.state = ControlState(step_size=self.config.initial_step)
        # Initialize phases to random starting point
        self.state.phases = np.random.uniform(0, 2 * np.pi, 4)
        self.plant.reset()

    def measure_matrix(self) -> NDArray[np.complex128]:
        """
        Measure the current transfer matrix using basis inputs.

        Returns:
            2x2 complex measured matrix
        """
        # Convert phases to DAC codes
        dac_codes = self._phases_to_dac(self.state.phases)

        # Measure column 0: input [1, 0]
        m00, m10 = self._measure_column(dac_codes, 1.0, 0.0)

        # Measure column 1: input [0, 1]
        m01, m11 = self._measure_column(dac_codes, 0.0, 1.0)

        return np.array([[m00, m01], [m10, m11]], dtype=np.complex128)

    def _measure_column(
        self, dac_codes: list[int], x0: float, x1: float
    ) -> tuple[complex, complex]:
        """Measure one column of the matrix with averaging."""
        y0_sum = 0j
        y1_sum = 0j

        for _ in range(self.config.num_averages):
            i0, q0, i1, q1 = self.plant.sample_outputs(
                dac_codes, complex(x0, 0), complex(x1, 0)
            )
            # Reconstruct complex values
            y0_sum += complex(i0, q0) / self.plant.receiver.receivers[0].gain
            y1_sum += complex(i1, q1) / self.plant.receiver.receivers[1].gain

        return y0_sum / self.config.num_averages, y1_sum / self.config.num_averages

    def _phases_to_dac(self, phases: NDArray[np.float64]) -> list[int]:
        """Convert phase values to DAC codes."""
        dac_max = (1 << self.plant.dac_bits) - 1
        return [
            int((p % (2 * np.pi)) / (2 * np.pi) * dac_max) for p in phases
        ]

    def compute_error(self, measured: NDArray[np.complex128]) -> float:
        """
        Compute error between measured and target matrices.

        For real targets, we minimize:
        - Real part difference from target
        - Imaginary part (should be zero)

        Args:
            measured: Measured complex matrix

        Returns:
            Scalar error value
        """
        real_diff = np.real(measured) - self._target
        imag_part = np.imag(measured)

        # Frobenius norm of differences
        error = np.sum(real_diff**2) + np.sum(imag_part**2)
        return float(error)

    def step(self) -> bool:
        """
        Execute one iteration of the control loop.

        Returns:
            True if locked, False otherwise
        """
        if self.state.locked:
            return True

        self.state.iteration += 1

        if self.config.algorithm == "coordinate_descent":
            self._step_coordinate_descent()
        else:
            self._step_gradient_descent()

        # Record history
        self.state.phase_history.append(self.state.phases.copy())
        self.state.error_history.append(self.state.error)

        measured = self.measure_matrix()
        self.state.matrix_history.append(measured.copy())

        # Check convergence
        if self.state.error < self.config.error_threshold:
            self.state.lock_counter += 1
            if self.state.lock_counter >= self.config.lock_count:
                self.state.locked = True
        else:
            self.state.lock_counter = 0

        # Check max iterations
        if self.state.iteration >= self.config.max_iterations:
            return False

        return self.state.locked

    def _step_coordinate_descent(self):
        """One iteration of coordinate descent with adaptive step sizing."""
        # Measure current error
        measured = self.measure_matrix()
        error_current = self.compute_error(measured)
        original_phases = self.state.phases.copy()

        improved = False
        total_improvement = 0.0

        # Try each phase
        for i in range(4):
            best_phase = self.state.phases[i]
            best_error = error_current

            # Try multiple step sizes for this phase
            for step_mult in [1.0, 0.5, 0.25]:
                step = self.state.step_size * step_mult

                # Try positive step
                self.state.phases[i] = original_phases[i] + step
                measured_plus = self.measure_matrix()
                error_plus = self.compute_error(measured_plus)

                if error_plus < best_error:
                    best_phase = self.state.phases[i]
                    best_error = error_plus

                # Try negative step
                self.state.phases[i] = original_phases[i] - step
                measured_minus = self.measure_matrix()
                error_minus = self.compute_error(measured_minus)

                if error_minus < best_error:
                    best_phase = self.state.phases[i]
                    best_error = error_minus

            # Apply best phase for this dimension
            if best_error < error_current:
                self.state.phases[i] = best_phase
                total_improvement += error_current - best_error
                error_current = best_error
                improved = True
            else:
                self.state.phases[i] = original_phases[i]

            # Update original for next dimension
            original_phases[i] = self.state.phases[i]

        self.state.error = error_current

        # Adaptive step size: decay if no improvement, or if improvement is small
        if not improved or (total_improvement < error_current * 0.01):
            self.state.step_size = max(
                self.config.min_step,
                self.state.step_size * self.config.step_decay,
            )

    def _step_gradient_descent(self):
        """One iteration of gradient descent using finite differences."""
        measured = self.measure_matrix()
        error_current = self.compute_error(measured)

        # Compute gradient via finite differences
        gradient = np.zeros(4)
        delta = 0.01  # Small perturbation

        for i in range(4):
            phases_plus = self.state.phases.copy()
            phases_plus[i] += delta
            self.state.phases = phases_plus
            measured_plus = self.measure_matrix()
            error_plus = self.compute_error(measured_plus)

            gradient[i] = (error_plus - error_current) / delta
            self.state.phases[i] -= delta

        # Update phases
        self.state.phases -= self.config.learning_rate * gradient
        self.state.phases = self.state.phases % (2 * np.pi)

        # Measure new error
        measured = self.measure_matrix()
        self.state.error = self.compute_error(measured)

    def run_calibration(
        self,
        callback: Callable[[ControlState], None] | None = None,
    ) -> bool:
        """
        Run full calibration loop.

        Args:
            callback: Optional callback called after each iteration

        Returns:
            True if locked, False if max iterations reached
        """
        self.reset()

        while self.state.iteration < self.config.max_iterations:
            locked = self.step()

            if callback:
                callback(self.state)

            if locked:
                return True

        return False

    def evaluate(self, x0: float, x1: float) -> tuple[float, float]:
        """
        Evaluate the locked matrix on an input.

        Args:
            x0, x1: Input values

        Returns:
            Output values (y0, y1)
        """
        dac_codes = self._phases_to_dac(self.state.phases)

        # Average multiple samples
        y0_sum = 0.0
        y1_sum = 0.0

        for _ in range(self.config.num_averages):
            i0, q0, i1, q1 = self.plant.sample_outputs(
                dac_codes, complex(x0, 0), complex(x1, 0)
            )
            # For real targets, use I channel
            y0_sum += i0 / self.plant.receiver.receivers[0].gain
            y1_sum += i1 / self.plant.receiver.receivers[1].gain

        return y0_sum / self.config.num_averages, y1_sum / self.config.num_averages


def demo_control_loop():
    """Simple demo of the control loop."""
    print("=" * 60)
    print("Coherent Photonic Matrix Multiply - Control Loop Demo")
    print("=" * 60)

    # Create plant with realistic parameters
    plant = PhotonicPlant(
        drift_rate=0.0,  # No drift for demo
        noise_std=2.0,  # Low noise
        tau_thermal=1e-9,  # Fast thermal for demo
        crosstalk_coeff=0.0,
        seed=42,
    )

    # Create controller
    config = ControlConfig(
        initial_step=0.5,
        min_step=0.01,
        error_threshold=1e-3,
        max_iterations=200,
    )
    controller = CoherentController(plant, config)

    # Test different target matrices
    test_cases = [
        ("Identity", 1.0, 0.0, 0.0, 1.0),
        ("Swap", 0.0, 1.0, 1.0, 0.0),
        ("Hadamard-like", 0.707, 0.707, 0.707, -0.707),
        ("Custom", 0.5, -0.3, 0.8, 0.2),
    ]

    for name, w0, w1, w2, w3 in test_cases:
        print(f"\nTarget: {name} matrix")
        print(f"  [[{w0:6.3f}, {w1:6.3f}],")
        print(f"   [{w2:6.3f}, {w3:6.3f}]]")

        controller.set_target(w0, w1, w2, w3)

        def progress(state: ControlState):
            if state.iteration % 20 == 0:
                print(f"  Iteration {state.iteration:3d}: error = {state.error:.2e}")

        locked = controller.run_calibration(callback=progress)

        if locked:
            print(f"  LOCKED in {controller.state.iteration} iterations!")
            print(f"  Final error: {controller.state.error:.2e}")

            # Show achieved matrix
            measured = controller.measure_matrix()
            print(f"  Achieved matrix (real part):")
            print(f"    [[{np.real(measured[0, 0]):6.3f}, {np.real(measured[0, 1]):6.3f}],")
            print(f"     [{np.real(measured[1, 0]):6.3f}, {np.real(measured[1, 1]):6.3f}]]")

            # Test evaluation
            test_input = (0.5, 0.5)
            y0, y1 = controller.evaluate(*test_input)
            expected = np.array([[w0, w1], [w2, w3]]) @ np.array(test_input)
            print(f"  Test input {test_input}:")
            print(f"    Output: ({y0:.3f}, {y1:.3f})")
            print(f"    Expected: ({expected[0]:.3f}, {expected[1]:.3f})")
        else:
            print(f"  Failed to lock after {config.max_iterations} iterations")
            print(f"  Final error: {controller.state.error:.2e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_control_loop()
