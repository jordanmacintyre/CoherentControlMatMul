"""
SVD-based control loop for arbitrary 2x2 matrix realization.

Implements calibration for the SVD photonic architecture (M = U·Σ·V†)
using sequential calibration of components followed by optional joint refinement.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Callable
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from plant.svd_plant import SVDPhotonicPlant
from plant.mzi_mesh import MZIMesh


@dataclass
class SVDControlConfig:
    """Configuration for the SVD control loop."""

    # Phase update parameters (shared for V and U calibration)
    initial_step: float = 0.3
    min_step: float = 0.01
    step_decay: float = 0.8

    # Convergence parameters
    error_threshold: float = 1e-4
    lock_count: int = 5
    max_iterations: int = 500  # Per stage

    # Measurement parameters
    num_averages: int = 4

    # SVD-specific parameters
    joint_refinement: bool = True
    refinement_iterations: int = 100
    refinement_threshold: float = 1e-5


@dataclass
class SVDControlState:
    """State of the SVD control loop."""

    # Phase state for each component
    phases_v: NDArray[np.float64] = field(default_factory=lambda: np.zeros(4))
    sigma_dac: list[int] = field(default_factory=lambda: [0, 0])
    phases_u: NDArray[np.float64] = field(default_factory=lambda: np.zeros(4))

    # Overall state
    error: float = float("inf")
    iteration: int = 0
    locked: bool = False

    # Stage tracking
    stage: str = "idle"  # idle, calibrating_v, setting_sigma, calibrating_u, refining
    v_locked: bool = False
    u_locked: bool = False

    # Per-stage counters
    stage_iteration: int = 0
    lock_counter: int = 0
    step_size: float = 0.3

    # History for visualization
    phase_history_v: list = field(default_factory=list)
    phase_history_u: list = field(default_factory=list)
    sigma_history: list = field(default_factory=list)
    error_history: list = field(default_factory=list)
    matrix_history: list = field(default_factory=list)


class SVDCoherentController:
    """
    Controller for SVD-based arbitrary matrix realization.

    Calibrates the photonic circuit to realize M = U·Σ·V† where:
    - V† is calibrated first (input-side unitary)
    - Σ is set directly from computed singular values
    - U is calibrated second (output-side unitary)
    - Optional joint refinement optimizes all parameters together
    """

    def __init__(
        self,
        plant: SVDPhotonicPlant,
        config: SVDControlConfig | None = None,
    ):
        """
        Initialize SVD controller.

        Args:
            plant: SVDPhotonicPlant instance to control
            config: Control configuration
        """
        self.plant = plant
        self.config = config or SVDControlConfig()
        self.state = SVDControlState(step_size=self.config.initial_step)

        # Target decomposition
        self._target = np.eye(2, dtype=np.float64)
        self._target_u: NDArray[np.complex128] = np.eye(2, dtype=np.complex128)
        self._target_sigma: NDArray[np.float64] = np.ones(2)
        self._target_vh: NDArray[np.complex128] = np.eye(2, dtype=np.complex128)

    def set_target(self, w0: float, w1: float, w2: float, w3: float):
        """
        Set target matrix and compute SVD decomposition.

        Args:
            w0, w1, w2, w3: Target weights in [-1, 1]
        """
        self._target = np.array([[w0, w1], [w2, w3]], dtype=np.float64)

        # Compute SVD: M = U · Σ · V†
        U, sigma, Vh = np.linalg.svd(self._target)

        self._target_u = U.astype(np.complex128)
        self._target_sigma = sigma
        self._target_vh = Vh.astype(np.complex128)

    @property
    def target_matrix(self) -> NDArray[np.float64]:
        """Get target matrix."""
        return self._target.copy()

    @property
    def target_svd(self) -> tuple[NDArray, NDArray, NDArray]:
        """Get target SVD components (U, sigma, Vh)."""
        return self._target_u.copy(), self._target_sigma.copy(), self._target_vh.copy()

    def reset(self):
        """Reset controller state."""
        self.state = SVDControlState(step_size=self.config.initial_step)
        # Random initialization for phases
        self.state.phases_v = np.random.uniform(0, 2 * np.pi, 4)
        self.state.phases_u = np.random.uniform(0, 2 * np.pi, 4)
        self.plant.reset()

    def _get_combined_dac_codes(self) -> list[int]:
        """Get combined DAC codes for all parameters."""
        v_codes = self._phases_to_dac(self.state.phases_v)
        u_codes = self._phases_to_dac(self.state.phases_u)
        return v_codes + self.state.sigma_dac + u_codes

    def _phases_to_dac(self, phases: NDArray[np.float64]) -> list[int]:
        """Convert phase values to DAC codes."""
        dac_max = (1 << self.plant.dac_bits) - 1
        return [
            int((p % (2 * np.pi)) / (2 * np.pi) * dac_max) for p in phases
        ]

    def measure_matrix(self) -> NDArray[np.complex128]:
        """
        Measure the current transfer matrix using basis inputs.

        Returns:
            2x2 complex measured matrix
        """
        dac_codes = self._get_combined_dac_codes()

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
            y0_sum += complex(i0, q0) / self.plant.receiver.receivers[0].gain
            y1_sum += complex(i1, q1) / self.plant.receiver.receivers[1].gain

        return y0_sum / self.config.num_averages, y1_sum / self.config.num_averages

    def _measure_unitary(
        self, mesh: str, phases: NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        """
        Measure transfer matrix for one MZI mesh in isolation.

        For V†: Measure with Σ = I (no attenuation) and U = I
        For U: Measure with V† = I and Σ = I

        Args:
            mesh: "v" or "u"
            phases: Phase values for the mesh

        Returns:
            2x2 complex measured matrix
        """
        # Set up identity for other components
        v_phases = phases if mesh == "v" else np.zeros(4)
        u_phases = phases if mesh == "u" else np.zeros(4)

        # No attenuation (σ = 1)
        sigma_dac = [0, 0]

        v_codes = self._phases_to_dac(v_phases)
        u_codes = self._phases_to_dac(u_phases)
        dac_codes = v_codes + sigma_dac + u_codes

        # Measure
        m00, m10 = self._measure_column(dac_codes, 1.0, 0.0)
        m01, m11 = self._measure_column(dac_codes, 0.0, 1.0)

        return np.array([[m00, m01], [m10, m11]], dtype=np.complex128)

    def compute_error(self, measured: NDArray[np.complex128]) -> float:
        """
        Compute error between measured and target matrices.

        Args:
            measured: Measured complex matrix

        Returns:
            Scalar error value (Frobenius norm squared)
        """
        real_diff = np.real(measured) - self._target
        imag_part = np.imag(measured)
        return float(np.sum(real_diff**2) + np.sum(imag_part**2))

    def _compute_unitary_error(
        self, measured: NDArray[np.complex128], target: NDArray[np.complex128]
    ) -> float:
        """Compute error for unitary calibration (magnitude matters, phase ambiguity OK)."""
        # For unitary targets, we care about |measured - target|
        # But there's global phase ambiguity, so we minimize over global phase
        diff = measured - target
        error_base = np.sum(np.abs(diff)**2)

        # Try phase-corrected version
        phase_correction = np.angle(np.sum(measured.conj() * target))
        diff_corrected = measured * np.exp(-1j * phase_correction) - target
        error_corrected = np.sum(np.abs(diff_corrected)**2)

        return float(min(error_base, error_corrected))

    def _calibrate_mesh(
        self,
        mesh: str,
        target: NDArray[np.complex128],
        callback: Callable | None = None,
    ) -> bool:
        """
        Calibrate one MZI mesh to approximate a target unitary.

        Uses coordinate descent on the 4 phase parameters.

        Note: The MZI structure [[e^{jφ0}cos(θ/2), j·sin(θ/2)],
                                  [j·sin(θ/2), e^{jφ1}cos(θ/2)]]
        has equal off-diagonal elements (both j·sin), so it cannot exactly
        realize matrices with opposite-sign off-diagonals. The calibration
        finds the best approximation.

        Args:
            mesh: "v" or "u"
            target: Target unitary matrix
            callback: Optional progress callback

        Returns:
            True if locked, False otherwise
        """
        # Try to get initial phases from decomposition
        # If decomposition gives poor results, we'll still optimize from there
        try:
            phases = MZIMesh.decompose_unitary(target)
            # Verify the decomposition isn't too bad
            test_mesh = MZIMesh()
            reconstructed = test_mesh.compute_transfer_matrix(phases)
            decomp_error = np.linalg.norm(reconstructed - target)
            if decomp_error > 0.5:
                # Decomposition is poor, use random init instead
                phases = np.random.uniform(0, 2 * np.pi, 4)
        except Exception:
            phases = np.random.uniform(0, 2 * np.pi, 4)

        if mesh == "v":
            self.state.phases_v = phases.copy()
        else:
            self.state.phases_u = phases.copy()

        self.state.stage_iteration = 0
        self.state.lock_counter = 0
        self.state.step_size = self.config.initial_step

        while self.state.stage_iteration < self.config.max_iterations:
            self.state.iteration += 1
            self.state.stage_iteration += 1

            # Get current phases
            if mesh == "v":
                current_phases = self.state.phases_v
            else:
                current_phases = self.state.phases_u

            # Measure current state
            measured = self._measure_unitary(mesh, current_phases)
            error_current = self._compute_unitary_error(measured, target)

            # Coordinate descent
            improved = False
            original_phases = current_phases.copy()

            for i in range(4):
                best_phase = current_phases[i]
                best_error = error_current

                for step_mult in [1.0, 0.5, 0.25]:
                    step = self.state.step_size * step_mult

                    # Try positive step
                    current_phases[i] = original_phases[i] + step
                    measured_plus = self._measure_unitary(mesh, current_phases)
                    error_plus = self._compute_unitary_error(measured_plus, target)

                    if error_plus < best_error:
                        best_phase = current_phases[i]
                        best_error = error_plus

                    # Try negative step
                    current_phases[i] = original_phases[i] - step
                    measured_minus = self._measure_unitary(mesh, current_phases)
                    error_minus = self._compute_unitary_error(measured_minus, target)

                    if error_minus < best_error:
                        best_phase = current_phases[i]
                        best_error = error_minus

                if best_error < error_current:
                    current_phases[i] = best_phase
                    error_current = best_error
                    improved = True
                else:
                    current_phases[i] = original_phases[i]

                original_phases[i] = current_phases[i]

            # Update state
            if mesh == "v":
                self.state.phases_v = current_phases
                self.state.phase_history_v.append(current_phases.copy())
            else:
                self.state.phases_u = current_phases
                self.state.phase_history_u.append(current_phases.copy())

            # Adaptive step size
            if not improved:
                self.state.step_size = max(
                    self.config.min_step,
                    self.state.step_size * self.config.step_decay,
                )

            # Record history
            self.state.error_history.append(error_current)

            # Check convergence
            if error_current < self.config.error_threshold:
                self.state.lock_counter += 1
                if self.state.lock_counter >= self.config.lock_count:
                    if mesh == "v":
                        self.state.v_locked = True
                    else:
                        self.state.u_locked = True
                    return True
            else:
                self.state.lock_counter = 0

            if callback:
                callback(self.state)

        return False

    def _set_sigma(self):
        """Set VOA DAC codes directly from target singular values."""
        self.state.sigma_dac = self.plant.voas.sigma_to_dac(self._target_sigma)
        self.state.sigma_history.append(self._target_sigma.copy())

    def _joint_refinement(self, callback: Callable | None = None) -> bool:
        """
        Joint refinement of all 10 parameters.

        After sequential calibration, fine-tune all parameters together
        to minimize the full matrix error.

        Args:
            callback: Optional progress callback

        Returns:
            True if converged to threshold
        """
        self.state.stage = "refining"
        self.state.stage_iteration = 0
        self.state.lock_counter = 0
        self.state.step_size = self.config.min_step * 2  # Start smaller for refinement

        for _ in range(self.config.refinement_iterations):
            self.state.iteration += 1
            self.state.stage_iteration += 1

            # Measure full matrix
            measured = self.measure_matrix()
            error_current = self.compute_error(measured)
            self.state.error = error_current

            # Record
            self.state.error_history.append(error_current)
            self.state.phase_history_v.append(self.state.phases_v.copy())
            self.state.phase_history_u.append(self.state.phases_u.copy())
            self.state.matrix_history.append(measured.copy())

            # Check convergence
            if error_current < self.config.refinement_threshold:
                self.state.lock_counter += 1
                if self.state.lock_counter >= self.config.lock_count:
                    self.state.locked = True
                    return True
            else:
                self.state.lock_counter = 0

            # Coordinate descent on all 8 phases (V and U)
            improved = False

            # V phases
            original_v = self.state.phases_v.copy()
            for i in range(4):
                best_phase = self.state.phases_v[i]
                best_error = error_current

                for delta in [self.state.step_size, -self.state.step_size]:
                    self.state.phases_v[i] = original_v[i] + delta
                    m = self.measure_matrix()
                    e = self.compute_error(m)
                    if e < best_error:
                        best_phase = self.state.phases_v[i]
                        best_error = e

                if best_error < error_current:
                    self.state.phases_v[i] = best_phase
                    error_current = best_error
                    improved = True
                else:
                    self.state.phases_v[i] = original_v[i]

            # U phases
            original_u = self.state.phases_u.copy()
            for i in range(4):
                best_phase = self.state.phases_u[i]
                best_error = error_current

                for delta in [self.state.step_size, -self.state.step_size]:
                    self.state.phases_u[i] = original_u[i] + delta
                    m = self.measure_matrix()
                    e = self.compute_error(m)
                    if e < best_error:
                        best_phase = self.state.phases_u[i]
                        best_error = e

                if best_error < error_current:
                    self.state.phases_u[i] = best_phase
                    error_current = best_error
                    improved = True
                else:
                    self.state.phases_u[i] = original_u[i]

            # Adaptive step
            if not improved:
                self.state.step_size = max(
                    self.config.min_step / 2,
                    self.state.step_size * 0.9,
                )

            if callback:
                callback(self.state)

        # Check final error
        measured = self.measure_matrix()
        self.state.error = self.compute_error(measured)
        return self.state.error < self.config.error_threshold

    def run_calibration(
        self,
        callback: Callable[[SVDControlState], None] | None = None,
    ) -> bool:
        """
        Run full SVD calibration.

        The calibration focuses on minimizing the final matrix error ||M - target||²
        rather than individual component errors. This is necessary because the MZI
        structure has constraints that prevent exact realization of arbitrary unitaries,
        but the product U·Σ·V† can still achieve the target real matrix.

        Strategy:
        1. Initialize phases randomly or from SVD decomposition
        2. Set Σ directly from target singular values
        3. Run joint optimization on all 8 phase parameters

        Args:
            callback: Optional callback called after iterations

        Returns:
            True if locked, False if failed
        """
        self.reset()

        # Initialize with SVD decomposition as starting point
        self.state.stage = "initializing"

        # Try to get reasonable starting phases from decomposition
        try:
            phases_v = MZIMesh.decompose_unitary(self._target_vh)
            phases_u = MZIMesh.decompose_unitary(self._target_u)
        except Exception:
            phases_v = np.random.uniform(0, 2 * np.pi, 4)
            phases_u = np.random.uniform(0, 2 * np.pi, 4)

        self.state.phases_v = phases_v
        self.state.phases_u = phases_u

        # Set Σ directly from singular values
        self.state.stage = "setting_sigma"
        self._set_sigma()

        # Measure initial error
        measured = self.measure_matrix()
        self.state.error = self.compute_error(measured)
        self.state.matrix_history.append(measured.copy())

        # Main calibration: joint optimization of all parameters
        # This directly minimizes ||M_realized - M_target||²
        self.state.stage = "calibrating"
        locked = self._full_calibration(callback)

        # Final check
        measured = self.measure_matrix()
        self.state.error = self.compute_error(measured)

        if self.state.error < self.config.error_threshold:
            self.state.locked = True
            locked = True

        return locked

    def _full_calibration(self, callback: Callable | None = None) -> bool:
        """
        Full joint calibration of all phase parameters.

        Optimizes all 8 phases (4 for V†, 4 for U) to minimize
        the final matrix error against the target.

        Args:
            callback: Optional progress callback

        Returns:
            True if converged, False otherwise
        """
        self.state.lock_counter = 0
        self.state.step_size = self.config.initial_step

        max_iters = self.config.max_iterations + self.config.refinement_iterations

        for _ in range(max_iters):
            self.state.iteration += 1
            self.state.stage_iteration += 1

            # Measure full matrix and error
            measured = self.measure_matrix()
            error_current = self.compute_error(measured)
            self.state.error = error_current

            # Record history
            self.state.error_history.append(error_current)
            self.state.phase_history_v.append(self.state.phases_v.copy())
            self.state.phase_history_u.append(self.state.phases_u.copy())
            self.state.matrix_history.append(measured.copy())

            # Check convergence
            if error_current < self.config.error_threshold:
                self.state.lock_counter += 1
                if self.state.lock_counter >= self.config.lock_count:
                    self.state.locked = True
                    self.state.v_locked = True
                    self.state.u_locked = True
                    return True
            else:
                self.state.lock_counter = 0

            # Coordinate descent on all 8 phases
            improved = False
            total_improvement = 0.0

            # V phases
            original_v = self.state.phases_v.copy()
            for i in range(4):
                best_phase = self.state.phases_v[i]
                best_error = error_current

                for step_mult in [1.0, 0.5, 0.25]:
                    step = self.state.step_size * step_mult

                    # Try positive step
                    self.state.phases_v[i] = original_v[i] + step
                    m = self.measure_matrix()
                    e = self.compute_error(m)
                    if e < best_error:
                        best_phase = self.state.phases_v[i]
                        best_error = e

                    # Try negative step
                    self.state.phases_v[i] = original_v[i] - step
                    m = self.measure_matrix()
                    e = self.compute_error(m)
                    if e < best_error:
                        best_phase = self.state.phases_v[i]
                        best_error = e

                if best_error < error_current:
                    self.state.phases_v[i] = best_phase
                    total_improvement += error_current - best_error
                    error_current = best_error
                    improved = True
                else:
                    self.state.phases_v[i] = original_v[i]

                original_v[i] = self.state.phases_v[i]

            # U phases
            original_u = self.state.phases_u.copy()
            for i in range(4):
                best_phase = self.state.phases_u[i]
                best_error = error_current

                for step_mult in [1.0, 0.5, 0.25]:
                    step = self.state.step_size * step_mult

                    # Try positive step
                    self.state.phases_u[i] = original_u[i] + step
                    m = self.measure_matrix()
                    e = self.compute_error(m)
                    if e < best_error:
                        best_phase = self.state.phases_u[i]
                        best_error = e

                    # Try negative step
                    self.state.phases_u[i] = original_u[i] - step
                    m = self.measure_matrix()
                    e = self.compute_error(m)
                    if e < best_error:
                        best_phase = self.state.phases_u[i]
                        best_error = e

                if best_error < error_current:
                    self.state.phases_u[i] = best_phase
                    total_improvement += error_current - best_error
                    error_current = best_error
                    improved = True
                else:
                    self.state.phases_u[i] = original_u[i]

                original_u[i] = self.state.phases_u[i]

            self.state.error = error_current

            # Adaptive step size
            if not improved or (total_improvement < error_current * 0.01):
                self.state.step_size = max(
                    self.config.min_step,
                    self.state.step_size * self.config.step_decay,
                )

            if callback:
                callback(self.state)

        return self.state.error < self.config.error_threshold

    def evaluate(self, x0: float, x1: float) -> tuple[float, float]:
        """
        Evaluate the calibrated matrix on an input.

        Args:
            x0, x1: Input values

        Returns:
            Output values (y0, y1)
        """
        dac_codes = self._get_combined_dac_codes()

        y0_sum = 0.0
        y1_sum = 0.0

        for _ in range(self.config.num_averages):
            i0, q0, i1, q1 = self.plant.sample_outputs(
                dac_codes, complex(x0, 0), complex(x1, 0)
            )
            y0_sum += i0 / self.plant.receiver.receivers[0].gain
            y1_sum += i1 / self.plant.receiver.receivers[1].gain

        return y0_sum / self.config.num_averages, y1_sum / self.config.num_averages

    def get_current_dac_codes(self) -> list[int]:
        """Get the current combined DAC codes."""
        return self._get_combined_dac_codes()


def demo_svd_control():
    """Demo of SVD control loop with non-unitary matrices."""
    print("=" * 60)
    print("SVD Photonic Matrix Multiply - Control Loop Demo")
    print("=" * 60)

    # Create SVD plant
    plant = SVDPhotonicPlant(
        drift_rate=0.0,
        noise_std=2.0,
        tau_thermal=1e-9,
        crosstalk_coeff=0.0,
        seed=42,
    )

    # Create controller
    config = SVDControlConfig(
        initial_step=0.5,
        min_step=0.01,
        error_threshold=1e-3,
        max_iterations=200,
        joint_refinement=True,
        refinement_iterations=100,
    )
    controller = SVDCoherentController(plant, config)

    # Test matrices including non-unitary ones
    test_cases = [
        ("Identity", 1.0, 0.0, 0.0, 1.0),
        ("Hadamard-like", 0.707, 0.707, 0.707, -0.707),
        ("Attenuator (non-unitary)", 0.5, 0.0, 0.0, 0.3),
        ("Arbitrary (previously failing)", 0.91, -0.32, 0.15, -0.74),
    ]

    for name, w0, w1, w2, w3 in test_cases:
        print(f"\nTarget: {name}")
        print(f"  [[{w0:6.3f}, {w1:6.3f}],")
        print(f"   [{w2:6.3f}, {w3:6.3f}]]")

        # Show SVD decomposition
        M = np.array([[w0, w1], [w2, w3]])
        U, sigma, Vh = np.linalg.svd(M)
        print(f"  SVD: σ = [{sigma[0]:.3f}, {sigma[1]:.3f}]")

        controller.set_target(w0, w1, w2, w3)

        def progress(state: SVDControlState):
            if state.stage_iteration % 20 == 0:
                print(f"  [{state.stage}] Iter {state.stage_iteration:3d}: "
                      f"error = {state.error_history[-1] if state.error_history else float('inf'):.2e}")

        locked = controller.run_calibration(callback=progress)

        if locked:
            print(f"  LOCKED in {controller.state.iteration} total iterations!")
            print(f"  Final error: {controller.state.error:.2e}")

            # Show achieved matrix
            measured = controller.measure_matrix()
            print(f"  Achieved matrix (real part):")
            print(f"    [[{np.real(measured[0, 0]):6.3f}, {np.real(measured[0, 1]):6.3f}],")
            print(f"     [{np.real(measured[1, 0]):6.3f}, {np.real(measured[1, 1]):6.3f}]]")

            # Test evaluation
            test_input = (0.5, 0.5)
            y0, y1 = controller.evaluate(*test_input)
            expected = M @ np.array(test_input)
            print(f"  Test input {test_input}:")
            print(f"    Output: ({y0:.3f}, {y1:.3f})")
            print(f"    Expected: ({expected[0]:.3f}, {expected[1]:.3f})")
        else:
            print(f"  Failed to lock after {controller.state.iteration} iterations")
            print(f"  Final error: {controller.state.error:.2e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_svd_control()
